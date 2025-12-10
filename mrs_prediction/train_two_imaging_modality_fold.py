import os
import pandas as pd
import wandb
import dotenv
import torch
import time
import signal
from mrs_prediction.losses import MultiTaskLoss
from mrs_prediction.utils import parse_train_args, save_checkpoint_to_wandb, log_to_wandb, load_config, calculate_metric
from mrs_prediction.model_zoo import *
from mrs_prediction.dataset import *
from mrs_prediction.transforms import *
from monai.utils.misc import set_determinism
import torch.distributed as dist
import torch.utils.data.distributed
from monai.data.dataloader import DataLoader
from monai.data.image_dataset import ImageDataset
import datetime
from datetime import timedelta
import socket
import random
from mrs_prediction.transforms import WindowCTd
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd, Resized, RandFlipd
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.data.dataset import Dataset

def handle_slurm_timeout(signal, frame, run, model_name, model, optimizer, lr_scheduler, log_dict, best_metric):

    if run:
        save_checkpoint_to_wandb(f"{model_name}_latest", model, "torch", run, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=run.step, best_metric=log_dict[best_metric])
        run.finish()

    dist.barrier()

    dist.destroy_process_group()

    print("Exitting gracefully")

    exit()

def train(model, criterion, criterion_weights, optimizer, data_loader, rank, log_dict):
    t_s = time.perf_counter()
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        labels = batch["labels"]
        outputs = model(batch["ncct"].cuda(), batch["cta"].cuda())
        _, total_loss = criterion(outputs, labels.cuda().to(torch.float32), criterion_weights)
        total_loss.backward()
        optimizer.step()

    if rank == 0:
        log_dict["train/time"] = round(time.perf_counter() - t_s, 3)
    return

def validate(model, criterion, criterion_weights, data_loader, data_type, log_dict, tasks):
    model.eval()
    true = torch.empty((len(data_loader.dataset), len(tasks))).cuda()
    logits = torch.empty((len(data_loader.dataset), len(tasks))).cuda()
    idx = 0
    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"]
            outputs = model(batch["ncct"].cuda(), batch["cta"].cuda())
            true[idx:idx+labels.shape[0]] = labels.cuda()
            logits[idx:idx+labels.shape[0]] = outputs
            idx += labels.shape[0]

    tasks_loss, total_loss = criterion(logits, true, criterion_weights)
    log_dict[f"{data_type}/loss"] = total_loss.item()

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits)

    for i,task in enumerate(tasks):
        task_name = task["target"]
        log_dict[f"{data_type}/{task_name}/loss"] = tasks_loss[i][1]

        for metric in task["metrics"]:
            log_dict[f"{data_type}/{task_name}/{metric}"] = calculate_metric(probs[:,i], true[:,i], metric)

    return

def main(args):
    configs = load_config(args.config)

    local_rank = int(os.environ.get("SLURM_LOCALID") or 0)
    global_rank = int(os.environ.get("SLURM_PROCID") or 0)
    current_device = local_rank

    torch.cuda.set_device(current_device)


    """ this block initializes a process group and initiate communications
    between all processes running on all nodes """

    print(f'From Rank: {global_rank}, ==> Initializing Process Group...')

    #init the process group
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1.0), init_method=f"tcp://{os.environ.get('MASTER_ADDR') or socket.gethostname()}:{random.randint(1000,10000)}", world_size=int(os.environ.get('SLURM_NTASKS_PER_NODE') or 1) * int(os.environ.get('SLURM_JOB_NUM_NODES') or 1), rank=global_rank)
    print("process group ready!")

    print(f'From Rank: {global_rank}, ==> Making model..')

    dist.barrier()

    cv_fold = args.fold

    set_determinism(configs["random_state"])

    run = None
    if global_rank == 0:
        run = wandb.init(name=configs["experiment_name"] + f"_{cv_fold}", resume="allow", config=configs)

    train_params = configs["hyperparameters"]
    epochs = train_params["epochs"]
    epoch_log_interval = train_params["log_interval"]
    batch_size = train_params["batch_size"]
    lr = train_params["learning_rate"]
    early_stopping = train_params["early_stopping"]
    early_stopping_patience = early_stopping["patience"]
    early_stopping_target = early_stopping["target"]
    early_stopping_metric = early_stopping["metric"]
    early_stopping_goal = early_stopping["goal"]
    early_stopping_min_improvement = early_stopping["min_improvement"]
    lr_scheduler_params = train_params["learning_rate_scheduler"]

    data_params = configs["data"]
    modality = data_params["modality"]
    data_path = os.path.join(os.environ["PROJECT_DIR"], data_params["path"])
    fold_path = os.path.join(data_path, f"fold_{cv_fold}")

    tasks = configs["tasks"]
    losses = []
    loss_weights = []
    targets = []
    for task in tasks:
        targets.append(task["target"])
        losses.append(task["loss"])
        loss_weights.append(task["weight"])

    loss_weights = torch.tensor(loss_weights, dtype=torch.float32).cuda()

    model_params = configs["model"]
    model_name = model_params["name"]
    model_checkpoint = model_params["checkpoint"]

    ncct_preprocessing_params = configs["preprocessing"]["B0_NCCT"]
    cta_preprocessing_params = configs["preprocessing"]["B0_CTA"]

    ncct_spacing = ncct_preprocessing_params["spacing"]
    ncct_shape = ncct_preprocessing_params["size"]
    ncct_wl, ncct_ww = ncct_preprocessing_params["windowing"]["level"], ncct_preprocessing_params["windowing"]["width"]
    ncct_mean_intensity, ncct_std_intesity = ncct_preprocessing_params["normalize"]["mean"], ncct_preprocessing_params["normalize"]["std"]

    cta_spacing = cta_preprocessing_params["spacing"]
    cta_shape = cta_preprocessing_params["size"]
    cta_wl, cta_ww = cta_preprocessing_params["windowing"]["level"], cta_preprocessing_params["windowing"]["width"]
    cta_mean_intensity, cta_std_intesity = cta_preprocessing_params["normalize"]["mean"], cta_preprocessing_params["normalize"]["std"]
    

    bad_epochs = 0
    start_epoch = 0
    best_metric = 0.0 if early_stopping_goal == "max" else float('inf')
    converged = False

    # Define transforms
    # train_transform = get_train_transforms(modality, window_level=window_level, window_width=window_width, voxel_spacing=voxel_spacing, image_size=image_size, mean_intensity=mean_intensity, std_intensity=std_intensity)
    # val_transform = get_test_transforms(modality, window_level=window_level, window_width=window_width, voxel_spacing=voxel_spacing, image_size=image_size, mean_intensity=mean_intensity, std_intensity=std_intensity)

    train_df, val_df = pd.read_csv(os.path.join(fold_path, "train.csv")), pd.read_csv(os.path.join(fold_path, "val.csv"))

    train_data = [{"ncct": ncct, "cta": cta, "labels": [labels]} for ncct, cta, labels in train_df[["B0_NCCT","B0_CTA"] + targets].values]

    img_train_transform = Compose([
        LoadImaged(keys=["ncct","cta"], image_only=False),
        EnsureChannelFirstd(keys=["ncct","cta"]),
        Orientationd(keys=["ncct","cta"],axcodes="RAS"),
        WindowCTd(keys=["ncct"], level=ncct_wl, width=ncct_ww), 
        WindowCTd(keys=["cta"], level=cta_wl, width=cta_ww), 
        CropForegroundd(keys=["ncct"],source_key="ncct",select_fn=lambda x: x > 0, margin=0),
        CropForegroundd(keys=["cta"],source_key="cta",select_fn=lambda x: x > 0, margin=0),
        Spacingd(keys=["ncct"],pixdim=ncct_spacing),
        Spacingd(keys=["cta"],pixdim=cta_spacing),
        Resized(keys=["ncct"],spatial_size=ncct_shape, mode="bilinear"),
        Resized(keys=["cta"],spatial_size=cta_shape, mode="bilinear"),
        NormalizeIntensityd(keys=["ncct"],subtrahend=torch.tensor([ncct_mean_intensity]), divisor=torch.tensor([ncct_std_intesity])),
        NormalizeIntensityd(keys=["cta"],subtrahend=torch.tensor([cta_mean_intensity]), divisor=torch.tensor([cta_std_intesity])),
        RandFlipd(keys=["ncct","cta"],prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["ncct","cta"],prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["ncct","cta"],prob=0.5, spatial_axis=[2]),
        ToTensord(keys=["labels"])
    ], lazy=True)

    img_val_transform = Compose([
        LoadImaged(keys=["ncct","cta"], image_only=False),
        EnsureChannelFirstd(keys=["ncct","cta"]),
        Orientationd(keys=["ncct","cta"],axcodes="RAS"),
        WindowCTd(keys=["ncct"], level=ncct_wl, width=ncct_ww), 
        WindowCTd(keys=["cta"], level=cta_wl, width=cta_ww), 
        CropForegroundd(keys=["ncct"],source_key="ncct",select_fn=lambda x: x > 0, margin=0),
        CropForegroundd(keys=["cta"],source_key="cta",select_fn=lambda x: x > 0, margin=0),
        Spacingd(keys=["ncct"],pixdim=ncct_spacing),
        Spacingd(keys=["cta"],pixdim=cta_spacing),
        Resized(keys=["ncct"],spatial_size=ncct_shape, mode="bilinear"),
        Resized(keys=["cta"],spatial_size=cta_shape, mode="bilinear"),
        NormalizeIntensityd(keys=["ncct"],subtrahend=torch.tensor([ncct_mean_intensity]), divisor=torch.tensor([ncct_std_intesity])),
        NormalizeIntensityd(keys=["cta"],subtrahend=torch.tensor([cta_mean_intensity]), divisor=torch.tensor([cta_std_intesity])),
        ToTensord(keys=["labels"])
    ], lazy=True)

    train_ds = Dataset(data=train_data, transform=img_train_transform)


    # train_ds = ImageDataset(image_files=train_df[modality], labels=train_df[targets].values, transform=train_transform, image_only=False) # type: ignore
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=2, pin_memory=True)


    if global_rank == 0:
        train_loader_for_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        val_data = [{"ncct": ncct, "cta": cta, "labels": [labels]} for ncct, cta, labels in val_df[["B0_NCCT","B0_CTA"] + targets].values]
        val_ds = Dataset(data=val_data, transform=img_val_transform)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        

    model = init_model(model_name, out_channels=len(tasks))
    
    checkpoint = {}
    if model_checkpoint:
        print(f"Loading checkpoint for device {local_rank}")
        # checkpoint = torch.load(os.path.join(run.dir, f"{model_checkpoint}.tar"), weights_only=True, map_location=local_rank)
        checkpoint = torch.load(model_checkpoint, weights_only=True, map_location="cpu")
        checkpoint['model_state_dict'] = {k.replace("module._orig_mod.",""):v for k,v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch')
        # best_metric = checkpoint.get('best_metric')

        # for m in model.parameters(True):
        #     m.requires_grad = False
        # for m in model.last_linear.parameters(True):
        #     m.requires_grad = True

    model = torch.compile(model)    
    
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

    # checkpoint = torch.load("/home/zeyad.abouyoussef/mrs_prediction/Barlow.tar", weights_only=True, map_location="cpu")


    # encoder_state_dict = {}
    # for k,v in checkpoint["model_state_dict"].items():
    #     if k.startswith("backbone."):
    #         encoder_state_dict[k.replace("backbone.","")] = v

    # print(model.backbone.load_state_dict(encoder_state_dict, strict=False))

    # model.backbone.load_state_dict(checkpoint['model_state_dict'])

    # for m in model.backbone.parameters(True):
    #     m.requires_grad = False
    

    #model.load_mae_encoder_weights(torch.load(os.path.join("/home/zeyad.abouyoussef/mrs_prediction/", "LabelGuidedMAE.tar"), weights_only=True, map_location='cpu'))
    
    #for m in model.parameters(True):
    #    m.requires_grad = False
    #for m in model.classification_head.parameters(True):
    #    m.requires_grad = True


    criterion = MultiTaskLoss(losses, targets).cuda()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(lr))
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=early_stopping_goal, threshold=lr_scheduler_params["threshold"], threshold_mode="abs", patience=lr_scheduler_params["patience"], factor=lr_scheduler_params["factor"])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_scheduler_params["factor"])

    if checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    log_dict = {}

    signal.signal(signal.SIGTERM, lambda signal, frame, run=run, model_name=model_name, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, log_dict=log_dict, best_metric=f"val/{early_stopping_target}/{early_stopping_metric}": handle_slurm_timeout(signal, frame, run, model_name, model, optimizer, lr_scheduler, log_dict, best_metric))

    print("Started training...", flush=True)
    for epoch in range(start_epoch + 1, epochs + 1):
        train_sampler.set_epoch(epoch)

        print(f"{datetime.datetime.now()} Epoch {epoch}/{epochs}", flush=True)

        log_dict["train/lr"] = lr_scheduler.get_last_lr()[0]
        log_dict["train/time"] = 0.0

        train(model, criterion, loss_weights, optimizer, train_loader, global_rank, log_dict)

        print(f"{datetime.datetime.now()} \tFinished training in {global_rank}...", flush=True)

        dist.barrier()
        if global_rank == 0 and epoch % epoch_log_interval == 0:
            validate(model.module, criterion, loss_weights, train_loader_for_eval, "train", log_dict, tasks)
            validate(model.module, criterion, loss_weights, val_loader, "val", log_dict, tasks)

            print(f"{datetime.datetime.now()} \tFinished validating...", flush=True)

            if early_stopping_goal == "max":
                improvement = (log_dict[f"val/{early_stopping_target}/{early_stopping_metric}"]) > (best_metric + early_stopping_min_improvement)
            else:
                improvement = (log_dict[f"val/{early_stopping_target}/{early_stopping_metric}"] + early_stopping_min_improvement) < best_metric
            
            if improvement:
                best_metric = log_dict[f"val/{early_stopping_target}/{early_stopping_metric}"]
                bad_epochs = 0

                print("\tSaving best model...")
                save_checkpoint_to_wandb(model_name, model, "torch", run, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, best_metric=best_metric)

                if best_metric == 0.0:
                    converged = True
                    print(f"{datetime.datetime.now()} \t{early_stopping_target} {early_stopping_metric} {early_stopping_goal}d...")
            else:
                bad_epochs += 1
                if bad_epochs == early_stopping_patience:
                    converged = True
                    print("\tEarly stopping triggered...")

            print("\tLogging log_dict to wandb...")
            log_to_wandb(run, log_dict, epoch)

            if converged:
                break
        
        dist.barrier()

        # lr_scheduler.step(log_dict[f"val/{early_stopping_target}/{early_stopping_metric}"])
        lr_scheduler.step()
    
    if global_rank == 0 and run:
        save_checkpoint_to_wandb(model_name, model, "torch", run, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, best_metric=best_metric)
        run.finish()

    dist.barrier()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_train_args()
    main(args)
