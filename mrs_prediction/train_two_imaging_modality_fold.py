import os
import pandas as pd
import wandb
import dotenv
import torch
import time
from mrs_prediction.losses import MultiTaskLoss
from mrs_prediction.utils import parse_train_args, save_checkpoint_to_wandb, log_to_wandb, load_config, calculate_metric
from mrs_prediction.model_zoo import *
from mrs_prediction.dataset import *
from mrs_prediction.transforms import *
from monai.utils.misc import set_determinism
from monai.data.dataloader import DataLoader
import datetime
from mrs_prediction.transforms import WindowCTd
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd, Resized, RandFlipd
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.data.dataset import Dataset

def train(model, criterion, criterion_weights, optimizer, data_loader, log_dict):
    t_s = time.perf_counter()
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        labels = batch["labels"]
        outputs = model(batch["ncct"].cuda(), batch["cta"].cuda())
        _, total_loss = criterion(outputs, labels.cuda().to(torch.float32), criterion_weights)
        total_loss.backward()
        optimizer.step()

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

    cv_fold = args.fold

    set_determinism(configs["random_state"])

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    train_loader_for_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_data = [{"ncct": ncct, "cta": cta, "labels": [labels]} for ncct, cta, labels in val_df[["B0_NCCT","B0_CTA"] + targets].values]
    val_ds = Dataset(data=val_data, transform=img_val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        

    model = init_model(model_name, out_channels=len(tasks))
    
    checkpoint = {}
    if model_checkpoint:
        print(f"Loading checkpoint")
        checkpoint = torch.load(model_checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch')

    model = torch.compile(model)    
    
    model.cuda()

    criterion = MultiTaskLoss(losses, targets).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(lr))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_scheduler_params["factor"])

    if checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    log_dict = {}

    print("Started training...", flush=True)
    for epoch in range(start_epoch + 1, epochs + 1):

        print(f"{datetime.datetime.now()} Epoch {epoch}/{epochs}", flush=True)

        log_dict["train/lr"] = lr_scheduler.get_last_lr()[0]
        log_dict["train/time"] = 0.0

        train(model, criterion, loss_weights, optimizer, train_loader, log_dict)

        print(f"{datetime.datetime.now()} \tFinished training...", flush=True)

        if epoch % epoch_log_interval == 0:
            validate(model, criterion, loss_weights, train_loader_for_eval, "train", log_dict, tasks)
            validate(model, criterion, loss_weights, val_loader, "val", log_dict, tasks)

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
        
        lr_scheduler.step()
    
    save_checkpoint_to_wandb(model_name, model, "torch", run, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, best_metric=best_metric)
    run.finish()

if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_train_args()
    main(args)
