import datetime
import os
import pandas as pd
import dotenv
import torch
from mrs_prediction.utils import parse_eval_args, load_config, save_outputs, bootstrap_summary
from mrs_prediction.model_zoo import *
from mrs_prediction.dataset import *
from mrs_prediction.transforms import *
from mrs_prediction.transforms import WindowCTd
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd, Resized
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from monai.utils.misc import set_determinism

def validate(model, data_loader, device, tasks, metrics, bootstrap_rounds, ci, random_state):
    model.eval()
    true = torch.empty((len(data_loader.dataset), len(tasks))).to(device)
    logits = torch.empty((len(data_loader.dataset), len(tasks))).to(device)
    file_paths = []
    idx = 0
    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"]
            outputs = model(batch["ncct"].to(device), batch["cta"].to(device))
            true[idx:idx+labels.shape[0]] = labels.to(device)
            logits[idx:idx+labels.shape[0]] = outputs
            file_paths[idx:idx+labels.shape[0]] = batch["ncct_meta_dict"]["filename_or_obj"]
            idx += labels.shape[0]
            print(idx, flush=True)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits)

    df = pd.DataFrame({"img_path": file_paths})
    for i,task in enumerate(tasks):
        task_name = task["target"]
        df[f"{task_name}_prob"] = probs[:, i].numpy(force=True)
        df[f"{task_name}_true"] = true[:, i].numpy(force=True)

    return bootstrap_summary(bootstrap_rounds, metrics, probs[:,0], true[:,0], ci, random_state), df

def main(args):
    now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    
    configs = load_config(args.config)
    experiment_name = configs["experiment_name"]
    batch_size = configs["batch_size"]
    random_state = configs["random_state"]

    set_determinism(random_state)

    data_params = configs["data"]
    data_path = os.path.join(os.environ["PROJECT_DIR"], data_params["path"])
    
    tasks = configs["tasks"]
    metrics = tasks[0]["metrics"]
    bootstrap_rounds = tasks[0]["bootstrap"]
    confidence_interval = tasks[0]["ci"]
    targets = []
    for task in tasks:
        targets.append(task["target"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    

    test_df = pd.read_csv(data_path)

    test_data = [{"ncct": ncct, "cta": cta, "labels": [labels]} for ncct, cta, labels in test_df[["B0_NCCT","B0_CTA"] + targets].values]

    test_transform = Compose([
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


    train_ds = Dataset(data=test_data, transform=test_transform)

    test_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = init_model(model_name, out_channels=len(tasks))

    if model_checkpoint:
        print("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(os.environ["PROJECT_DIR"], model_checkpoint), weights_only=True, map_location=device)
        checkpoint["model_state_dict"]={k.replace("module._orig_mod.",""):v for k,v in checkpoint["model_state_dict"].items()}
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("Started evaluating...")

    summary, predictions = validate(model, test_loader, device, tasks, metrics, bootstrap_rounds, confidence_interval, random_state)

    save_outputs(predictions, summary, os.path.join("outputs", f'{experiment_name}_{now}'))

    print("Finished evaluating...")

if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_eval_args()
    main(args)