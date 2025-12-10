import datetime
import os
import pandas as pd
import dotenv
import torch
from mrs_prediction.utils import parse_eval_args, load_config, save_outputs, bootstrap_summary
from mrs_prediction.model_zoo import *
from mrs_prediction.dataset import *
from mrs_prediction.transforms import *
from monai.utils.misc import set_determinism

def validate(model, data_loader, device, tasks, metrics, bootstrap_rounds, ci, random_state):
    model.eval()
    true = torch.empty((len(data_loader.dataset), len(tasks))).to(device)
    logits = torch.empty((len(data_loader.dataset), len(tasks))).to(device)
    file_paths = []
    idx = 0
    with torch.no_grad():
        for inputs, labels, metadata in data_loader:
            tabular = labels[:,1:]
            labels = labels[:,0:len(tasks)]
            outputs = model(inputs.to(device), tabular.to(device).to(torch.float32))
            true[idx:idx+labels.shape[0]] = labels[:,0:len(tasks)].to(device)
            logits[idx:idx+labels.shape[0]] = outputs
            file_paths[idx:idx+labels.shape[0]] = metadata["filename_or_obj"]
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
    modality = data_params["modality"]
    tabular = data_params["tabular"]
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

    preprocessing_params = configs["preprocessing"]
    voxel_spacing = preprocessing_params["spacing"]
    image_size = preprocessing_params["size"]
    window_level, window_width = preprocessing_params["windowing"]["level"], preprocessing_params["windowing"]["width"]
    mean_intensity, std_intensity = preprocessing_params["normalize"]["mean"], preprocessing_params["normalize"]["std"]

    test_transform = get_test_transforms(modality, window_level=window_level, window_width=window_width, voxel_spacing=voxel_spacing, image_size=image_size, mean_intensity=mean_intensity, std_intensity=std_intensity)

    test_df = pd.read_csv(data_path)
    test_loader = get_dataloader(test_df, modality, targets+tabular, batch_size=batch_size, transform=test_transform, num_workers=2)

    model = init_model(model_name, out_channels=len(tasks))

    if model_checkpoint:
        print("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(os.environ["PROJECT_DIR"], model_checkpoint), weights_only=True, map_location=device)
        checkpoint["model_state_dict"]={k.replace("module._orig_mod.",""):v for k,v in checkpoint["model_state_dict"].items()}
        # checkpoint["model_state_dict"]={k.replace("module.",""):v for k,v in checkpoint["model_state_dict"].items()}
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