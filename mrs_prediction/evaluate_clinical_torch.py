import datetime
import os
import pandas as pd
import dotenv
import torch
from mrs_prediction.utils import parse_eval_args, save_outputs, load_config, bootstrap_summary
from mrs_prediction.model_zoo import *
from mrs_prediction.dataset import *
from mrs_prediction.transforms import *
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.data.dataset import Dataset
from monai.utils.misc import set_determinism

def validate(model, data_loader, tasks, metrics, bootstrap_rounds, ci, random_state):
    model.eval()
    true = torch.empty((len(data_loader.dataset), len(tasks)))
    logits = torch.empty((len(data_loader.dataset), len(tasks)))
    ids = torch.empty((len(data_loader.dataset), 1))
    idx = 0
    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"]
            batch_ids = labels[:,0:1]
            tabular = labels[:,2:]
            labels = labels[:,1:2].to(torch.float32)
            outputs = model(tabular.to(torch.float32))
            true[idx:idx+labels.shape[0]] = labels
            logits[idx:idx+labels.shape[0]] = outputs
            ids[idx:idx+labels.shape[0]] = batch_ids
            idx += labels.shape[0]

    task_name = tasks[0]["target"]
    
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits)

    df = pd.DataFrame({
        "id": ids.numpy(force=True).astype(int)[:,0],
        f"{task_name}_prob": probs.numpy(force=True)[:,0],
        f"{task_name}_true": true.numpy(force=True)[:,0]
    })
    return bootstrap_summary(bootstrap_rounds, metrics, probs[:,0], true[:,0], ci, random_state), df

def main(args):
    now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    configs = load_config(args.config)

    experiment_name = configs["experiment_name"]
    random_state = configs["random_state"]
    
    set_determinism(configs["random_state"])

    data_params = configs["data"]
    data_path = os.path.join(os.environ["PROJECT_DIR"], data_params["path"])
    tabular = data_params["tabular"]

    test_df = pd.read_csv(data_path)

    # fu = ["volume","cl_aspects_24h"]
    # tabular = data_params["tabular"] + fu
    # clinical = pd.read_csv("/work/souza_lab/Data/ESCAPE-NA1/clinical.csv",usecols=["usubjid"]+fu)

    # test_df = pd.merge(test_df, clinical, left_on="id", right_on="usubjid").drop(columns=["usubjid"]).dropna(subset=fu)

    model_params = configs["model"]
    checkpoint = model_params["checkpoint"]

    tasks = configs["tasks"]
    target = tasks[0]["target"]
    metrics = tasks[0]["metrics"]
    bootstrap_rounds = tasks[0]["bootstrap"]
    confidence_interval = tasks[0]["ci"]


    test_data = [{"labels": labels} for labels in test_df[["id", target] + tabular].values]

    test_transform = Compose([
        ToTensord(keys=["labels"])
    ], lazy=True)

    test_ds = Dataset(data=test_data, transform=test_transform)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = init_model("MLP", in_features=len(tabular))
    
    if checkpoint:
        # checkpoint = torch.load(os.path.join(run.dir, f"{model_checkpoint}.tar"), weights_only=True, map_location=local_rank)
        checkpoint = torch.load(checkpoint, weights_only=True, map_location="cpu")
        checkpoint['model_state_dict'] = {k.replace("_orig_mod.",""):v for k,v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(checkpoint['model_state_dict'])

    print("Started evaluating...")

    summary, predictions = validate(model, test_loader, tasks, metrics, bootstrap_rounds, confidence_interval, random_state)

    save_outputs(predictions, summary, os.path.join("outputs", f'{experiment_name}_{now}'))

    print("Finished evaluating...")

if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_eval_args()
    main(args)
