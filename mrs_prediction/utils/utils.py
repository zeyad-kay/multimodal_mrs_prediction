import argparse
import os
import pickle
import numpy as np
import shap
import torch
import wandb
import yaml
import json
from torcheval.metrics.functional import binary_f1_score, binary_accuracy, binary_auroc, binary_precision, binary_recall, mean_squared_error, binary_auprc, binary_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def parse_train_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-f", "--fold", type=int, required=True, help="Fold number for cross-validation")

    return parser.parse_args()

def parse_eval_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")

    return parser.parse_args()

def parse_download_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--runs", nargs="+", required=True, help="Ids of W&B runs from which to download the models")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save the model in")

    return parser.parse_args()

def parse_dataset_creation_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_name", type=str, required=True, help="Arbitrary dataset name. This name will be used when creating the training and testing csv files in the data directory")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to directory containing the dataset. The directory must contain a clinical.csv file containing the clinical data and a separate directory containing a modalities' nii.gz files (e.g., NCCT/subj_1/subj_1.nii.gz)")
    parser.add_argument("--modalities", nargs="+", required=True, help="List of imaging modalities")
    parser.add_argument("--id_column", type=str, required=True, help="Column name corresponding to subject id")
    parser.add_argument("--mrs_column", type=str, required=True, help="Column name corresponding to mRS")
    parser.add_argument("--include_columns", nargs="+", default=[], required=False, help="Extra columns to include (e.g., age)")
    parser.add_argument("--exclude_ids", nargs="+", default=[], required=False, help="Ids to remove from the dataset")
    parser.add_argument("--join", type=str, required=True, help="How to merge subjects across modalities, should either be 'inner' or 'outer'")
    parser.add_argument("--test_split", action=argparse.BooleanOptionalAction, help="Split the datset to train/test prior to creating cross validation folds or create the folds using the full data")
    parser.add_argument("--skull_stripped", action=argparse.BooleanOptionalAction, help="Use skull stripped scans instead of raw ones")

    return parser.parse_args()

def load_config(cfg_path):
    config = {}
    with open(cfg_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def mean_absolute_error(pred, true):
    return torch.mean(torch.abs(pred - true))

def calculate_metric(pred, true, metric):
    metric_dict = {
        "accuracy": binary_accuracy,
        "f1": binary_f1_score,
        "auroc": binary_auroc,
        "auprc": binary_auprc,
        "precision": binary_precision,
        "recall": recall,
        "specificity": specificity,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error
    }
    return metric_dict[metric](pred, true).item()

def specificity(pred, true):
    tn, fp = binary_confusion_matrix(pred, true.to(int))[0]
    return tn / (tn + fp)

def recall(pred, true):
    return binary_recall(pred, true.to(torch.int))

def log_to_wandb(wandb_run, metrics, epoch):
    wandb_run.log(metrics, step=epoch)

def save_checkpoint_to_wandb(name, model, model_type, wandb_run, **kwargs):
    if model_type == "torch":
        optimizer = kwargs["optimizer"]
        lr_scheduler = kwargs["lr_scheduler"]
        epoch = kwargs["epoch"]
        best_metric = kwargs["best_metric"]
        checkpoint = {
            "name": name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        }
        name = f"{name}.tar"
        torch.save(checkpoint, os.path.join(wandb_run.dir, name))
    elif model_type == "sklearn":
        name = f"{name}.sav"
        with open(os.path.join(wandb_run.dir, name), 'wb') as file:
            pickle.dump(model, file)
    else:
        raise NotImplementedError(f"Saving {model_type} models is not supported")

    wandb_run.log_artifact(os.path.join(wandb_run.dir, name), name=name, type="model")

def download_models_from_wandb(runs_ids, save_dir):
    print(f"Saving models in {save_dir}")

    for run_id in runs_ids:
        os.makedirs(os.path.join(save_dir, run_id), exist_ok=True)
        api = wandb.Api()

        models = [artifact for artifact in api.run(run_id).logged_artifacts() if artifact.type == "model"]

        # best model is always the last created moodel
        best_model = sorted(models, key=lambda m: m.created_at, reverse=True)[0]
        best_model.download(root=os.path.join(save_dir, run_id))

def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def save_outputs(output_df, metrics, path):
    os.makedirs(path, exist_ok=True)
    write_json(metrics, os.path.join(path, "metrics.json"))
    if isinstance(output_df, pd.DataFrame):
        output_df.to_csv(os.path.join(path, "predictions.csv"), index=False)

def load_json(pth):
    config = {}
    with open(pth) as stream:
        try:
            config = json.load(stream)
        except json.JSONDecodeError as exc:
            print(exc)
    return config

def bootstrap_summary(rounds, metrics, predictions, true, ci=95, random_state=None):

    rng = np.random.RandomState(seed=random_state)

    alpha = (100 - ci) / 2
    lower_ci = alpha
    upper_ci = round(100 - alpha, 1)

    metrics_samples = {m:[] for m in metrics}
    summary = {}

    idx = np.arange(predictions.shape[0])
    for _ in range(rounds):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        if len(np.unique(true[pred_idx])) < 2:
            continue
        for metric in metrics:
            metrics_samples[metric].append(calculate_metric(torch.tensor(predictions[pred_idx]), torch.tensor(true[pred_idx]), metric))
        
    for metric in metrics:
        summary[metric] = { 
            "mean": np.mean(metrics_samples[metric]).round(2),
            "lower_ci": np.percentile(metrics_samples[metric], lower_ci).round(2),
            "upper_ci": np.percentile(metrics_samples[metric], upper_ci).round(2)
        }

    return summary

def save_shap_values(model, x_train, x_test, file_name, feature_names_mapping):

    explainer = shap.Explainer(model.predict_proba, x_train)

    shap_values = explainer(x_test)

    plt.figure(dpi=600)

    shap.plots.beeswarm(shap_values[:,:,1], max_display=x_train.shape[1])
    ax = plt.gca()

    yticklabels = [feature_names_mapping[label.get_text()] for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels)

    plt.tight_layout()

    plt.savefig(file_name)

    return