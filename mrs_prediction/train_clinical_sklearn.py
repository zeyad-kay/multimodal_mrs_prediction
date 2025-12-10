import pandas as pd
import torch
from mrs_prediction.utils import calculate_metric, load_config, parse_train_args, log_to_wandb, save_checkpoint_to_wandb
from mrs_prediction.model_zoo import init_model
import os
import dotenv
import wandb

def main(args):
    configs = load_config(args.config)

    cv_fold = args.fold
    with wandb.init(name=configs["experiment_name"] + f"_{cv_fold}", resume="allow", config=configs) as run:
        random_state = configs["random_state"]

        train_params = configs["hyperparameters"]

        data_params = configs["data"]
        data_path = os.path.join(os.environ["PROJECT_DIR"], data_params["path"])
        tabular = data_params["tabular"]

        fold_path = os.path.join(data_path, f"fold_{cv_fold}")

        train_df, val_df = pd.read_csv(os.path.join(fold_path, "train.csv")), pd.read_csv(os.path.join(fold_path, "val.csv"))


        # fu = []
        fu = ["volume"]
        tabular = data_params["tabular"] + fu
        clinical = pd.read_csv("/work/souza_lab/Data/ESCAPE-NA1/clinical.csv",usecols=["usubjid"]+fu)

        print(train_df.shape)
        print(val_df.shape)

        train_df = pd.merge(train_df, clinical, left_on="id", right_on="usubjid").drop(columns=["usubjid"]).dropna(subset=fu)
        val_df = pd.merge(val_df, clinical, left_on="id", right_on="usubjid").drop(columns=["usubjid"]).dropna(subset=fu)

        print(train_df.shape)
        print(val_df.shape)

        # train_df["age"] = train_df["age"] / 103.8
        # train_df["mrs_BL"] = train_df["mrs_BL"] / 6.0
        # train_df["onset_time_to_treatment"] = train_df["onset_time_to_treatment"] / 1652.0
        # train_df["baseline_nihss"] = train_df["baseline_nihss"] / 42.0
        # train_df["baseline_aspects"] = train_df["baseline_aspects"] / 10.0
        
        # val_df["age"] = val_df["age"] / 103.8
        # val_df["mrs_BL"] = val_df["mrs_BL"] / 6.0
        # val_df["onset_time_to_treatment"] = val_df["onset_time_to_treatment"] / 1652.0
        # val_df["baseline_nihss"] = val_df["baseline_nihss"] / 42.0
        # val_df["baseline_aspects"] = val_df["baseline_aspects"] / 10.0



        model_params = configs["model"]
        model_name = model_params["name"]

        model = init_model(model_name, random_state=random_state, **train_params)

        task = configs["tasks"][0]
        target = task["target"]
        metrics = task["metrics"]

        log_dict = {}

        print("Started training...")

        model.fit(train_df[tabular], train_df[target])

        print("Finished training...")

        for metric in metrics:
            train_metric = calculate_metric(torch.tensor(model.predict_proba(train_df[tabular])[:,1]), torch.tensor(train_df[target].values), metric)
            val_metric = calculate_metric(torch.tensor(model.predict_proba(val_df[tabular])[:,1]), torch.tensor(val_df[target].values), metric)
            log_dict[f"train/{target}/{metric}"] = train_metric
            log_dict[f"val/{target}/{metric}"] = val_metric
        
        print("Finished validating...")

        save_checkpoint_to_wandb(model_name, model, "sklearn", run)

        print("Logging log_dict to wandb...")
        
        log_to_wandb(run, log_dict, None)

if __name__ == "__main__":
    dotenv.load_dotenv()
    
    args = parse_train_args()

    main(args)