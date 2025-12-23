from matplotlib import pyplot as plt
import wandb
import dotenv
import pandas as pd

if __name__ == "__main__":
    dotenv.load_dotenv()

    api = wandb.Api()

    experiments = ["fysml9pj","buaedt9w","oa1ywy5h","kovo8oxl","jiq1job2"]
    
    runs = api.runs(filters={"name": {"$in": experiments}}, per_page=5)
    print(len(runs))
    
    BEST_METRIC = "f1"

    metric_keys = ["train/good_outcome/auroc","val/good_outcome/auroc","train/good_outcome/auprc","val/good_outcome/auprc","train/good_outcome/precision","val/good_outcome/precision","train/good_outcome/f1", "val/good_outcome/f1"]

    df = pd.concat({run.name: run.history(keys=metric_keys) for run in runs}, join="outer", names=["run_name","idx"]).reset_index("idx",drop=True).reset_index()
    df["fold"] = df["run_name"].str.extract(r'_(\d+)$').astype(int)
    df = df.drop(columns=["run_name"])

    best_epochs = []
    for fold in range(1,len(runs)+1):
        df_fold = df[df["fold"] == fold]
        best_step = df_fold[f"val/good_outcome/{BEST_METRIC}"].argmax()
        max_perf = df_fold.iloc[best_step:best_step+1]
        best_epochs.append(max_perf)
        print(f"{fold=}, {best_step=}, max_{BEST_METRIC}={max_perf[f'val/good_outcome/{BEST_METRIC}'].round(2).values[0]}")

    df = pd.concat(best_epochs)

    print(df[metric_keys].describe().round(2)*100)