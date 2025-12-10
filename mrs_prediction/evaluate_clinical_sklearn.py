import datetime
import pandas as pd
from mrs_prediction.utils import bootstrap_summary, load_config, parse_eval_args, save_outputs, save_shap_values
from mrs_prediction.model_zoo import load_pickled_model
import os
import dotenv

def validate(model, test_df, features, target, metrics, bootstrap_rounds, ci, random_state):

    inputs, labels = test_df[features], test_df[target].values
    probs = model.predict_proba(inputs)[:,1]
    df = pd.DataFrame({
        "id": test_df["id"],
        f"{target}_prob": probs,
        f"{target}_true": labels
    })

    return bootstrap_summary(bootstrap_rounds, metrics, probs, labels, ci, random_state), df

def main(args):

    now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    
    configs = load_config(args.config)

    experiment_name = configs["experiment_name"]
    random_state = configs["random_state"]
    
    data_params = configs["data"]
    data_path = os.path.join(os.environ["PROJECT_DIR"], data_params["path"])
    train_path = os.path.join(os.environ["PROJECT_DIR"], data_params["train_path"])
    tabular = data_params["tabular"]
    pretty_tabular = data_params["tabular"]
    feature_names_mapping = dict(zip(tabular, pretty_tabular))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(data_path)

    # fu = ["volume"]
    # tabular = data_params["tabular"] + fu
    # clinical = pd.read_csv("/work/souza_lab/Data/ESCAPE-NA1/clinical.csv",usecols=["usubjid"]+fu)

    # print(train_df.shape)
    # print(test_df.shape)
    # test_df = pd.merge(test_df, clinical, left_on="id", right_on="usubjid").drop(columns=["usubjid"]).dropna(subset=fu)
    # train_df = pd.merge(train_df, clinical, left_on="id", right_on="usubjid").drop(columns=["usubjid"]).dropna(subset=fu)
    # print(test_df.shape)
    # print(train_df.shape)

    model_params = configs["model"]
    checkpoint = model_params["checkpoint"]

    model = load_pickled_model(checkpoint)

    tasks = configs["tasks"]
    target = tasks[0]["target"]
    metrics = tasks[0]["metrics"]
    bootstrap_rounds = tasks[0]["bootstrap"]
    confidence_interval = tasks[0]["ci"]

    print("Started evaluating...")

    summary, predictions = validate(model, test_df, tabular, target, metrics, bootstrap_rounds, confidence_interval, random_state)

    save_outputs(predictions, summary, os.path.join("outputs", f'{experiment_name}_{now}'))

    save_shap_values(model, train_df[tabular], test_df[tabular], os.path.join("outputs", f'{experiment_name}_{now}', "shap.png"), feature_names_mapping)

    print("Finished evaluating...")

if __name__ == "__main__":
    dotenv.load_dotenv()
    
    args = parse_eval_args()

    main(args)