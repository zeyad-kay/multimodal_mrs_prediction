import os
import pandas as pd
from mrs_prediction.dataset.dataset import create_dataset, extract_dataset_statistics
from mrs_prediction.utils import parse_dataset_creation_args
from mrs_prediction.dataset import save_data, split_train_for_cv
import shutil

def train_test_split_escapena1_by_site(df):
    train_sites = [
        "Toronto SMH",
        "Pittsburgh",
        "Ottawa",
        "London",
        "Dresden",
        "Seattle",
        "Melbourne",
        "Saskatoon",
        "Yale",
        "Rhode Island",
        "Montreal CHUM",
        "Chicago",
        "Hamburg",
        "Incheon",
        "Seoul SMC",
        "Seoul YUSH",
        "Winston Salem",
        "Brooklyn",
    ]
    train = df[df.siteid.isin(train_sites)].drop(columns=["siteid"])
    test = df[~df.siteid.isin(train_sites)].drop(columns=["siteid"])
    return train, test

def train_test_split_isles24(df):
    good_outcome = df[df.good_outcome == 1].sample(frac=1, random_state=1)
    bad_outcome = df[df.good_outcome == 0].sample(frac=1, random_state=1)
    good_train, good_test = good_outcome.iloc[:30], good_outcome.iloc[30:]
    bad_train, bad_test = bad_outcome.iloc[:30], bad_outcome.iloc[30:]
    return pd.concat([good_train, bad_train]).sample(frac=1, random_state=1), pd.concat([good_test, bad_test]).sample(frac=1, random_state=1)

def train_test_split(dataset_name, df):
   available_datasets = {
       "escapena1": train_test_split_escapena1_by_site,
       "isles24": train_test_split_isles24
   }
   return available_datasets[dataset_name](df)

if __name__ == "__main__":
    args = parse_dataset_creation_args()

    print(f"Creating {args.dataset_name} dataset from {args.dataset_dir}...")
    df = create_dataset(args.dataset_dir, args.modalities, args.skull_stripped, args.id_column, args.mrs_column, args.include_columns, args.exclude_ids, args.join)

    train_data, test_data = df, None
    if args.test_split:
        print(f"Creating train/test splits...")
        train_data, test_data = train_test_split(args.dataset_name, df)

    print(f"Creating cross validation folds from the training set...")
    cv_splits = split_train_for_cv(train_data, "good_outcome", cv_splits=5, random_state=1)

    print(f"Extracting dataset statistics...")
    statistics = extract_dataset_statistics(train_data.drop(columns=["id","good_outcome",*args.include_columns], errors="ignore"))

    print(f"Cleaning any previous dataset at {os.path.join('data', args.dataset_name)}...")
    shutil.rmtree(os.path.join('data', args.dataset_name),ignore_errors=True)

    print(f"Saving new dataset at {os.path.join('data', args.dataset_name)}...")
    save_data(train_data, cv_splits, test_data, statistics, os.path.join("data", args.dataset_name))
