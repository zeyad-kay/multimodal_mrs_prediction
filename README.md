# Validating the Benefit of Combining Imaging and Clinical Data for Ischemic Stroke Outcome Prediction

## Data

The data used in this study may be available upon request. The directory structure should look like:
```
data/escapena1/
├── fold_1
│   ├── train.csv
│   ├── val.csv
├── fold_2
│   ├── train.csv
│   ├── val.csv
├── fold_3
│   ├── train.csv
│   ├── val.csv
├── fold_4
│   ├── train.csv
│   ├── val.csv
├── fold_5
│   ├── train.csv
│   ├── val.csv
├── train.csv
├── test.csv
```

## Setup

1. Initialize a virtual environment:

```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
```

2. Install dependencies:
```sh
$ pip install -r requirements.txt
```

3. Create a `.env` file and enter Weights and Biases credentials and the rest of environment variables:
```sh
WANDB_API_KEY=<WANDB_API_KEY>
WANDB_PROJECT=<WANDB_PROJECT_NAME>
WANDB_ENTITY=<WANDB_ENTITY>
PROJECT_DIR=<PATH_TO_PROJECT>
DATASET_DIR=<PATH_TO_DATASET>
```

4. Load environment variables
```sh
$ source .env
```

## Training

1. Run cross validation on clinical data only:

```sh
# Random Forest or Logistic Regression
$ source scripts/train_cv_clinical_sklearn.sh configs/train_config_escapena1_clinical.yaml

# MLP
$ source scripts/train_cv_clinical_torch.sh configs/train_config_escapena1_clinical.yaml
```

All models use the same config file, so comment/uncomment the parameters in the yaml file to run a specific experiment. 

2. Run cross validation on one imaging modality:
```sh
# NCCT
$ source scripts/train_cv_one_imaging_modality.sh configs/train_config_escapena1_ncct.yaml

# CTA
$ source scripts/train_cv_one_imaging_modality.sh configs/train_config_escapena1_cta.yaml
```

3. Run cross validation on one imaging modality and clinical data:
```sh
# NCCT + Clinical
$ source scripts/train_cv_one_imaging_modality_and_clinical.sh configs/train_config_escapena1_ncct_clinical.yaml

# CTA + Clinical
$ source scripts/train_cv_one_imaging_modality.sh configs/train_config_escapena1_cta_clinical.yaml
```

4. Run cross validation on two imaging modalities:
```sh
# NCCT + CTA
$ source scripts/train_cv_two_imaging_modality.sh configs/train_config_escapena1_ncct_cta.yaml
```

5. Run cross validation on two imaging modalities and clinical data:
```sh
# NCCT + CTA + Clinical
$ source scripts/train_cv_two_imaging_modality_and_clinical.sh configs/train_config_escapena1_ncct_cta_clinical.yaml
```

6. Print summary of cross validation statistics for a single experiment:
```sh
$ python summary.py
```

Inside `summary.py`, the list of experiment ids of a cross validation experiment are obtained from wandb. The fold with the best performance is chosen to evaluate on the test set.

## Evaluating

In the evaluation config files, update the checkpoint key to the path of the best model.

Evaluate on the test set and save predictions and metrics to `outputs/` directory:
```sh
# Random Forest or Logistic Regression
$ source scripts/evaluate_clinical_sklearn.sh configs/eval_config_escapena1_clinical.yaml

# MLP
$ source scripts/evaluate_clinical_torch.sh configs/eval_config_escapena1_clinical.yaml

# NCCT
$ source scripts/evaluate_one_imaging_modality.sh configs/eval_config_escapena1_ncct.yaml

# CTA
$ source scripts/evaluate_one_imaging_modality.sh configs/eval_config_escapena1_cta.yaml

# NCCT + CTA
$ source scripts/evaluate_two_imaging_modality.sh configs/eval_config_escapena1_ncct_cta.yaml

# NCCT + Clinical
$ source scripts/evaluate_one_imaging_modality_and_clinical.sh configs/eval_config_escapena1_ncct_clinical.yaml

# CTA + Clinical
$ source scripts/evaluate_one_imaging_modality_and_clinical.sh configs/eval_config_escapena1_cta_clinical.yaml

# NCCT + CTA + Clinical
$ source scripts/evaluate_two_imaging_modality_and_clinical.sh configs/eval_config_escapena1_ncct_cta_clinical.yaml
```

## Explainability

For `LogisticRegression` and `RandomForestClassifier`, we calculate Shapley values and visualize them when running the evaluation scripts.

For the NCCT + Clinical model, we generate a heatmap of the model weights. To generate this heatmap, update the `model_checkpoint` variable in `explainability.py` and run:

```sh
$ python explainability.py
```
