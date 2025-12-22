#!/bin/bash

source .venv/bin/activate

source .env

wandb enabled

for i in {1..5}; do
    nohup python ${PROJECT_DIR}/mrs_prediction/train_clinical_sklearn.py --config ${1} --fold ${i} > "train_clinical_sklearn_${i}.txt" &
done