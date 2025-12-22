#!/bin/bash

source .venv/bin/activate

source .env

wandb enabled

for i in {1..5}; do
    nohup python ${PROJECT_DIR}/mrs_prediction/train_one_imaging_modality_and_clinical_fold.py --config ${1} --fold ${i} > "train_one_imaging_modality_and_clinical_fold_${i}.txt" &
done