#!/bin/bash

source .venv/bin/activate

source .env

wandb enabled

for i in {1..5}; do
    python ${PROJECT_DIR}/mrs_prediction/train_two_imaging_modality_fold.py --config ${1} --fold ${i} > "train_two_imaging_modality_fold_${i}.txt"
done