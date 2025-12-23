#!/bin/bash

source .venv/bin/activate

source .env

wandb enabled

for i in {1..5}; do
    python ${PROJECT_DIR}/mrs_prediction/transop_combination.py --config ${1} --fold ${i} > "transop_combination_${i}.txt"
done