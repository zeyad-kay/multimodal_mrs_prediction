#!/bin/bash
source .venv/bin/activate

source .env

python ${PROJECT_DIR}/mrs_prediction/evaluate_two_imaging_modality.py --config ${1}
