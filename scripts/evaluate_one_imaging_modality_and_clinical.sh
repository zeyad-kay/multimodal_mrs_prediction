#!/bin/bash

source .venv/bin/activate

source .env

python ${PROJECT_DIR}/mrs_prediction/evaluate_one_imaging_modality_and_clinical.py --config ${1}

