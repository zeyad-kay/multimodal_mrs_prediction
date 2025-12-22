#!/bin/bash

source .venv/bin/activate

source .env

python ${PROJECT_DIR}/mrs_prediction/evaluate_clinical_sklearn.py --config ${1}
