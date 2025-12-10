#!/bin/bash
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=128GB

#rm -rf ~/software/miniforge3/envs/.mrs_prediction_env

source ~/software/init-conda
source .env
# conda create -n .mrs_prediction_env python=3.11 -y
conda activate .mrs_prediction_env
# pip install -r requirements.txt
# pip install -e .

python ${PROJECT_DIR}/mrs_prediction/evaluate_one_imaging_modality_and_clinical.py --config ${1}

