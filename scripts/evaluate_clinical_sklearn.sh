#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH --mem=36GB


#rm -rf ~/software/miniforge3/envs/.mrs_prediction_env

source ~/software/init-conda
source .env
# conda create -n .mrs_prediction_env python=3.11 -y
conda activate .mrs_prediction_env
# pip install -r requirements.txt
# pip install -e .

python ${PROJECT_DIR}/mrs_prediction/evaluate_clinical_sklearn.py --config ${1}
