#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --array=1-5
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out

#rm -rf ~/software/miniforge3/envs/.mrs_prediction_env

source ~/software/init-conda
source .env
# conda create -n .mrs_prediction_env python=3.11 -y
conda activate .mrs_prediction_env
# pip install -r requirements.txt
# pip install -e .

wandb enabled

python ${PROJECT_DIR}/mrs_prediction/train_clinical_sklearn.py --config ${1} --fold ${SLURM_ARRAY_TASK_ID}
