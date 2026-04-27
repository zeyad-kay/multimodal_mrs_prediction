#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1-00:00:00
#SBATCH --mem=3000GB
#SBATCH --partition=bigmem
#SBATCH -J image_shap
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out

#rm -rf ~/software/miniforge3/envs/.mrs_prediction_env

source ~/software/init-conda
source .env
# conda create -n .mrs_prediction_env python=3.11 -y
conda activate .mrs_prediction_env
# pip install -r requirements.txt
# pip install -e .

export nnUNet_raw="/home/zeyad.abouyoussef/mrs_prediction/data/nnUNet_raw"
export nnUNet_results="/home/zeyad.abouyoussef/mrs_prediction/data/nnUNet_results"
export nnUNet_preprocessed="/home/zeyad.abouyoussef/mrs_prediction/data/nnUNet_preprocessed"


# python ${PROJECT_DIR}/saliency_maps_two_imaging_modalities.py --run ${1} --bg_samples ${2} --wl ${3} --ww ${4}
python ${PROJECT_DIR}/saliency_maps.py --run ${1} --bg_samples ${2} --wl ${3} --ww ${4}

cd ${PROJECT_DIR}/saliency_maps

zip -r ${1}_shap.zip ${1}_shap