#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu-h100,gpu-a100
#SBATCH --array=1-5
#SBATCH --signal=TERM@180
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out

srun -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF
source ~/software/init-conda
conda activate .mrs_prediction_env
source .env
EOF

export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
export NCCL_DEBUG=INFO

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

source ~/software/init-conda

conda activate .mrs_prediction_env

source .env

wandb enabled

srun python ${PROJECT_DIR}/mrs_prediction/train_one_imaging_modality_and_clinical_fold.py --config ${1} --fold ${SLURM_ARRAY_TASK_ID}