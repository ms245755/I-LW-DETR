#!/usr/bin/env bash
# set -euo pipefail

# Specify resource requirements
#SBATCH --time=100:00:00               # Job time limit (2 hours, change as necessary)
#SBATCH --partition=prismgpup           # Partition for GPU (adjust based on your cluster's partition)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=64G                     # Memory allocation for the job
#SBATCH --gres=gpu:8                  # 8 GPUs per node (adjust based on how many GPUs you need)
#SBATCH --output=./slurms_logs/R-%j-%x.out

# Configure environment variables
source /home/users/tle/miniforge3/etc/profile.d/conda.sh  # Activate conda
conda init
conda activate lwdetr                       # Activate the environment

# Set the number of GPUs and processes per node
export NPROC_PER_NODE=8  
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export BATCH_SIZE=2
export BACKBONE_QUANT_COMPONENTS="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
export ENCODER_QUANT_COMPONENTS="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
export PROJECTOR_QUANT_COMPONENTS="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"

exec bash setup/lwdetr_quant_qat.sh "$@"
