#!/usr/bin/env bash

# Specify resource requirements
#SBATCH --time=40:00:00               # Job time limit (2 hours, change as necessary)
#SBATCH --partition=gpu40G           # Partition for GPU (adjust based on your cluster's partition)
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
export NPROC_PER_NODE=8  # This should match the number of GPUs you're using (A100 has 8 GPUs)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun python -u -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --use_env \
    main.py \
    --lr 1e-4 \
    --lr_encoder 1.5e-4 \
    --batch_size 8 \
    --weight_decay 1e-4 \
    --epochs 60 \
    --lr_drop 60 \
    --lr_vit_layer_decay 0.8 \
    --lr_component_decay 0.7 \
    --encoder dinov2_small \
    --vit_encoder_num_layers 6 \
    --window_block_indexes 0 2 4 \
    --out_feature_indexes 1 3 5 \
    --dec_layers 3 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --num_queries 100 \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 100 \
    --dataset_file coco \
    --coco_path "../../../../../data/dataset/coco" \
    --square_resize_div_64 \
    --use_ema \
    --output_dir output/$model_name
