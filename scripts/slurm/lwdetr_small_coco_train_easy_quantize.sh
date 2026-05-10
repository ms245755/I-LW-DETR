#!/usr/bin/env bash
#SBATCH --job-name=lwdetr_small_easy_quant
#SBATCH --time=100:00:00
#SBATCH --partition=prismgpup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo_root"

conda_sh="${CONDA_SH:-/home/tl286648/miniforge3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-lwdetr2}"
if [[ -f "$conda_sh" ]]; then
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$conda_env"
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

coco_path="${COCO_PATH:-${1:-../../../../../data/dataset/coco}}"
checkpoint_path="${CHECKPOINT_PATH:-${2:-}}"
output_dir="${OUTPUT_DIR:-${3:-}}"
if [[ -z "$checkpoint_path" ]]; then
  echo "Set CHECKPOINT_PATH or pass the existing small checkpoint as the second argument." >&2
  exit 2
fi
if [[ -z "$output_dir" ]]; then
  echo "Set OUTPUT_DIR or pass it as the third argument." >&2
  exit 2
fi

echo "repo_root=$repo_root"
echo "coco_path=$coco_path"
echo "checkpoint_path=$checkpoint_path"
echo "output_dir=$output_dir"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"

srun python -u -m torch.distributed.launch \
    --nproc_per_node="$NPROC_PER_NODE" \
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
    --encoder vit_tiny \
    --vit_encoder_num_layers 10 \
    --window_block_indexes 0 1 3 6 7 9 \
    --out_feature_indexes 2 4 5 9 \
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
    --easy_quantize \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 300 \
    --dataset_file coco \
    --coco_path "$coco_path" \
    --square_resize_div_64 \
    --use_ema \
    --pretrain_weights "$checkpoint_path" \
    --pretrain_exclude_keys \
        "bbox_embed.*" \
        "transformer.decoder.bbox_embed.*" \
        "transformer.enc_out_bbox_embed.*" \
        "transformer.decoder.ref_point_coord_proj.*" \
    --output_dir "$output_dir"
