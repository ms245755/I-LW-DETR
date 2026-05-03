#!/usr/bin/env bash
#SBATCH --job-name=lwdetr_tiny_qat_vit_proj
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --partition=prismgpup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:8

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo_root"
mkdir -p logs

conda_sh="${CONDA_SH:-/home/tl286648/miniforge3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-lwdetr2}"
if [[ -f "$conda_sh" ]]; then
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$conda_env"
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MODEL_SIZE="${MODEL_SIZE:-tiny}"
export SCENARIO="vit_projector"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACT_OBSERVER="${ACT_OBSERVER:-ema_minmax}"
export ACT_PERCENTILE="${ACT_PERCENTILE:-99.9}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/lwdetr_${MODEL_SIZE}_qat_vit_projector_safe}"

export QUANT_MODEL_COMPONENTS="${QUANT_MODEL_COMPONENTS:-backbone}"
export BACKBONE_QUANT_COMPONENTS="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
export ENCODER_QUANT_COMPONENTS="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul layernorm}"
export PROJECTOR_QUANT_COMPONENTS="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"
export TRANSFORMER_QUANT_COMPONENTS="${TRANSFORMER_QUANT_COMPONENTS:-none}"
export LWDETR_QUANT_COMPONENTS="${LWDETR_QUANT_COMPONENTS:-none}"

coco_path="${COCO_PATH:-${1:-../../../../../data/dataset/coco}}"
checkpoint_path="${CHECKPOINT_PATH:-${2:-pretrain_weights/checkpoint_easy_quant.pth}}"

echo "repo_root=$repo_root"
echo "coco_path=$coco_path"
echo "checkpoint_path=$checkpoint_path"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "ENCODER_QUANT_COMPONENTS=$ENCODER_QUANT_COMPONENTS"
echo "PROJECTOR_QUANT_COMPONENTS=$PROJECTOR_QUANT_COMPONENTS"

srun bash "setup/qat/${SCENARIO}.sh" "$coco_path" "$checkpoint_path"
