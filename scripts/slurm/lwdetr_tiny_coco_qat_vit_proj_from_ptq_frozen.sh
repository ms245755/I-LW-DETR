#!/usr/bin/env bash
#SBATCH --job-name=lwdetr_tiny_qat_vit_proj_frozen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=prismgpup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4

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

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MODEL_SIZE="${MODEL_SIZE:-tiny}"
export SCENARIO="vit_projector"
export BATCH_SIZE="${BATCH_SIZE:-4}"

# Keep PTQ and QAT quantization configs identical.
export QUANT_MODEL_COMPONENTS="${QUANT_MODEL_COMPONENTS:-backbone}"
export BACKBONE_QUANT_COMPONENTS="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
export ENCODER_QUANT_COMPONENTS="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
export PROJECTOR_QUANT_COMPONENTS="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"
export TRANSFORMER_QUANT_COMPONENTS="${TRANSFORMER_QUANT_COMPONENTS:-none}"
export LWDETR_QUANT_COMPONENTS="${LWDETR_QUANT_COMPONENTS:-none}"

# Match PTQ observer defaults and calibrate on 500 images before QAT.
export ACT_OBSERVER="${ACT_OBSERVER:-mean}"
export ACT_PERCENTILE="${ACT_PERCENTILE:-99}"
export CALIB_BATCHES="${CALIB_BATCHES:-500}"
export CALIB_EVAL="${CALIB_EVAL:-true}"

export LR="${LR:-2e-5}"
export LR_ENCODER="${LR_ENCODER:-2e-5}"
export EPOCHS="${EPOCHS:-20}"
export QAT_CALIB_BATCHES="${QAT_CALIB_BATCHES:-0}"
export QAT_FREEZE_OBSERVERS="${QAT_FREEZE_OBSERVERS:-true}"

coco_path="${COCO_PATH:-${1:-../../../../../data/dataset/coco}}"
checkpoint_path="${CHECKPOINT_PATH:-${2:-pretrain_weights/checkpoint_easy_quant.pth}}"
if [[ -f "$checkpoint_path" ]]; then
  checkpoint_root="$(cd "$(dirname "$checkpoint_path")" && pwd)"
else
  checkpoint_root="$checkpoint_path"
fi

ptq_output_dir="${PTQ_OUTPUT_DIR:-output/lwdetr_${MODEL_SIZE}_ptq_vit_projector_calib${CALIB_BATCHES}}"
qat_output_dir="${OUTPUT_DIR:-output/lwdetr_${MODEL_SIZE}_qat_vit_projector_from_ptq_frozen}"

echo "repo_root=$repo_root"
echo "coco_path=$coco_path"
echo "checkpoint_path=$checkpoint_path"
echo "checkpoint_root=$checkpoint_root"
echo "PTQ output dir=$ptq_output_dir"
echo "QAT output dir=$qat_output_dir"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MODEL_SIZE=$MODEL_SIZE"
echo "CALIB_BATCHES=$CALIB_BATCHES"
echo "QAT_FREEZE_OBSERVERS=$QAT_FREEZE_OBSERVERS"
echo "ENCODER_QUANT_COMPONENTS=$ENCODER_QUANT_COMPONENTS"
echo "PROJECTOR_QUANT_COMPONENTS=$PROJECTOR_QUANT_COMPONENTS"

export OUTPUT_DIR="$ptq_output_dir"
srun bash "setup/ptq/${SCENARIO}.sh" "$coco_path" "$checkpoint_path"

calibrated_checkpoint="$ptq_output_dir/checkpoint_calibrated.pth"
if [[ ! -f "$calibrated_checkpoint" ]]; then
  echo "Expected calibrated checkpoint not found: $calibrated_checkpoint" >&2
  exit 1
fi

export OUTPUT_DIR="$qat_output_dir"
export RESUME="${QAT_RESUME:-$calibrated_checkpoint}"
export QAT_CALIB_BATCHES="${EXTRA_QAT_CALIB_BATCHES:-0}"
echo "QAT resume checkpoint=$RESUME"
echo "QAT_CALIB_BATCHES for QAT phase=$QAT_CALIB_BATCHES"

srun bash "setup/qat/${SCENARIO}.sh" "$coco_path" "$checkpoint_root"
