#!/usr/bin/env bash
#SBATCH --job-name=lwdetr_tiny_qat_full
#SBATCH --time=150:00:00
#SBATCH --partition=gpu40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo_root"

conda_sh="${CONDA_SH:-/home/tl286648/miniforge3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-lwdetr2}"
if [[ -f "$conda_sh" ]]; then
  # shellcheck source=/dev/null
  source "$conda_sh"
  conda activate "$conda_env"
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MODEL_SIZE="${MODEL_SIZE:-tiny}"
export SCENARIO="${SCENARIO:-fully_quantized}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACT_OBSERVER="${ACT_OBSERVER:-ema_minmax}"
export ACT_PERCENTILE="${ACT_PERCENTILE:-99.9}"
export CALIB_BATCHES="${CALIB_BATCHES:-200}"
export CALIB_EVAL="${CALIB_EVAL:-true}"
qat_output_dir="${OUTPUT_DIR:-output/lwdetr_${MODEL_SIZE}_qat_${SCENARIO}}"
ptq_output_dir="${PTQ_OUTPUT_DIR:-${qat_output_dir}_ptq_calib${CALIB_BATCHES}}"

export QUANT_MODEL_COMPONENTS="${QUANT_MODEL_COMPONENTS:-backbone transformer}"
export BACKBONE_QUANT_COMPONENTS="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
export ENCODER_QUANT_COMPONENTS="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
export PROJECTOR_QUANT_COMPONENTS="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"
export TRANSFORMER_QUANT_COMPONENTS="${TRANSFORMER_QUANT_COMPONENTS:-self_attn cross_attn ffn norm refpoint two_stage decoder_output_norm}"
export LWDETR_QUANT_COMPONENTS="${LWDETR_QUANT_COMPONENTS:-linear}"

coco_path="${COCO_PATH:-${1:-../../../../../data/dataset/coco}}"
checkpoint_path="${CHECKPOINT_PATH:-${2:-pretrain_weights/checkpoint_easy_quant.pth}}"
if [[ -f "$checkpoint_path" ]]; then
  checkpoint_root="$(cd "$(dirname "$checkpoint_path")" && pwd)"
else
  checkpoint_root="$checkpoint_path"
fi

echo "COCO path: $coco_path"
echo "Checkpoint path: $checkpoint_path"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "SCENARIO: $SCENARIO"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "CALIB_BATCHES: $CALIB_BATCHES"
echo "CALIB_EVAL: $CALIB_EVAL"
echo "PTQ output dir: $ptq_output_dir"
echo "QAT output dir: $qat_output_dir"
echo "Repo root: $repo_root"

export OUTPUT_DIR="$ptq_output_dir"
srun bash "setup/ptq/${SCENARIO}.sh" "$coco_path" "$checkpoint_path"

calibrated_checkpoint="$ptq_output_dir/checkpoint_calibrated.pth"
if [[ ! -f "$calibrated_checkpoint" ]]; then
  echo "Expected calibrated checkpoint not found: $calibrated_checkpoint" >&2
  exit 1
fi

export OUTPUT_DIR="$qat_output_dir"
export RESUME="${QAT_RESUME:-$calibrated_checkpoint}"
echo "QAT resume checkpoint: $RESUME"

srun bash "setup/qat/${SCENARIO}.sh" "$coco_path" "$checkpoint_root"
