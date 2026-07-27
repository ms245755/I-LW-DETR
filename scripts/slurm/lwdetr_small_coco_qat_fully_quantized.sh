#!/usr/bin/env bash
#SBATCH --job-name=lwdetr_small_qat_full
#SBATCH --time=100:00:00
#SBATCH --partition=gpu80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:6

set -euo pipefail
trap 'status=$?; echo "FAILED at line $LINENO with exit code $status"; exit $status' ERR

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo_root"
mkdir -p logs/qat

conda_sh="${CONDA_SH:-/home/tl286648/miniforge3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-lwdetr2}"
if [[ -f "$conda_sh" ]]; then
  # shellcheck source=/dev/null
  source "$conda_sh"
  conda activate "$conda_env"
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"
else
  export MASTER_PORT="${MASTER_PORT:-29500}"
fi

export MODEL_SIZE="${MODEL_SIZE:-small}"
export SCENARIO="${SCENARIO:-fully_quantized}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACT_OBSERVER="${ACT_OBSERVER:-ema_minmax}"
export ACT_PERCENTILE="${ACT_PERCENTILE:-99.9}"
export CALIB_BATCHES="${CALIB_BATCHES:-500}"
export CALIB_EVAL="${CALIB_EVAL:-true}"
export QAT_CALIB_BATCHES="${QAT_CALIB_BATCHES:-0}"
export QAT_FREEZE_OBSERVERS="${QAT_FREEZE_OBSERVERS:-true}"
qat_output_dir="${OUTPUT_DIR:-output/lwdetr_${MODEL_SIZE}_qat_${SCENARIO}}"
ptq_output_dir="${PTQ_OUTPUT_DIR:-$qat_output_dir}"

coco_path="${COCO_PATH:-${1:-../../../../../data/dataset/coco}}"
checkpoint_path="${CHECKPOINT_PATH:-${2:-pretrain_weights/checkpoint_easy_quant_small.pth}}"
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
echo "QAT_CALIB_BATCHES: $QAT_CALIB_BATCHES"
echo "QAT_FREEZE_OBSERVERS: $QAT_FREEZE_OBSERVERS"
echo "PTQ output dir: $ptq_output_dir"
echo "QAT output dir: $qat_output_dir"
echo "Repo root: $repo_root"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "MASTER_PORT: $MASTER_PORT"

if [[ -n "${QAT_RESUME:-}" ]]; then
  if [[ ! -f "$QAT_RESUME" ]]; then
    echo "QAT_RESUME does not exist: $QAT_RESUME" >&2
    exit 1
  fi
  echo "QAT_RESUME is set; skipping PTQ phase."
  calibrated_checkpoint="$QAT_RESUME"
else
  echo "Starting PTQ phase"
  export OUTPUT_DIR="$ptq_output_dir"
  srun bash "setup/ptq/${SCENARIO}.sh" "$coco_path" "$checkpoint_path"

  calibrated_checkpoint="$ptq_output_dir/checkpoint_calibrated.pth"
  if [[ ! -f "$calibrated_checkpoint" ]]; then
    echo "Expected calibrated checkpoint not found: $calibrated_checkpoint" >&2
    exit 1
  fi
fi

export OUTPUT_DIR="$qat_output_dir"
export RESUME="$calibrated_checkpoint"
export SKIP_PRETRAIN="${SKIP_PRETRAIN:-true}"
export QAT_CALIB_BATCHES="${EXTRA_QAT_CALIB_BATCHES:-0}"
echo "QAT resume checkpoint: $RESUME"
echo "SKIP_PRETRAIN for QAT phase: $SKIP_PRETRAIN"
echo "QAT_CALIB_BATCHES for QAT phase: $QAT_CALIB_BATCHES"
echo "Starting QAT phase"

srun bash "setup/qat/${SCENARIO}.sh" "$coco_path" "$checkpoint_root"
