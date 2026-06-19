#!/usr/bin/env bash
#SBATCH --job-name=lwdetr
#SBATCH --time=40:00:00
#SBATCH --partition=gpu40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Keep all diagnostics in the file selected by sbatch -o/--output.
exec 2>&1

trap 'status=$?; echo "[$(date --iso-8601=seconds)] Slurm wrapper exiting with status $status"' EXIT

# sbatch may execute a copied script from Slurm's spool directory, so resolving
# the repository relative to BASH_SOURCE is unreliable inside an allocation.
repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

echo "[$(date --iso-8601=seconds)] LW-DETR Slurm wrapper started"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "hostname=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unset}"
echo "repo_root=$repo_root"
echo "arguments=$*"

usage() {
  cat >&2 <<'EOF'
Usage:
  sbatch setup/slurm/run.sh <workflow> [workflow arguments]

Workflows:
  finetune              Runs setup/finetune/run.sh
  eval                  Runs setup/eval/run.sh
  ptq <scenario>        Runs setup/ptq/<scenario>.sh
  qat <scenario>        Runs setup/qat/<scenario>.sh

PTQ/QAT scenarios:
  fully_quantized
  vit_only
  vit_projector
  vit_projector_transformer_float_head

Factory AI environment overrides:
  CONDA_SH              Conda initialization script
  CONDA_ENV             Conda environment. Default: lwdetr2
  REPO_ROOT             Repository root override
  FACTORY_SETUP         Optional site-specific shell script to source
  NPROC_PER_NODE        GPU worker count. Default: SLURM allocation or 4
  MASTER_PORT           Distributed master port. Default: 29500
  PYTORCH_CUDA_ALLOC_CONF
  SLURM_WRAPPER_DRY_RUN Print dispatch command without calling srun

Resource requests can be overridden directly through sbatch, for example:
  sbatch --partition=gpu80G --gres=gpu:8 setup/slurm/run.sh ...

Examples:
  sbatch setup/slurm/run.sh finetune tiny /path/to/custom-coco

  sbatch setup/slurm/run.sh eval \
    --model-size tiny --variant original --dataset-file custom_coco \
    --checkpoint /path/to/checkpoint.pth --coco-path /path/to/custom-coco

  sbatch setup/slurm/run.sh ptq fully_quantized \
    /path/to/custom-coco /path/to/checkpoint.pth

  MODEL_SIZE=small DATASET_FILE=custom_coco \
  sbatch setup/slurm/run.sh qat fully_quantized \
    /path/to/custom-coco /path/to/checkpoint.pth
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

workflow="$1"
shift

case "$workflow" in
  finetune)
    command=(bash setup/finetune/run.sh "$@")
    ;;
  eval)
    command=(bash setup/eval/run.sh "$@")
    ;;
  ptq|qat)
    if [[ $# -lt 1 ]]; then
      echo "$workflow requires a scenario." >&2
      usage
      exit 2
    fi
    scenario="$1"
    shift
    case "$scenario" in
      fully_quantized|vit_only|vit_projector|vit_projector_transformer_float_head) ;;
      *)
        echo "Unsupported $workflow scenario: $scenario" >&2
        usage
        exit 2
        ;;
    esac
    command=(bash "setup/$workflow/$scenario.sh" "$@")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unsupported workflow: $workflow" >&2
    usage
    exit 2
    ;;
esac

if [[ ! -f "$repo_root/main.py" || ! -d "$repo_root/setup" ]]; then
  echo "Repository root is invalid: $repo_root" >&2
  echo "Submit from the LW-DETR root or set REPO_ROOT explicitly." >&2
  exit 2
fi

cd "$repo_root"
mkdir -p logs

factory_setup="${FACTORY_SETUP:-}"
if [[ -n "$factory_setup" ]]; then
  if [[ ! -f "$factory_setup" ]]; then
    echo "FACTORY_SETUP does not exist: $factory_setup" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$factory_setup"
fi

conda_sh="${CONDA_SH:-/home/tl286648/miniforge3/etc/profile.d/conda.sh}"
conda_env="${CONDA_ENV:-lwdetr2}"
if [[ -f "$conda_sh" ]]; then
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$conda_env"
fi

allocated_gpus="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
if [[ "$allocated_gpus" =~ ([0-9]+)$ ]]; then
  allocated_gpus="${BASH_REMATCH[1]}"
else
  allocated_gpus=4
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-$allocated_gpus}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "workflow=$workflow"
echo "repo_root=$repo_root"
echo "conda_env=$conda_env"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
printf 'command='
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${SLURM_WRAPPER_DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

srun --unbuffered "${command[@]}"
