#!/usr/bin/env bash
#SBATCH --job-name=lwdetr
#SBATCH --time=150:00:00
#SBATCH --partition=gpu80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
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
  ptq                   Runs setup/ptq/run.sh
  qat                   Runs setup/qat/run.sh

Factory AI environment overrides:
  CONDA_SH              Conda initialization script
  CONDA_ENV             Conda environment. Default: lwdetr2
  REPO_ROOT             Repository root override
  FACTORY_SETUP         Optional site-specific shell script to source
  NPROC_PER_NODE        GPU worker count. Default: SLURM allocation or #SBATCH --gres GPU count
  MASTER_PORT           Distributed master port. Default: derived from SLURM_JOB_ID
  PYTORCH_CUDA_ALLOC_CONF
  SLURM_WRAPPER_DRY_RUN Print dispatch command without calling srun

Resource requests can be overridden directly through sbatch, for example:
  sbatch --partition=gpu80G --gres=gpu:8 setup/slurm/run.sh ...

Examples:
  sbatch setup/slurm/run.sh finetune tiny /path/to/custom-coco

  sbatch setup/slurm/run.sh eval \
    tiny original /path/to/checkpoint.pth /path/to/custom-coco

  sbatch setup/slurm/run.sh ptq tiny \
    /path/to/custom-coco /path/to/checkpoint.pth

  DATASET_FILE=custom_coco sbatch setup/slurm/run.sh qat small \
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
    command=(bash "setup/$workflow/run.sh" "$@")
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

extract_gpu_count() {
  local value="$1"
  if [[ "$value" =~ gpu(:[^,:[:space:]]+)*:([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  elif [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  elif [[ "$value" =~ ([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

sbatch_gres_gpus=4
script_file="${BASH_SOURCE[0]}"
if [[ -f "$script_file" ]]; then
  while IFS= read -r line; do
    if [[ "$line" =~ ^#SBATCH[[:space:]]+--gres=([^[:space:]]+) ]]; then
      if parsed_gpus="$(extract_gpu_count "${BASH_REMATCH[1]}")"; then
        sbatch_gres_gpus="$parsed_gpus"
      fi
      break
    fi
  done < "$script_file"
fi

allocated_gpus="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-$sbatch_gres_gpus}}"
if parsed_gpus="$(extract_gpu_count "$allocated_gpus")"; then
  allocated_gpus="$parsed_gpus"
else
  allocated_gpus="$sbatch_gres_gpus"
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-$allocated_gpus}"
default_master_port=29500
if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  default_master_port=$((10000 + SLURM_JOB_ID % 50000))
fi
export MASTER_PORT="${MASTER_PORT:-$default_master_port}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "workflow=$workflow"
echo "repo_root=$repo_root"
echo "conda_env=$conda_env"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MASTER_PORT=$MASTER_PORT"
printf 'command='
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${SLURM_WRAPPER_DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

srun --unbuffered "${command[@]}"
