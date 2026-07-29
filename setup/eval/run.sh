#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash setup/eval/run.sh <tiny|small|medium> <original|qr|ptq|qat> <checkpoint> <dataset_root>

Environment variables:
  DATASET_FILE       coco or custom_coco. Default: coco
  OUTPUT_DIR         Default: output/eval/<size>_<variant>
  NPROC_PER_NODE     Default: size-specific common.sh value
  MASTER_PORT        Default: 29500
  BATCH_SIZE         Default: size-specific common.sh value
  NUM_CLASSES        Optional classification-head size override
  DEVICE             Default: cuda

Example:
  bash setup/eval/run.sh tiny qat /path/to/checkpoint.pth /path/to/COCO
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 2
fi

export MODEL_SIZE="$1"
variant="$2"
checkpoint="$(absolute_existing_path "$3")"
coco_path="$(absolute_existing_path "$4")"
export DATASET_FILE="${DATASET_FILE:-coco}"

case "$variant" in
  original|qr|ptq|qat) ;;
  *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
esac
case "$DATASET_FILE" in
  coco|custom_coco) ;;
  *) echo "Unsupported DATASET_FILE: $DATASET_FILE" >&2; exit 2 ;;
esac
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint is not a file: $checkpoint" >&2
  exit 2
fi

configure_model_size
prepare_common_args
build_architecture_model_args
build_common_model_args

device="${DEVICE:-cuda}"
master_port="${MASTER_PORT:-29500}"
output_dir="${OUTPUT_DIR:-output/eval/${MODEL_SIZE}_${variant}}"

if [[ "$device" == cuda* ]]; then
  visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "$visible_gpus" =~ ^[0-9]+$ && "$visible_gpus" -gt 0 && "$nproc_per_node" -gt "$visible_gpus" ]]; then
    echo "NPROC_PER_NODE=$nproc_per_node but only $visible_gpus CUDA device(s) are visible." >&2
    exit 2
  fi
fi

model_args=(
  "${architecture_model_args[@]}"
  --dataset_file "$DATASET_FILE"
  --coco_path "$coco_path"
  "${num_classes_args[@]}"
  --square_resize_div_64
)
variant_args=()
case "$variant" in
  original) ;;
  qr) variant_args=(--easy_quantize) ;;
  ptq)
    model_args=("${common_model_args[@]}")
    variant_args=(--quant)
    ;;
  qat)
    model_args=("${common_model_args[@]}")
    variant_args=(--quant --ia_bce_loss --cls_loss_coef 1 --find_unused_parameters)
    ;;
esac

mkdir -p "$output_dir"
log_file="$output_dir/eval.log"
cd "$repo_root"

echo "Model size: $MODEL_SIZE"
echo "Variant: $variant"
echo "Checkpoint: $checkpoint"
echo "Dataset: $coco_path"
echo "Output: $output_dir"

set +e
python -u -m torch.distributed.launch \
  --nproc_per_node "$nproc_per_node" \
  --master_port "$master_port" \
  --use_env \
  main.py \
  --batch_size "$batch_size" \
  "${model_args[@]}" \
  "${variant_args[@]}" \
  --eval \
  --resume "$checkpoint" \
  --output_dir "$output_dir" \
  --device "$device" 2>&1 | tee "$log_file"
status="${PIPESTATUS[0]}"
set -e

if [[ "$status" -ne 0 ]]; then
  echo "Evaluation failed. See $log_file" >&2
  exit "$status"
fi
echo "Evaluation complete: $log_file"
