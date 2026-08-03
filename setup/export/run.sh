#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash setup/export/run.sh <tiny|small|medium> <coco_path> <checkpoint> [export_model args]

Examples:
  bash setup/export/run.sh tiny ../../data/COCO pretrain_weights/coco/tiny/lwdetr_fp32.pth
  bash setup/export/run.sh small ../../data/COCO /path/to/checkpoint.pth --simplify
  bash setup/export/run.sh medium ../../data/COCO /path/to/checkpoint.pth --simplify --trt

Optional environment variables:
  DATASET_FILE=coco|custom_coco   Dataset metadata type. Default: coco
  OUTPUT_DIR=<path>               Export output directory. Default: output/lwdetr_<size>_<dataset>
  DEVICE=<cuda|cpu|index>         Device id passed to export. Default: cuda
  EASY_QUANTIZE=true              Export a checkpoint trained with --easy_quantize
  QUANT=true                      Export a checkpoint using quantized modules
  DECODER_CROSS_ATTN_VALUE_PER_CHANNEL=true
  DECODER_NORM_OUTPUT_QACT=true
  AIDGE_EXPORT=true               Avoid selected ONNX patterns unsupported by Aidge CPU
  DRY_RUN=true                    Print the generated command and exit

Output:
  <OUTPUT_DIR>/inference_model.onnx
  <OUTPUT_DIR>/inference_model.sim.onnx when --simplify is passed
  <OUTPUT_DIR>/inference_model[.sim].engine when --trt is passed
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

export MODEL_SIZE="$1"
coco_path="$2"
checkpoint="$3"
shift 3

dataset_file="${DATASET_FILE:-coco}"
device="${DEVICE:-cuda}"
easy_quantize="${EASY_QUANTIZE:-false}"
quant="${QUANT:-false}"
aidge_export="${AIDGE_EXPORT:-false}"

case "$dataset_file" in
  coco|custom_coco) ;;
  *)
    echo "Unsupported DATASET_FILE: $dataset_file" >&2
    usage
    exit 2
    ;;
esac

case "$easy_quantize" in
  true|false) ;;
  *)
    echo "EASY_QUANTIZE must be true or false, got: $easy_quantize" >&2
    usage
    exit 2
    ;;
esac

case "$quant" in
  true|false) ;;
  *)
    echo "QUANT must be true or false, got: $quant" >&2
    usage
    exit 2
    ;;
esac

case "$aidge_export" in
  true|false) ;;
  *)
    echo "AIDGE_EXPORT must be true or false, got: $aidge_export" >&2
    usage
    exit 2
    ;;
esac

configure_model_size
prepare_common_args
build_architecture_model_args

if [[ ! -d "$coco_path" ]]; then
  echo "COCO path does not exist or is not a directory: $coco_path" >&2
  exit 2
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint does not exist or is not a file: $checkpoint" >&2
  exit 2
fi

output_dir="${OUTPUT_DIR:-output/$model_name}"

bool_flag_args() {
  local value="$1"
  local flag="$2"
  local -n out_ref="$3"
  case "$value" in
    true) out_ref=("$flag") ;;
    false) out_ref=() ;;
    *)
      echo "${flag#--} must be true or false, got: $value" >&2
      usage
      exit 2
      ;;
  esac
}

quant_args=()
if [[ "$quant" == "true" ]]; then
  quant_args=(--quant)
  decoder_cross_attn_value_per_channel_args=()
  decoder_norm_output_qact_args=()
  bool_flag_args "${DECODER_CROSS_ATTN_VALUE_PER_CHANNEL:-false}" --decoder_cross_attn_value_per_channel decoder_cross_attn_value_per_channel_args
  bool_flag_args "${DECODER_NORM_OUTPUT_QACT:-false}" --decoder_norm_output_qact decoder_norm_output_qact_args
  quant_args+=(
    "${decoder_cross_attn_value_per_channel_args[@]}"
    "${decoder_norm_output_qact_args[@]}"
  )
fi

easy_quantize_args=()
if [[ "$easy_quantize" == "true" ]]; then
  easy_quantize_args=(--easy_quantize)
fi

aidge_export_args=()
if [[ "$aidge_export" == "true" ]]; then
  aidge_export_args=(--aidge_export)
fi

cmd=(
  python main.py
  --batch_size 1
  "${architecture_model_args[@]}"
  "${easy_quantize_args[@]}"
  "${quant_args[@]}"
  --dataset_file "$dataset_file"
  --coco_path "$coco_path"
  --square_resize_div_64
  --use_ema
  --eval
  --resume "$checkpoint"
  --output_dir "$output_dir"
  --device "$device"
  export_model
  "${aidge_export_args[@]}"
  "$@"
)

cd "$repo_root"
printf 'Export command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi
exec "${cmd[@]}"
