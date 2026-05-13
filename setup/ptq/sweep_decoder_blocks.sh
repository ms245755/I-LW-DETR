#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

if [[ $# -ne 2 ]]; then
  cat >&2 <<EOF
Usage: bash setup/ptq/sweep_decoder_blocks.sh <coco_path> <checkpoint_path_or_dir>

Runs PTQ sequentially with one transformer decoder block quantized at a time.

Optional environment variables:
  MODEL_SIZE=tiny|small|medium        Default: tiny
  SWEEP_OUTPUT_ROOT=output/ptq_decoder_block_sweep
  SWEEP_CSV=<path>                    Default: <SWEEP_OUTPUT_ROOT>/<model_size>/summary.csv
  BLOCKS="self_attn cross_attn ..."   Override block list
  CALIB_BATCHES=500                   Forwarded to setup/ptq/fully_quantized.sh
  CALIB_EVAL=true                     Forwarded to setup/ptq/fully_quantized.sh
EOF
  exit 2
fi

coco_path="$1"
checkpoint_path="$2"
model_size="${MODEL_SIZE:-tiny}"
sweep_root="${SWEEP_OUTPUT_ROOT:-output/ptq_decoder_block_sweep}/${model_size}"
csv_path="${SWEEP_CSV:-$sweep_root/summary.csv}"
blocks="${BLOCKS:-self_attn cross_attn ffn norm refpoint two_stage decoder_output_norm}"
ap_pattern="Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"

mkdir -p "$sweep_root/logs"

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

extract_ap() {
  local log_file="$1"
  local line
  line="$(grep -F "$ap_pattern" "$log_file" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    printf ''
    return
  fi
  printf '%s\n' "$line" | sed -E 's/.*= *([0-9.]+)$/\1/'
}

write_row() {
  local block="$1"
  local status="$2"
  local exit_code="$3"
  local ap="$4"
  local output_dir="$5"
  local log_file="$6"

  {
    csv_escape "$block"; printf ','
    csv_escape "$status"; printf ','
    csv_escape "$exit_code"; printf ','
    csv_escape "$ap"; printf ','
    csv_escape "$output_dir"; printf ','
    csv_escape "$log_file"; printf '\n'
  } >> "$csv_path"
}

block_subcomponent_env() {
  local block="$1"

  export DECODER_SELF_ATTN_QUANT_COMPONENTS="none"
  export DECODER_CROSS_ATTN_QUANT_COMPONENTS="none"
  export DECODER_FFN_QUANT_COMPONENTS="none"
  export DECODER_REFPOINT_QUANT_COMPONENTS="none"
  export DECODER_TWO_STAGE_QUANT_COMPONENTS="none"
  export DECODER_OUTPUT_NORM_QUANT_COMPONENTS="none"

  case "$block" in
    self_attn)
      export DECODER_SELF_ATTN_QUANT_COMPONENTS="all"
      ;;
    cross_attn)
      export DECODER_CROSS_ATTN_QUANT_COMPONENTS="all"
      ;;
    ffn)
      export DECODER_FFN_QUANT_COMPONENTS="all"
      ;;
    norm)
      export DECODER_QUANT_NORMS="${DECODER_QUANT_NORMS:-all}"
      ;;
    refpoint)
      export DECODER_REFPOINT_QUANT_COMPONENTS="all"
      ;;
    two_stage)
      export DECODER_TWO_STAGE_QUANT_COMPONENTS="all"
      ;;
    decoder_output_norm)
      export DECODER_OUTPUT_NORM_QUANT_COMPONENTS="all"
      ;;
    *)
      echo "Unknown decoder block: $block" >&2
      return 2
      ;;
  esac
}

printf 'block,status,exit_code,ap_50_95,output_dir,log_file\n' > "$csv_path"

for block in $blocks; do
  output_dir="$sweep_root/$block"
  log_file="$sweep_root/logs/${block}.log"
  mkdir -p "$output_dir"

  echo "==> Running PTQ decoder block: $block"
  echo "    output: $output_dir"
  echo "    log:    $log_file"

  unset DECODER_QUANT_NORMS
  if ! block_subcomponent_env "$block"; then
    write_row "$block" "invalid_block" "2" "" "$output_dir" "$log_file"
    continue
  fi

  export MODEL_SIZE="$model_size"
  export OUTPUT_DIR="$output_dir"
  export QUANT_MODEL_COMPONENTS="transformer"
  export BACKBONE_QUANT_COMPONENTS=""
  export TRANSFORMER_QUANT_COMPONENTS="$block"
  export LWDETR_QUANT_COMPONENTS="none"

  set +e
  bash "$script_dir/fully_quantized.sh" "$coco_path" "$checkpoint_path" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  ap="$(extract_ap "$log_file")"
  if [[ "$exit_code" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi
  write_row "$block" "$status" "$exit_code" "$ap" "$output_dir" "$log_file"
  echo "==> Finished $block: status=$status exit_code=$exit_code ap=${ap:-NA}"
done

echo "Wrote summary CSV: $csv_path"
