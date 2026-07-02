#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

if [[ $# -ne 2 ]]; then
  cat >&2 <<EOF
Usage: bash setup/ptq/sweep_softmax_m_guard.sh <coco_path> <checkpoint_path_or_dir>

Runs PTQ sequentially to compare IViTSoftmax m behavior:
  no_m_calibration
  m_guard_0 m_guard_4 m_guard_8 m_guard_12 m_guard_16

The sweep intentionally quantizes only softmax:
  - ViT encoder attention softmax
  - Transformer decoder self-attention softmax
  - Transformer decoder cross-attention softmax
All other model components remain float, except the activation quantizers needed
to provide scales to IViTSoftmax.

Optional environment variables:
  MODEL_SIZE=tiny|small|medium        Default: tiny
  SWEEP_OUTPUT_ROOT=<path>            Default: output/ptq_softmax_m_guard_sweep
  SWEEP_CSV=<path>                    Default: <SWEEP_OUTPUT_ROOT>/<model_size>/summary.csv
  SWEEP_LOG=<path>                    Default: <SWEEP_OUTPUT_ROOT>/<model_size>/summary.log
  SWEEP_OVERWRITE=true                Replace existing summary files instead of appending
  M_GUARDS="0 4 8 12 16"              m_guard_bits values to sweep
  NPROC_PER_NODE=1|2|4|8              Forwarded to setup/ptq/fully_quantized.sh
  CALIB_BATCHES=500                   Forwarded to setup/ptq/fully_quantized.sh
  CALIB_EVAL=true                     Forwarded to setup/ptq/fully_quantized.sh
  ACT_OBSERVER=mean|ema_minmax        Forwarded to setup/ptq/fully_quantized.sh
  ACT_PERCENTILE=99|99.9              Forwarded to setup/ptq/fully_quantized.sh
EOF
  exit 2
fi

coco_path="$1"
checkpoint_path="$2"
model_size="${MODEL_SIZE:-tiny}"
sweep_root="${SWEEP_OUTPUT_ROOT:-output/ptq_softmax_m_guard_sweep_v2}/${model_size}"
csv_path="${SWEEP_CSV:-$sweep_root/summary.csv}"
summary_log="${SWEEP_LOG:-$sweep_root/summary.log}"
guards="${M_GUARDS:-${SAFE_GUARDS:-0 4 8 12 16}}"
nproc_per_node="${NPROC_PER_NODE:-4}"
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
  printf '%s\n' "$line" | sed -E 's/.*= *([0-9.]+)$/\1/' | awk '{ printf "%.3f", $1 * 100 }'
}

extract_softmax_r_summary() {
  local log_file="$1"
  local line
  line="$(grep -F "Calibrated IViTSoftmax M/R:" "$log_file" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    printf ','
    return
  fi

  local r_min
  local r_max
  r_min="$(printf '%s\n' "$line" | sed -nE 's/.*r_min=([0-9-]+).*/\1/p')"
  r_max="$(printf '%s\n' "$line" | sed -nE 's/.*r_max=([0-9-]+).*/\1/p')"
  printf '%s,%s' "$r_min" "$r_max"
}

write_csv_row() {
  local case_name="$1"
  local calibrate_m="$2"
  local m_guard_bits="$3"
  local status="$4"
  local exit_code="$5"
  local ap="$6"
  local output_dir="$7"
  local log_file="$8"
  local r_min="$9"
  local r_max="${10}"

  {
    csv_escape "$case_name"; printf ','
    csv_escape "$calibrate_m"; printf ','
    csv_escape "$m_guard_bits"; printf ','
    csv_escape "$status"; printf ','
    csv_escape "$exit_code"; printf ','
    csv_escape "$ap"; printf ','
    csv_escape "$r_min"; printf ','
    csv_escape "$r_max"; printf ','
    csv_escape "$output_dir"; printf ','
    csv_escape "$log_file"; printf '\n'
  } >> "$csv_path"
}

run_case() {
  local case_name="$1"
  local calibrate_m="$2"
  local m_guard_bits="$3"
  local output_dir="$sweep_root/$case_name"
  local log_file="$sweep_root/logs/${case_name}.log"
  local exit_code
  local status
  local ap
  local r_summary
  local r_min
  local r_max

  mkdir -p "$output_dir"

  echo "==> Running PTQ IViTSoftmaxV2 m_guard case: $case_name"
  echo "    calibrate_m: $calibrate_m"
  echo "    m_guard_bits: ${m_guard_bits:-NA}"
  echo "    quantized:   ViT softmax + transformer self/cross-attn softmax only"
  echo "    output:      $output_dir"
  echo "    log:         $log_file"

  export MODEL_SIZE="$model_size"
  export NPROC_PER_NODE="$nproc_per_node"
  export OUTPUT_DIR="$output_dir"
  export QUANT_MODEL_COMPONENTS="backbone transformer"
  export BACKBONE_QUANT_COMPONENTS="vit"
  export ENCODER_QUANT_COMPONENTS="softmax"
  export PROJECTOR_QUANT_COMPONENTS="conv2d gelu silu layernorm"
  export TRANSFORMER_QUANT_COMPONENTS="self_attn cross_attn"
  export DECODER_SELF_ATTN_QUANT_COMPONENTS="softmax"
  export DECODER_CROSS_ATTN_QUANT_COMPONENTS="softmax"
  export DECODER_FFN_QUANT_COMPONENTS="none"
  export DECODER_REFPOINT_QUANT_COMPONENTS="none"
  export DECODER_TWO_STAGE_QUANT_COMPONENTS="none"
  export DECODER_OUTPUT_NORM_QUANT_COMPONENTS="none"
  export DECODER_QUANT_NORMS="none"
  export LWDETR_QUANT_COMPONENTS="none"
  export SOFTMAX_CALIBRATE_M="$calibrate_m"
  if [[ -n "$m_guard_bits" ]]; then
    export SOFTMAX_M_SAFE_GUARD="$m_guard_bits"
    export SAFE_GUARD="$m_guard_bits"
  else
    unset SOFTMAX_M_SAFE_GUARD
    unset SAFE_GUARD
  fi

  set +e
  bash "$script_dir/fully_quantized.sh" "$coco_path" "$checkpoint_path" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  ap="$(extract_ap "$log_file")"
  r_summary="$(extract_softmax_r_summary "$log_file")"
  r_min="${r_summary%,*}"
  r_max="${r_summary#*,}"
  if [[ "$exit_code" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi

  write_csv_row "$case_name" "$calibrate_m" "$m_guard_bits" "$status" "$exit_code" "$ap" "$output_dir" "$log_file" "$r_min" "$r_max"
  {
    echo "$case_name: calibrate_m=$calibrate_m m_guard_bits=${m_guard_bits:-NA} r_min=${r_min:-NA} r_max=${r_max:-NA} status=$status exit_code=$exit_code ap=${ap:-NA} output_dir=$output_dir log_file=$log_file"
  } >> "$summary_log"
  echo "==> Finished $case_name: status=$status exit_code=$exit_code ap=${ap:-NA}"
}

if [[ ! -f "$csv_path" || "${SWEEP_OVERWRITE:-false}" == "true" ]]; then
  printf 'case,softmax_calibrate_m,m_guard_bits,status,exit_code,map_50_95,r_min,r_max,output_dir,log_file\n' > "$csv_path"
fi
if [[ "${SWEEP_OVERWRITE:-false}" == "true" ]]; then
  : > "$summary_log"
fi

{
  echo "IViTSoftmaxV2 m_guard_bits PTQ sweep"
  echo "repo_root=$repo_root"
  echo "model_size=$model_size"
  echo "coco_path=$coco_path"
  echo "checkpoint_path=$checkpoint_path"
  echo "guards=$guards"
  echo "nproc_per_node=$nproc_per_node"
  echo "csv_path=$csv_path"
  echo
} >> "$summary_log"

run_case "no_m_calibration" "false" ""

for guard in $guards; do
  run_case "m_guard_${guard}" "true" "$guard"
done

echo "Wrote summary CSV: $csv_path"
echo "Wrote summary log: $summary_log"
