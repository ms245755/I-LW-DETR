#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

if [[ $# -ne 2 ]]; then
  cat >&2 <<EOF
Usage: bash setup/ptq/sweep_single_components.sh <coco_path> <checkpoint_path_or_dir>

Runs PTQ sequentially with one quantization component/family enabled at a time.

Case format:
  <scope>:<component>

Default cases:
  vit:linear vit:conv2d vit:matmul vit:softmax vit:gelu vit:layernorm
  projector:conv2d projector:silu projector:layernorm
  transformer:linear transformer:matmul transformer:softmax transformer:layernorm
  transformer_two_stage:float_exp transformer_two_stage:ivit_exp
  lwdetr:linear lwdetr_linear:float_exp lwdetr_linear:ivit_exp

Extra fine-grained transformer cases are also supported:
  transformer_self_attn:linear|matmul|softmax
  transformer_cross_attn:value_proj|sampling_offsets|attention_weights|softmax|grid_sample_core|weighted_sum|output_proj|core
  transformer_ffn:linear
  transformer_refpoint:linear|sine_embed|coord_proj
  transformer_two_stage:proposal|enc_output|enc_output_norm|class_head|bbox_head|bbox_reparam|selected_inputs
  transformer_output_norm:layernorm|output_qact
  transformer_two_stage:float_exp|ivit_exp
  lwdetr_linear:float_exp|ivit_exp

Optional environment variables:
  MODEL_SIZE=tiny|small|medium        Default: tiny
  CASES="vit:linear ..."             Override the case list
  SWEEP_OUTPUT_ROOT=output/ptq_single_component_sweep
  SWEEP_CSV=<path>                    Default: <SWEEP_OUTPUT_ROOT>/<model_size>/summary.csv
  SWEEP_LOG=<path>                    Default: <SWEEP_OUTPUT_ROOT>/<model_size>/summary.log
  SWEEP_OVERWRITE=true                Replace existing summary files instead of appending
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
sweep_root="${SWEEP_OUTPUT_ROOT:-output/ptq_single_component_sweep}/${model_size}"
csv_path="${SWEEP_CSV:-$sweep_root/summary.csv}"
summary_log="${SWEEP_LOG:-$sweep_root/summary.log}"
cases="${CASES:-vit:linear vit:conv2d vit:matmul vit:softmax vit:gelu vit:layernorm projector:conv2d projector:silu projector:layernorm transformer:linear transformer:matmul transformer:softmax transformer:layernorm transformer_two_stage:float_exp transformer_two_stage:ivit_exp lwdetr:linear lwdetr_linear:float_exp lwdetr_linear:ivit_exp}"
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

write_csv_row() {
  local case_name="$1"
  local scope="$2"
  local component="$3"
  local status="$4"
  local exit_code="$5"
  local ap="$6"
  local output_dir="$7"
  local log_file="$8"

  {
    csv_escape "$case_name"; printf ','
    csv_escape "$scope"; printf ','
    csv_escape "$component"; printf ','
    csv_escape "$status"; printf ','
    csv_escape "$exit_code"; printf ','
    csv_escape "$ap"; printf ','
    csv_escape "$output_dir"; printf ','
    csv_escape "$log_file"; printf '\n'
  } >> "$csv_path"
}

reset_quant_env() {
  export QUANT_MODEL_COMPONENTS="transformer"
  export BACKBONE_QUANT_COMPONENTS="vit"
  export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
  export ENCODER_QUANT_COMPONENTS="linear"
  export PROJECTOR_QUANT_COMPONENTS="conv2d"
  export TRANSFORMER_QUANT_COMPONENTS="none"
  export DECODER_QUANT_BLOCKS="${DECODER_QUANT_BLOCKS:-all}"
  export DECODER_QUANT_NORMS="none"
  export DECODER_SELF_ATTN_QUANT_COMPONENTS="none"
  export DECODER_CROSS_ATTN_QUANT_COMPONENTS="none"
  export DECODER_FFN_QUANT_COMPONENTS="none"
  export DECODER_REFPOINT_QUANT_COMPONENTS="none"
  export DECODER_TWO_STAGE_QUANT_COMPONENTS="none"
  export DECODER_OUTPUT_NORM_QUANT_COMPONENTS="none"
  export LWDETR_QUANT_COMPONENTS="none"
  export TRANSFORMER_EXP_MODE="ivit"
  export LWDETR_EXP_MODE="ivit"
}

enable_transformer_linear_family() {
  export TRANSFORMER_QUANT_COMPONENTS="self_attn cross_attn ffn refpoint two_stage"
  export DECODER_SELF_ATTN_QUANT_COMPONENTS="linear"
  export DECODER_CROSS_ATTN_QUANT_COMPONENTS="value_proj sampling_offsets attention_weights output_proj"
  export DECODER_FFN_QUANT_COMPONENTS="linear"
  export DECODER_REFPOINT_QUANT_COMPONENTS="linear coord_proj"
  export DECODER_TWO_STAGE_QUANT_COMPONENTS="enc_output class_head bbox_head"
}

configure_case() {
  local case_name="$1"
  local scope="${case_name%%:*}"
  local component="${case_name#*:}"

  if [[ "$scope" == "$component" ]]; then
    echo "Case must use <scope>:<component>: $case_name" >&2
    return 2
  fi

  reset_quant_env

  case "$scope:$component" in
    vit:linear|vit:conv2d|vit:matmul|vit:softmax|vit:gelu|vit:layernorm)
      export QUANT_MODEL_COMPONENTS="backbone"
      export BACKBONE_QUANT_COMPONENTS="vit"
      export ENCODER_QUANT_COMPONENTS="$component"
      ;;
    projector:conv2d|projector:silu|projector:layernorm)
      export QUANT_MODEL_COMPONENTS="backbone"
      export BACKBONE_QUANT_COMPONENTS="projector"
      export PROJECTOR_QUANT_COMPONENTS="$component"
      ;;
    transformer:linear)
      enable_transformer_linear_family
      ;;
    transformer:matmul)
      export TRANSFORMER_QUANT_COMPONENTS="self_attn"
      export DECODER_SELF_ATTN_QUANT_COMPONENTS="matmul"
      ;;
    transformer:softmax)
      export TRANSFORMER_QUANT_COMPONENTS="self_attn cross_attn"
      export DECODER_SELF_ATTN_QUANT_COMPONENTS="softmax"
      export DECODER_CROSS_ATTN_QUANT_COMPONENTS="softmax"
      ;;
    transformer:layernorm)
      export TRANSFORMER_QUANT_COMPONENTS="norm decoder_output_norm"
      export DECODER_QUANT_NORMS="${DECODER_QUANT_NORMS_OVERRIDE:-all}"
      export DECODER_OUTPUT_NORM_QUANT_COMPONENTS="layernorm"
      ;;
    transformer_two_stage:float_exp|transformer_two_stage:ivit_exp)
      export TRANSFORMER_QUANT_COMPONENTS="two_stage"
      export DECODER_TWO_STAGE_QUANT_COMPONENTS="bbox_head bbox_reparam"
      export TRANSFORMER_EXP_MODE="${component%_exp}"
      ;;
    transformer_self_attn:linear|transformer_self_attn:matmul|transformer_self_attn:softmax)
      export TRANSFORMER_QUANT_COMPONENTS="self_attn"
      export DECODER_SELF_ATTN_QUANT_COMPONENTS="$component"
      ;;
    transformer_cross_attn:value_proj|transformer_cross_attn:sampling_offsets|transformer_cross_attn:attention_weights|transformer_cross_attn:softmax|transformer_cross_attn:grid_sample_core|transformer_cross_attn:weighted_sum|transformer_cross_attn:output_proj|transformer_cross_attn:core)
      export TRANSFORMER_QUANT_COMPONENTS="cross_attn"
      export DECODER_CROSS_ATTN_QUANT_COMPONENTS="$component"
      ;;
    transformer_ffn:linear)
      export TRANSFORMER_QUANT_COMPONENTS="ffn"
      export DECODER_FFN_QUANT_COMPONENTS="linear"
      ;;
    transformer_refpoint:linear|transformer_refpoint:sine_embed|transformer_refpoint:coord_proj)
      export TRANSFORMER_QUANT_COMPONENTS="refpoint"
      export DECODER_REFPOINT_QUANT_COMPONENTS="$component"
      ;;
    transformer_two_stage:proposal|transformer_two_stage:enc_output|transformer_two_stage:enc_output_norm|transformer_two_stage:class_head|transformer_two_stage:bbox_head|transformer_two_stage:bbox_reparam|transformer_two_stage:selected_inputs)
      export TRANSFORMER_QUANT_COMPONENTS="two_stage"
      export DECODER_TWO_STAGE_QUANT_COMPONENTS="$component"
      ;;
    transformer_output_norm:layernorm|transformer_output_norm:output_qact)
      export TRANSFORMER_QUANT_COMPONENTS="decoder_output_norm"
      export DECODER_OUTPUT_NORM_QUANT_COMPONENTS="$component"
      ;;
    lwdetr:linear)
      export QUANT_MODEL_COMPONENTS="transformer"
      export TRANSFORMER_QUANT_COMPONENTS="none"
      export LWDETR_QUANT_COMPONENTS="linear"
      ;;
    lwdetr_linear:float_exp|lwdetr_linear:ivit_exp)
      export QUANT_MODEL_COMPONENTS="transformer"
      export TRANSFORMER_QUANT_COMPONENTS="none"
      export LWDETR_QUANT_COMPONENTS="linear"
      export LWDETR_EXP_MODE="${component%_exp}"
      ;;
    *)
      echo "Unknown sweep case: $case_name" >&2
      return 2
      ;;
  esac
}

safe_case_dir() {
  printf '%s\n' "$1" | tr ':/' '__'
}

if [[ ! -f "$csv_path" || "${SWEEP_OVERWRITE:-false}" == "true" ]]; then
  printf 'case,scope,component,status,exit_code,map_50_95,output_dir,log_file\n' > "$csv_path"
fi
if [[ "${SWEEP_OVERWRITE:-false}" == "true" ]]; then
  : > "$summary_log"
fi
{
  echo "Single-component PTQ sweep"
  echo "repo_root=$repo_root"
  echo "model_size=$model_size"
  echo "coco_path=$coco_path"
  echo "checkpoint_path=$checkpoint_path"
  echo "cases=$cases"
  echo "csv_path=$csv_path"
  echo
} >> "$summary_log"

for case_name in $cases; do
  scope="${case_name%%:*}"
  component="${case_name#*:}"
  case_dir="$(safe_case_dir "$case_name")"
  output_dir="$sweep_root/$case_dir"
  log_file="$sweep_root/logs/${case_dir}.log"
  mkdir -p "$output_dir"

  echo "==> Running PTQ case: $case_name"
  echo "    output: $output_dir"
  echo "    log:    $log_file"

  if ! configure_case "$case_name"; then
    write_csv_row "$case_name" "$scope" "$component" "invalid_case" "2" "" "$output_dir" "$log_file"
    {
      echo "$case_name: status=invalid_case exit_code=2 ap=NA"
    } >> "$summary_log"
    continue
  fi

  export MODEL_SIZE="$model_size"
  export OUTPUT_DIR="$output_dir"

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

  write_csv_row "$case_name" "$scope" "$component" "$status" "$exit_code" "$ap" "$output_dir" "$log_file"
  {
    echo "$case_name: status=$status exit_code=$exit_code ap=${ap:-NA} output_dir=$output_dir log_file=$log_file"
  } >> "$summary_log"
  echo "==> Finished $case_name: status=$status exit_code=$exit_code ap=${ap:-NA}"
done

echo "Wrote summary CSV: $csv_path"
echo "Wrote summary log: $summary_log"
