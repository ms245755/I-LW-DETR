#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

if [[ $# -ne 2 ]]; then
  cat >&2 <<EOF
Usage: bash setup/ptq/sweep_vit_components.sh <coco_path> <checkpoint_path_or_dir>

Runs PTQ sequentially with one ViT encoder component quantized at a time.

Optional environment variables:
  MODEL_SIZE=tiny|small|medium        Default: small
  SWEEP_OUTPUT_ROOT=output/ptq_vit_component_sweep
  SWEEP_CSV=<path>                    Default: <SWEEP_OUTPUT_ROOT>/<model_size>/summary.csv
  SWEEP_LOG=<path>                    Default: <SWEEP_OUTPUT_ROOT>/<model_size>/summary.log
  COMPONENTS="linear conv2d ..."      Override component list
  NPROC_PER_NODE=1|2|4|8              Forwarded to setup/ptq/vit_only.sh
  CALIB_BATCHES=500                   Forwarded to setup/ptq/vit_only.sh
  CALIB_EVAL=true                     Forwarded to setup/ptq/vit_only.sh
  ACT_OBSERVER=mean|ema_minmax        Forwarded to setup/ptq/vit_only.sh
  ACT_PERCENTILE=99|99.9              Forwarded to setup/ptq/vit_only.sh
EOF
  exit 2
fi

coco_path="$1"
checkpoint_path="$2"
model_size="${MODEL_SIZE:-small}"
sweep_root="${SWEEP_OUTPUT_ROOT:-output/ptq_vit_component_sweep}/${model_size}"
csv_path="${SWEEP_CSV:-$sweep_root/summary.csv}"
summary_log="${SWEEP_LOG:-$sweep_root/summary.log}"
components="${COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
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

write_csv_row() {
  local component="$1"
  local status="$2"
  local exit_code="$3"
  local ap="$4"
  local output_dir="$5"
  local log_file="$6"

  {
    csv_escape "$component"; printf ','
    csv_escape "$status"; printf ','
    csv_escape "$exit_code"; printf ','
    csv_escape "$ap"; printf ','
    csv_escape "$output_dir"; printf ','
    csv_escape "$log_file"; printf '\n'
  } >> "$csv_path"
}

is_valid_component() {
  case "$1" in
    linear|conv2d|matmul|softmax|gelu|layernorm)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

printf 'component,status,exit_code,ap_50_95,output_dir,log_file\n' > "$csv_path"
{
  echo "ViT component PTQ sweep"
  echo "repo_root=$repo_root"
  echo "model_size=$model_size"
  echo "coco_path=$coco_path"
  echo "checkpoint_path=$checkpoint_path"
  echo "components=$components"
  echo "csv_path=$csv_path"
  echo
} > "$summary_log"

for component in $components; do
  output_dir="$sweep_root/$component"
  log_file="$sweep_root/logs/${component}.log"
  mkdir -p "$output_dir"

  echo "==> Running PTQ ViT component: $component"
  echo "    output: $output_dir"
  echo "    log:    $log_file"

  if ! is_valid_component "$component"; then
    echo "Unknown ViT component: $component" >&2
    write_csv_row "$component" "invalid_component" "2" "" "$output_dir" "$log_file"
    {
      echo "$component: status=invalid_component exit_code=2 ap=NA"
    } >> "$summary_log"
    continue
  fi

  export MODEL_SIZE="$model_size"
  export OUTPUT_DIR="$output_dir"
  export QUANT_MODEL_COMPONENTS="backbone"
  export BACKBONE_QUANT_COMPONENTS="vit"
  export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
  export ENCODER_QUANT_COMPONENTS="$component"
  # The projector is not selected in BACKBONE_QUANT_COMPONENTS, but the shared
  # PTQ wrapper still forwards this arg and argparse requires a valid choice.
  export PROJECTOR_QUANT_COMPONENTS="${PROJECTOR_QUANT_COMPONENTS:-conv2d}"
  export TRANSFORMER_QUANT_COMPONENTS="none"
  export LWDETR_QUANT_COMPONENTS="none"

  set +e
  bash "$script_dir/vit_only.sh" "$coco_path" "$checkpoint_path" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  ap="$(extract_ap "$log_file")"
  if [[ "$exit_code" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi

  write_csv_row "$component" "$status" "$exit_code" "$ap" "$output_dir" "$log_file"
  {
    echo "$component: status=$status exit_code=$exit_code ap=${ap:-NA} output_dir=$output_dir log_file=$log_file"
  } >> "$summary_log"
  echo "==> Finished $component: status=$status exit_code=$exit_code ap=${ap:-NA}"
done

echo "Wrote summary CSV: $csv_path"
echo "Wrote summary log: $summary_log"
