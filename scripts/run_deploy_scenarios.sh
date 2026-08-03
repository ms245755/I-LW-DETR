#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scenarios=(
  default
  safe_linear
  safe_vit_attention
  safe_all_attention
  safe_all_attention_norm_softmax
)

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_deploy_scenarios.sh <scenario|all> <onnx_path> <coco_path>

Examples:
  bash scripts/run_deploy_scenarios.sh default output/export/inference_model.sim.onnx /path/to/coco
  bash scripts/run_deploy_scenarios.sh safe_linear output/export/inference_model.sim.onnx /path/to/coco
  bash scripts/run_deploy_scenarios.sh all output/export/inference_model.sim.onnx /path/to/coco

Optional environment variables:
  OUT_DIR=deploy_runs/lwdetr_int8
  DEVICE=0
  IMAGE_SIZE=640
  CALIB_COUNT=500
  BATCH_SIZE=1
  WORKSPACE_GB=4
  PRECISION_CONSTRAINTS=prefer   # none|prefer|obey
  FP16=1                         # allow FP16 fallback
  RUN_BENCHMARK=1                # repeat inference 10x per image
  DISABLE_EVAL=1                 # latency only
  MAX_IMAGES=100                 # quick benchmark/eval subset
  IMAGE_DIR=/path/to/val2017
  ANN_FILE=/path/to/instances_val2017.json
  VERBOSE=1
EOF
}

contains_scenario() {
  local target="$1"
  local scenario
  for scenario in "${scenarios[@]}"; do
    [[ "$scenario" == "$target" ]] && return 0
  done
  return 1
}

run_scenario() {
  local scenario="$1"
  local onnx_path="$2"
  local coco_path="$3"

  local out_dir="${OUT_DIR:-deploy_runs/lwdetr_int8}"
  local device="${DEVICE:-0}"
  local image_size="${IMAGE_SIZE:-640}"
  local calib_count="${CALIB_COUNT:-500}"
  local batch_size="${BATCH_SIZE:-1}"
  local workspace_gb="${WORKSPACE_GB:-4}"
  local precision_constraints="${PRECISION_CONSTRAINTS:-prefer}"

  local scenario_dir="$out_dir/$scenario"
  local engine_path="$scenario_dir/inference_model.${scenario}.int8.engine"
  local cache_path="$scenario_dir/inference_model.${scenario}.calib.cache"
  local inspect_path="$scenario_dir/inspect.json"
  local build_log="$scenario_dir/build.log"
  local inspect_log="$scenario_dir/inspect.log"
  local benchmark_log="$scenario_dir/benchmark.log"

  mkdir -p "$scenario_dir"

  local common_data_args=()
  [[ -n "${IMAGE_DIR:-}" ]] && common_data_args+=(--image_dir "$IMAGE_DIR")
  [[ -n "${ANN_FILE:-}" ]] && common_data_args+=(--ann_file "$ANN_FILE")

  local build_flags=()
  [[ "${FP16:-1}" == "1" || "${FP16:-}" == "true" ]] && build_flags+=(--fp16)
  [[ "${VERBOSE:-0}" == "1" || "${VERBOSE:-}" == "true" ]] && build_flags+=(--verbose)

  local bench_flags=()
  [[ "${RUN_BENCHMARK:-0}" == "1" || "${RUN_BENCHMARK:-}" == "true" ]] && bench_flags+=(--run_benchmark)
  [[ "${DISABLE_EVAL:-0}" == "1" || "${DISABLE_EVAL:-}" == "true" ]] && bench_flags+=(--disable_eval)
  [[ -n "${MAX_IMAGES:-}" ]] && bench_flags+=(--max_images "$MAX_IMAGES")
  [[ -n "${IMAGE_DIR:-}" ]] && bench_flags+=(--image_dir "$IMAGE_DIR")

  echo "==> [$scenario] building TensorRT INT8 engine"
  python deploy/build_tensorrt_int8.py \
    --onnx "$onnx_path" \
    --engine "$engine_path" \
    --coco_path "$coco_path" \
    "${common_data_args[@]}" \
    --image_size "$image_size" \
    --calib_count "$calib_count" \
    --batch_size "$batch_size" \
    --device "$device" \
    --cache "$cache_path" \
    --workspace_gb "$workspace_gb" \
    --scenario "$scenario" \
    --precision_constraints "$precision_constraints" \
    "${build_flags[@]}" 2>&1 | tee "$build_log"

  echo "==> [$scenario] inspecting engine"
  python deploy/inspect_tensorrt_engine.py \
    --engine "$engine_path" \
    --output "$inspect_path" \
    --show_quantized_layers 2>&1 | tee "$inspect_log"

  echo "==> [$scenario] benchmarking/evaluating engine"
  python deploy/benchmark.py \
    --path "$engine_path" \
    --coco_path "$coco_path" \
    --device "$device" \
    --img_size "$image_size" \
    "${bench_flags[@]}" 2>&1 | tee "$benchmark_log"
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

target="$1"
onnx_path="$2"
coco_path="$3"

if [[ ! -f "$onnx_path" ]]; then
  echo "ONNX file does not exist: $onnx_path" >&2
  exit 2
fi

if [[ ! -d "$coco_path" ]]; then
  echo "COCO directory does not exist: $coco_path" >&2
  exit 2
fi

if [[ "$target" == "all" ]]; then
  for scenario in "${scenarios[@]}"; do
    run_scenario "$scenario" "$onnx_path" "$coco_path"
  done
elif contains_scenario "$target"; then
  run_scenario "$target" "$onnx_path" "$coco_path"
else
  echo "Unknown scenario: $target" >&2
  echo "Expected one of: ${scenarios[*]} all" >&2
  exit 2
fi
