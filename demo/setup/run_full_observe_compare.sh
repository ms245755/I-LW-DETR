#!/usr/bin/env bash
set -euo pipefail

usage="Usage: bash demo/setup/run_full_observe_compare.sh <tiny|small|medium> <float_checkpoint> <quant_checkpoint> <image> [run_dir]"

model_size="${1:?$usage}"
float_checkpoint="${2:?$usage}"
quant_checkpoint="${3:?$usage}"
image_path="${4:?$usage}"
run_dir="${5:-${OBSERVATION_ROOT:-observation}/${model_size}_full}"

bash demo/run_observe_compare.sh full "$model_size" "$float_checkpoint" "$quant_checkpoint" "$image_path" "$run_dir"
