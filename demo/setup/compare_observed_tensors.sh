#!/usr/bin/env bash
set -euo pipefail

usage="Usage: bash demo/setup/compare_observed_tensors.sh <observed_root> [components...]"

observed_root="${1:?$usage}"
shift || true

if [[ "$#" -gt 0 ]]; then
  components=("$@")
else
  components=(vit projector transformer lwdetr)
fi

python demo/generate_tensor_statistics.py \
  --components "${components[@]}" \
  --observed-root "$observed_root" \
  --skip-missing

python demo/read_comparision_result.py \
  --components "${components[@]}" \
  --observed-root "$observed_root" \
  --skip-missing
