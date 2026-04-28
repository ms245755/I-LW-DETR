#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. Prefer setup/ptq/<scenario>.sh.
if [[ $# -eq 3 ]]; then
  export MODEL_SIZE="$1"
  shift
fi

SCENARIO="${SCENARIO:-fully_quantized}" exec bash "$(dirname "$0")/ptq/run.sh" "$@"
