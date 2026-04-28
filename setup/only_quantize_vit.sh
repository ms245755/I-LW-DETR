#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. Prefer setup/ptq/vit_only.sh.
if [[ $# -eq 3 ]]; then
  export MODEL_SIZE="$1"
  shift
fi

exec bash "$(dirname "$0")/ptq/vit_only.sh" "$@"
