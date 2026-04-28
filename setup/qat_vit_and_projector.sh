#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. Prefer setup/qat/vit_projector.sh.
if [[ $# -eq 3 ]]; then
  export MODEL_SIZE="$1"
  shift
fi

exec bash "$(dirname "$0")/qat/vit_projector.sh" "$@"
