#!/usr/bin/env bash
set -euo pipefail

SCENARIO=vit_projector_transformer_float_head exec bash "$(dirname "$0")/run.sh" "$@"
