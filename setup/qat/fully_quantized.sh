#!/usr/bin/env bash
set -euo pipefail

SCENARIO=fully_quantized exec bash "$(dirname "$0")/run.sh" "$@"
