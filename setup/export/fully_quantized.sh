#!/usr/bin/env bash
set -euo pipefail

QUANT=true exec bash "$(dirname "$0")/run.sh" "$@"
