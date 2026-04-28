#!/usr/bin/env bash
set -euo pipefail

SCENARIO=vit_only exec bash "$(dirname "$0")/run.sh" "$@"
