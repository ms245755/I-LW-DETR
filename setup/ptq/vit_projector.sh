#!/usr/bin/env bash
set -euo pipefail

SCENARIO=vit_projector exec bash "$(dirname "$0")/run.sh" "$@"
