#!/usr/bin/env bash
# Run supervised fine-tuning of a frozen-backbone head.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/finetune/sample_classification.yaml}
shift || true

echo ">> fine-tuning with $CONFIG  $*"
gems finetune --config "$CONFIG" "$@"
