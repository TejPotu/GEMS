#!/usr/bin/env bash
# Run self-supervised pretraining. Defaults to the Phase 1 baseline (no Δm bias).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/experiment/phase1_baseline.yaml}
shift || true   # remaining args become OmegaConf dotlist overrides, e.g. model.dim=32

echo ">> pretraining with $CONFIG  $*"
gems pretrain --config "$CONFIG" "$@"
