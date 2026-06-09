#!/usr/bin/env bash
# Build the Δm vocabulary and the HDF5 spectrum corpus from the raw data/ assets.
set -euo pipefail
cd "$(dirname "$0")/.."

PKS_DIR=${1:-data/walking_calibrated_pks}
MDS_DIR=${2:-data/mds_csv}
OUT_DIR=${3:-data/processed}
LIMIT=${4:-}

mkdir -p "$OUT_DIR"

echo ">> building Δm vocabulary from $MDS_DIR"
gems build_vocab --mds_dir "$MDS_DIR" --out "$OUT_DIR/vocab.json"

echo ">> building HDF5 corpus from $PKS_DIR"
if [[ -n "$LIMIT" ]]; then
  gems build_corpus --pks_dir "$PKS_DIR" --out "$OUT_DIR/corpus.h5" --limit "$LIMIT"
else
  gems build_corpus --pks_dir "$PKS_DIR" --out "$OUT_DIR/corpus.h5"
fi
echo ">> done -> $OUT_DIR"
