#!/usr/bin/env bash
# CPU end-to-end skeleton check: exercises the concrete pyc2mc-backed path and reports where the
# stubs begin. Once the build-order stubs are filled in, this should complete one masked-peak step.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> gems smoke"
gems smoke

echo ">> pytest"
pytest -q
