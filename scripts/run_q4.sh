#!/usr/bin/env bash
set -euo pipefail

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m src.q4_clustering.pipeline --config configs/q4_clustering.yaml
