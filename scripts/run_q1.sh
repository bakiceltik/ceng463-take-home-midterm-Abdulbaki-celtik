#!/usr/bin/env bash
set -euo pipefail

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m src.q1_regression.pipeline --config configs/q1_regression.yaml
