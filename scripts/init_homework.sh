#!/usr/bin/env bash
set -euo pipefail

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m src.q1_regression.pipeline --config configs/q1_regression.yaml
"$PYTHON_BIN" -m src.q2_classification.pipeline --config configs/q2_classification.yaml
"$PYTHON_BIN" -m src.q3_dimensionality_reduction.pipeline --config configs/q3_dimensionality_reduction.yaml
"$PYTHON_BIN" -m src.q4_clustering.pipeline --config configs/q4_clustering.yaml
"$PYTHON_BIN" -m src.q5_neural_networks.pipeline --config configs/q5_neural_networks.yaml
