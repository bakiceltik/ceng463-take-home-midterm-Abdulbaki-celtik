"""Starter runner for Question 4."""

from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.run import bootstrap_run
from src.common.seed import set_global_seed


SUMMARY = (
    "Starter scaffold for clustering with K-Means, Gaussian Mixture Models, "
    "DBSCAN, Agglomerative Clustering, and ensemble stability analysis."
)

CHECKLIST = [
    "Confirm whether the chosen dataset has optional ground-truth labels.",
    "Tune K-Means, GMM, DBSCAN, and Agglomerative baselines with documented selection criteria.",
    "Measure internal clustering metrics and optional external validation metrics.",
    "Run bootstrap stability analysis using adjusted Rand similarity.",
    "Construct a simple cluster ensemble and compare it against individual methods.",
    "Visualise clusters with PCA or UMAP and discuss algorithmic assumptions.",
]

EXPECTED_ARTIFACTS = [
    "model-selection plots",
    "internal metrics table",
    "external validation table when labels exist",
    "stability analysis table",
    "cluster projection figure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Question 4 workspace.")
    parser.add_argument("--config", default="configs/q4_clustering.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(int(config.get("seed", 463)))
    run_dir = bootstrap_run("q4_clustering", config, SUMMARY, CHECKLIST, EXPECTED_ARTIFACTS)

    print(f"Initialized Question 4 scaffold at {run_dir}")
    print("Suggested first step: place the clustering dataset in data/external and inspect feature scales.")


if __name__ == "__main__":
    main()
