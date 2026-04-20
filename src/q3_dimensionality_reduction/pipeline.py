"""Starter runner for Question 3."""

from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.run import bootstrap_run
from src.common.seed import set_global_seed


SUMMARY = (
    "Starter scaffold for dimensionality reduction with PCA, Kernel PCA, "
    "t-SNE, UMAP, and an undercomplete autoencoder."
)

CHECKLIST = [
    "Load MNIST or Fashion-MNIST and standardise the preprocessing pipeline.",
    "Train PCA, Kernel PCA, t-SNE, UMAP, and autoencoder baselines.",
    "Record reconstruction metrics where applicable.",
    "Compute trustworthiness, continuity, and Kruskal stress for manifold methods.",
    "Train a k-NN classifier on reduced embeddings and compare cross-validation accuracy.",
    "Visualise latent spaces and discuss semantic structure and compute cost.",
]

EXPECTED_ARTIFACTS = [
    "2D embedding figures",
    "reconstruction error table",
    "trustworthiness and continuity table",
    "k-NN downstream accuracy table",
    "autoencoder latent-space visualisation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Question 3 workspace.")
    parser.add_argument(
        "--config",
        default="configs/q3_dimensionality_reduction.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(int(config.get("seed", 463)))
    run_dir = bootstrap_run(
        "q3_dimensionality_reduction",
        config,
        SUMMARY,
        CHECKLIST,
        EXPECTED_ARTIFACTS,
    )

    print(f"Initialized Question 3 scaffold at {run_dir}")
    print("Suggested first step: decide whether to use MNIST or Fashion-MNIST before tuning embeddings.")


if __name__ == "__main__":
    main()
