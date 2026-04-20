"""Starter runner for Question 5."""

from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.run import bootstrap_run
from src.common.seed import set_global_seed


SUMMARY = (
    "Starter scaffold for deep learning experiments with a deep MLP, CNN, "
    "transfer learning, interpretability, and adversarial robustness."
)

CHECKLIST = [
    "Implement the deep MLP with dropout, batch norm, and early stopping.",
    "Implement the CNN with data augmentation and tracked train-validation curves.",
    "Fine-tune a pretrained ResNet18 or similar transfer-learning baseline.",
    "Run Optuna over learning rate, batch size, dropout, and weight decay.",
    "Collect Grad-CAM and LIME outputs for misclassified samples.",
    "Measure adversarial robustness with FGSM or PGD and compare failure modes.",
]

EXPECTED_ARTIFACTS = [
    "training history plots",
    "final metrics table",
    "confusion matrix",
    "Grad-CAM visualisations",
    "adversarial robustness comparison table",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Question 5 workspace.")
    parser.add_argument("--config", default="configs/q5_neural_networks.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(int(config.get("seed", 463)))
    run_dir = bootstrap_run("q5_neural_networks", config, SUMMARY, CHECKLIST, EXPECTED_ARTIFACTS)

    print(f"Initialized Question 5 scaffold at {run_dir}")
    print("Suggested first step: verify whether you want CIFAR-10 or CIFAR-100 before training.")


if __name__ == "__main__":
    main()
