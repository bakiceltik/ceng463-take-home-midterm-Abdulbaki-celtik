"""Dataset loading utilities for Question 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class ClassificationDataBundle:
    """Container for the classification dataset and fixed holdout split."""

    full_frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scale_columns: list[str]
    passthrough_columns: list[str]
    class_counts: dict[int, int]
    positive_rate: float
    imbalance_ratio: float


def load_dataset(config: dict, project_root: Path) -> ClassificationDataBundle:
    """Load the configured credit-card fraud dataset and perform one holdout split."""

    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "credit_card_fraud")
    if dataset_name != "credit_card_fraud":
        raise ValueError(f"Unsupported Question 2 dataset: {dataset_name}")

    dataset_path = project_root / dataset_cfg.get("path", "data/external/creditcard.csv")
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Question 2 dataset not found. Expected the credit-card fraud CSV at "
            f"'{dataset_path}'. Please place 'creditcard.csv' there and rerun."
        )

    frame = pd.read_csv(dataset_path)
    target_column = dataset_cfg.get("target_column", "Class")
    if target_column not in frame.columns:
        raise ValueError(
            f"Question 2 dataset is missing the target column '{target_column}'. "
            f"Available columns start with: {list(frame.columns[:10])}"
        )

    features = frame.drop(columns=[target_column]).copy()
    target = frame[target_column].astype(int).copy()

    scale_columns = [
        column
        for column in config.get("preprocessing", {}).get("scale_columns", ["Time", "Amount"])
        if column in features.columns
    ]
    passthrough_columns = [column for column in features.columns if column not in scale_columns]

    positive_count = int(target.sum())
    negative_count = int((target == 0).sum())
    positive_rate = float(positive_count / len(target))
    imbalance_ratio = float(negative_count / max(1, positive_count))

    holdout_test_size = float(dataset_cfg.get("holdout_test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=holdout_test_size,
        stratify=target,
        random_state=int(config.get("seed", 463)),
    )

    return ClassificationDataBundle(
        full_frame=frame,
        features=features,
        target=target,
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        scale_columns=scale_columns,
        passthrough_columns=passthrough_columns,
        class_counts={0: negative_count, 1: positive_count},
        positive_rate=positive_rate,
        imbalance_ratio=imbalance_ratio,
    )
