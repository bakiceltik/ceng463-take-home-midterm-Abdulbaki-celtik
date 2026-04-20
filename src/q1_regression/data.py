"""Dataset loading utilities for Question 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "MedHouseVal"


@dataclass(slots=True)
class RegressionDataBundle:
    """Container for the regression dataset and train-test split."""

    full_frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_california_housing(cache_path: Path) -> pd.DataFrame:
    """Load and locally cache the California Housing dataset."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return pd.read_csv(cache_path)

    dataset = fetch_california_housing(as_frame=True, data_home=str(cache_path.parent / "sklearn_cache"))
    frame = dataset.frame.copy()
    frame.to_csv(cache_path, index=False)
    return frame


def load_dataset(config: dict, project_root: Path) -> RegressionDataBundle:
    """Resolve the configured dataset and perform the single holdout split."""

    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "california_housing")
    test_size = float(dataset_cfg.get("test_size", 0.2))
    random_state = int(config.get("seed", 463))

    if dataset_name != "california_housing":
        raise ValueError(f"Unsupported Question 1 dataset: {dataset_name}")

    cache_path = project_root / dataset_cfg.get("cache_path", "data/raw/california_housing.csv")
    frame = load_california_housing(cache_path)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    features = frame.drop(columns=[TARGET_COLUMN]).copy()
    target = frame[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    return RegressionDataBundle(
        full_frame=frame,
        features=features,
        target=target,
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
