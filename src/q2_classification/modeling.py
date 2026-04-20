"""Model and calibration helpers for Question 2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost_classifier": "XGBoost Classifier" if HAS_XGBOOST else "HistGradientBoosting",
    "mlp_classifier": "MLP Classifier",
}

STRATEGY_LABELS = {
    "no_resampling": "No Resampling",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
    "random_undersampling": "Random Undersampling",
    "cost_sensitive": "Cost Sensitive",
}


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Defines one imbalance-handling strategy."""

    key: str
    label: str
    sampler_factory: Callable[[dict], object] | None
    cost_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Defines one model + strategy pairing evaluated in CV."""

    key: str
    model_key: str
    model_label: str
    strategy_key: str
    strategy_label: str
    label: str
    builder: Callable[[dict, list[str], float], ImbPipeline]


class CalibratedBinaryClassifier(BaseEstimator, ClassifierMixin):
    """A lightweight prefit binary classifier wrapper with custom calibration."""

    def __init__(self, base_estimator: object, calibrator: object, method: str) -> None:
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.method = method

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CalibratedBinaryClassifier":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_scores = np.clip(self.base_estimator.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
        if self.method == "sigmoid":
            logits = np.log(raw_scores / (1.0 - raw_scores)).reshape(-1, 1)
            prob_pos = self.calibrator.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            prob_pos = self.calibrator.predict(raw_scores)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported calibration method: {self.method}")

        prob_pos = np.clip(np.asarray(prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def resolve_n_jobs(config: dict) -> int:
    """Convert config parallelism into a concrete worker count."""

    n_jobs = int(config.get("n_jobs", -1))
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, n_jobs)


def positive_class_ratio(y: pd.Series) -> float:
    """Return the negative-to-positive class ratio."""

    positives = max(1, int(y.sum()))
    negatives = int((y == 0).sum())
    return float(negatives / positives)


def build_preprocessor(scale_columns: list[str]) -> ColumnTransformer:
    """Build the feature preprocessor for the fraud dataset."""

    if not scale_columns:
        return ColumnTransformer(transformers=[], remainder="passthrough", sparse_threshold=0.0)

    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), scale_columns)],
        remainder="passthrough",
        sparse_threshold=0.0,
    )


def build_strategy_specs(config: dict) -> list[StrategySpec]:
    """Resolve the imbalance-handling strategies requested by config."""

    available = {
        "no_resampling": StrategySpec("no_resampling", STRATEGY_LABELS["no_resampling"], None, False),
        "smote": StrategySpec(
            "smote",
            STRATEGY_LABELS["smote"],
            lambda cfg: SMOTE(random_state=int(cfg.get("seed", 463))),
            False,
        ),
        "adasyn": StrategySpec(
            "adasyn",
            STRATEGY_LABELS["adasyn"],
            lambda cfg: ADASYN(random_state=int(cfg.get("seed", 463))),
            False,
        ),
        "random_undersampling": StrategySpec(
            "random_undersampling",
            STRATEGY_LABELS["random_undersampling"],
            lambda cfg: RandomUnderSampler(random_state=int(cfg.get("seed", 463))),
            False,
        ),
        "cost_sensitive": StrategySpec("cost_sensitive", STRATEGY_LABELS["cost_sensitive"], None, True),
    }

    strategies = []
    for key in config.get("strategies", []):
        if key not in available:
            raise ValueError(f"Unsupported Question 2 strategy: {key}")
        strategies.append(available[key])
    return strategies


def build_classifier(model_key: str, config: dict, strategy: StrategySpec, class_ratio: float) -> object:
    """Construct one classifier configured for a given strategy."""

    params = config.get("model_params", {})
    seed = int(config.get("seed", 463))
    n_jobs = min(4, resolve_n_jobs(config))

    if model_key == "logistic_regression":
        cfg = params.get("logistic_regression", {})
        return LogisticRegression(
            C=float(cfg.get("C", 1.0)),
            max_iter=int(cfg.get("max_iter", 2000)),
            solver="lbfgs",
            class_weight="balanced" if strategy.cost_sensitive else None,
        )

    if model_key == "random_forest":
        cfg = params.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 120)),
            max_depth=int(cfg.get("max_depth", 8)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            random_state=seed,
            n_jobs=n_jobs,
            class_weight="balanced" if strategy.cost_sensitive else None,
        )

    if model_key == "xgboost_classifier":
        cfg = params.get("xgboost_classifier", {})
        if HAS_XGBOOST:
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs,
                n_estimators=int(cfg.get("n_estimators", 150)),
                max_depth=int(cfg.get("max_depth", 4)),
                learning_rate=float(cfg.get("learning_rate", 0.08)),
                subsample=float(cfg.get("subsample", 0.9)),
                colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
                reg_lambda=float(cfg.get("reg_lambda", 1.0)),
                reg_alpha=float(cfg.get("reg_alpha", 0.0)),
                scale_pos_weight=float(class_ratio if strategy.cost_sensitive else 1.0),
                verbosity=0,
            )

        return HistGradientBoostingClassifier(
            learning_rate=float(cfg.get("learning_rate", 0.08)),
            max_depth=int(cfg.get("max_depth", 4)),
            max_iter=int(cfg.get("n_estimators", 150)),
            random_state=seed,
        )

    if model_key == "mlp_classifier":
        cfg = params.get("mlp_classifier", {})
        return MLPClassifier(
            hidden_layer_sizes=tuple(int(v) for v in cfg.get("hidden_layer_sizes", [64, 32])),
            alpha=float(cfg.get("alpha", 0.0005)),
            max_iter=int(cfg.get("max_iter", 60)),
            learning_rate_init=float(cfg.get("learning_rate_init", 0.001)),
            batch_size=int(cfg.get("batch_size", 2048)),
            early_stopping=bool(cfg.get("early_stopping", True)),
            validation_fraction=0.1,
            random_state=seed,
        )

    raise ValueError(f"Unsupported Question 2 model: {model_key}")


def build_pipeline(
    model_key: str,
    strategy: StrategySpec,
    config: dict,
    scale_columns: list[str],
    class_ratio: float,
) -> ImbPipeline:
    """Build a full imbalanced-learning pipeline for one candidate."""

    steps: list[tuple[str, object]] = [("preprocessor", build_preprocessor(scale_columns))]
    if strategy.sampler_factory is not None:
        steps.append(("sampler", strategy.sampler_factory(config)))
    steps.append(("classifier", build_classifier(model_key, config, strategy, class_ratio)))
    return ImbPipeline(steps=steps)


def build_candidate_specs(config: dict) -> list[CandidateSpec]:
    """Create all model + strategy combinations for evaluation."""

    candidates: list[CandidateSpec] = []
    for model_key in config.get("models", []):
        for strategy in build_strategy_specs(config):
            candidates.append(
                CandidateSpec(
                    key=f"{model_key}__{strategy.key}",
                    model_key=model_key,
                    model_label=MODEL_LABELS[model_key],
                    strategy_key=strategy.key,
                    strategy_label=strategy.label,
                    label=f"{MODEL_LABELS[model_key]} + {strategy.label}",
                    builder=lambda cfg, scale_cols, class_ratio, mk=model_key, st=strategy: build_pipeline(
                        mk, st, cfg, scale_cols, class_ratio
                    ),
                )
            )
    return candidates


def fit_pipeline(
    estimator: ImbPipeline,
    candidate: CandidateSpec,
    y_train: pd.Series,
    X_train: pd.DataFrame,
) -> ImbPipeline:
    """Fit one candidate pipeline, including sample-weight handling for MLP."""

    if candidate.strategy_key == "cost_sensitive" and candidate.model_key == "mlp_classifier":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        estimator.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        return estimator

    estimator.fit(X_train, y_train)
    return estimator


def extract_probability_scores(estimator: object, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a fitted model."""

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    raise ValueError("Estimator does not support probability outputs.")
