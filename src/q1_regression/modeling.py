"""Model construction helpers for Question 1."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - graceful fallback if xgboost is unusable
    HAS_XGBOOST = False


MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "ridge_cv": "RidgeCV",
    "lasso_cv": "LassoCV",
    "elastic_net": "Elastic Net CV",
    "xgboost_regressor": "XGBoost Regressor" if HAS_XGBOOST else "HistGradientBoosting",
    "huber_regressor": "Huber Regressor",
}


class ClippedLogTargetRegressor(BaseEstimator, RegressorMixin):
    """Apply log1p to the target and clip inverse predictions to a sane range."""

    def __init__(self, regressor: object, clip_margin: float = 0.25) -> None:
        self.regressor = regressor
        self.clip_margin = clip_margin

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClippedLogTargetRegressor":
        transformed = np.log1p(np.asarray(y, dtype=float))
        self.regressor_ = clone(self.regressor)
        self.lower_bound_ = float(np.min(transformed) - self.clip_margin)
        self.upper_bound_ = float(np.max(transformed) + self.clip_margin)
        self.regressor_.fit(X, transformed)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.asarray(self.regressor_.predict(X), dtype=float)
        predictions = np.clip(predictions, self.lower_bound_, self.upper_bound_)
        return np.expm1(predictions)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Describes a model option used in the outer evaluation loop."""

    key: str
    label: str
    description: str
    feature_degree: int
    builder: Callable[[dict, bool], object]


def resolve_n_jobs(config: dict) -> int:
    """Convert config-driven parallelism into a concrete value."""

    n_jobs = int(config.get("n_jobs", -1))
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, n_jobs)


def should_log_transform_target(y_train: pd.Series, config: dict) -> bool:
    """Decide whether a log transform is warranted for the target."""

    feature_cfg = config.get("feature_engineering", {})
    if not feature_cfg.get("use_log_target_if_skewed", True):
        return False

    if float(y_train.min()) <= -1:
        return False

    threshold = float(config.get("analysis", {}).get("target_skew_threshold", 0.75))
    return abs(float(skew(y_train.to_numpy()))) >= threshold


def make_preprocessor(degree: int) -> object:
    """Build the preprocessing stack for a model."""

    if degree <= 1:
        return Pipeline(
            steps=[
                ("identity", FunctionTransformer(validate=False)),
            ]
        )

    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )


def _wrap_target_transform(estimator: object, use_log_target: bool) -> object:
    if not use_log_target:
        return estimator

    return ClippedLogTargetRegressor(regressor=estimator)


def build_linear_regression(config: dict, use_log_target: bool) -> object:
    degree = int(config.get("feature_engineering", {}).get("default_polynomial_degree", 2))
    estimator = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(degree)),
            ("model", LinearRegression()),
        ]
    )
    return _wrap_target_transform(estimator, use_log_target)


def build_ridge_cv(config: dict, use_log_target: bool) -> object:
    degree = int(config.get("feature_engineering", {}).get("default_polynomial_degree", 2))
    estimator = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(degree)),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 40))),
        ]
    )
    return _wrap_target_transform(estimator, use_log_target)


def build_lasso_cv(config: dict, use_log_target: bool) -> object:
    degree = int(config.get("feature_engineering", {}).get("regularized_polynomial_degree", 3))
    estimator = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(degree)),
            (
                "model",
                LassoCV(
                    alphas=np.logspace(-3, 1, 16),
                    cv=int(config.get("tuning", {}).get("lasso_inner_cv_splits", 5)),
                    max_iter=15000,
                    tol=1e-3,
                    selection="random",
                    n_jobs=resolve_n_jobs(config),
                ),
            ),
        ]
    )
    return _wrap_target_transform(estimator, use_log_target)


def build_elastic_net_cv(config: dict, use_log_target: bool) -> object:
    degree = int(config.get("feature_engineering", {}).get("regularized_polynomial_degree", 3))
    estimator = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(degree)),
            (
                "model",
                ElasticNetCV(
                    alphas=np.logspace(-3, 1, 16),
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                    cv=int(config.get("tuning", {}).get("elastic_net_inner_cv_splits", 5)),
                    max_iter=15000,
                    tol=1e-3,
                    selection="random",
                    n_jobs=resolve_n_jobs(config),
                ),
            ),
        ]
    )
    return _wrap_target_transform(estimator, use_log_target)


def build_huber_regressor(config: dict, use_log_target: bool) -> object:
    degree = int(config.get("feature_engineering", {}).get("robust_polynomial_degree", 1))
    estimator = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(degree)),
            ("model", HuberRegressor(max_iter=5000, alpha=0.001)),
        ]
    )
    return estimator


def build_boosting_regressor(config: dict, use_log_target: bool) -> object:
    seed = int(config.get("seed", 463))
    search_iters = int(config.get("tuning", {}).get("xgboost_search_iterations", 6))
    inner_cv = int(config.get("tuning", {}).get("boosting_inner_cv_splits", 3))
    max_workers = min(4, resolve_n_jobs(config))

    if HAS_XGBOOST:
        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            tree_method="hist",
            n_jobs=max_workers,
            verbosity=0,
        )
        search_space = {
            "model__n_estimators": [150, 250, 350],
            "model__max_depth": [2, 3, 4, 6],
            "model__learning_rate": [0.03, 0.05, 0.1, 0.15],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [1.0, 2.0, 5.0],
        }
    else:
        base_model = HistGradientBoostingRegressor(random_state=seed)
        search_space = {
            "model__learning_rate": [0.03, 0.05, 0.1, 0.15],
            "model__max_depth": [None, 3, 5, 7],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(1)),
            ("model", base_model),
        ]
    )

    estimator = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=search_iters,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        random_state=seed,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    return _wrap_target_transform(estimator, use_log_target)


def build_model_specs(config: dict, use_log_target: bool) -> list[ModelSpec]:
    """Create the full ordered model list for the experiment."""

    base_degree = int(config.get("feature_engineering", {}).get("default_polynomial_degree", 2))
    regularized_degree = int(
        config.get("feature_engineering", {}).get("regularized_polynomial_degree", 3)
    )

    available_builders = {
        "linear_regression": (
            base_degree,
            "OLS baseline with polynomial feature expansion.",
            build_linear_regression,
        ),
        "ridge_cv": (
            base_degree,
            "Ridge regression with cross-validated alpha selection.",
            build_ridge_cv,
        ),
        "lasso_cv": (
            regularized_degree,
            "Lasso regression with sparse feature selection pressure.",
            build_lasso_cv,
        ),
        "elastic_net": (
            regularized_degree,
            "Elastic Net with tuned L1 ratio and alpha.",
            build_elastic_net_cv,
        ),
        "xgboost_regressor": (
            1,
            "Boosted trees with randomized hyperparameter search.",
            build_boosting_regressor,
        ),
        "huber_regressor": (
            base_degree,
            "Robust regression baseline to reduce outlier sensitivity.",
            build_huber_regressor,
        ),
    }

    specs: list[ModelSpec] = []
    for key in config.get("models", []):
        if key not in available_builders:
            raise ValueError(f"Unsupported model in config: {key}")
        degree, description, builder = available_builders[key]
        specs.append(
            ModelSpec(
                key=key,
                label=MODEL_LABELS[key],
                description=description,
                feature_degree=degree,
                builder=lambda cfg, transform, inner_builder=builder: inner_builder(cfg, transform),
            )
        )
    return specs


def unwrap_estimator(estimator: object) -> object:
    """Strip target transformation wrappers for downstream inspection."""

    if isinstance(estimator, TransformedTargetRegressor):
        return estimator.regressor_
    if isinstance(estimator, ClippedLogTargetRegressor):
        return estimator.regressor_
    return estimator


def extract_model_snapshot(estimator: object) -> dict[str, str]:
    """Capture tuned parameters for logging and reporting."""

    unwrapped = unwrap_estimator(estimator)
    model = unwrapped
    if hasattr(unwrapped, "named_steps"):
        model = unwrapped.named_steps["model"]

    snapshot: dict[str, str] = {}
    if hasattr(model, "alpha_"):
        snapshot["alpha"] = f"{float(model.alpha_):.6f}"
    if hasattr(model, "l1_ratio_"):
        snapshot["l1_ratio"] = f"{float(model.l1_ratio_):.4f}"
    if hasattr(model, "epsilon"):
        snapshot["epsilon"] = f"{float(model.epsilon):.4f}"
    if hasattr(model, "best_params_"):
        for key, value in model.best_params_.items():
            snapshot[key.replace("model__", "")] = str(value)
    return snapshot


def effective_feature_count(estimator: object, X_reference: pd.DataFrame) -> int:
    """Estimate the number of modeled predictors after preprocessing."""

    unwrapped = unwrap_estimator(estimator)
    if hasattr(unwrapped, "named_steps"):
        preprocessor = unwrapped.named_steps["preprocessor"]
        transformed = preprocessor.transform(X_reference.iloc[: min(64, len(X_reference))])
        if hasattr(transformed, "shape") and len(transformed.shape) == 2:
            return int(transformed.shape[1])
    return int(X_reference.shape[1])


def adjusted_r2_score(r2_value: float, n_samples: int, n_features: int) -> float:
    """Compute adjusted R-squared safely."""

    if n_samples <= n_features + 1:
        return math.nan
    return 1.0 - (1.0 - r2_value) * ((n_samples - 1) / (n_samples - n_features - 1))
