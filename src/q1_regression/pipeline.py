"""End-to-end regression workflow for Question 1."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from src.common.config import load_config
from src.common.io import ensure_dir, project_path, write_json
from src.common.run import bootstrap_run
from src.common.seed import set_global_seed
from src.q1_regression.analysis import (
    compute_residual_diagnostics,
    save_correlation_heatmap,
    save_cv_metric_summary,
    save_dataset_overview,
    save_feature_distributions,
    save_feature_space_summary,
    save_outlier_summary,
    save_pairwise_interactions,
    save_pairwise_significance_tests,
    save_residual_plots,
    save_rfe_feature_ranking,
    write_summary_markdown,
)
from src.q1_regression.data import load_dataset
from src.q1_regression.modeling import (
    adjusted_r2_score,
    build_model_specs,
    effective_feature_count,
    extract_model_snapshot,
    should_log_transform_target,
)


SUMMARY = (
    "Starter scaffold for the regression question focused on regularisation, "
    "feature engineering, repeated cross-validation, and residual analysis."
)

CHECKLIST = [
    "Run exploratory data analysis and save feature-distribution plots.",
    "Implement Linear Regression, RidgeCV, LassoCV, Elastic Net, and Gradient Boosting.",
    "Add polynomial features, interaction terms, and any justified target transform.",
    "Compute repeated 5-fold CV metrics with mean and standard deviation.",
    "Run paired statistical tests across comparable folds.",
    "Generate residual diagnostics and compare against a robust regression baseline.",
]

EXPECTED_ARTIFACTS = [
    "correlation heatmap",
    "outlier analysis summary",
    "cross-validation metrics table",
    "statistical significance table",
    "residual plots and Q-Q plot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Question 1 regression workflow.")
    parser.add_argument("--config", default="configs/q1_regression.yaml", help="Path to YAML config.")
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, n_features: int) -> dict[str, float]:
    """Compute the required regression metrics in original target space."""

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2_value = float(r2_score(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2_value,
        "adjusted_r2": float(adjusted_r2_score(r2_value, len(y_true), n_features)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_specs: list,
    config: dict,
    use_log_target: bool,
) -> pd.DataFrame:
    """Run repeated 5-fold CV and collect fold-level metrics for all models."""

    cv_cfg = config.get("cross_validation", {})
    splitter = RepeatedKFold(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        n_repeats=int(cv_cfg.get("n_repeats", 3)),
        random_state=int(config.get("seed", 463)),
    )

    rows: list[dict[str, float | int | str]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
        X_fold_valid = X_train.iloc[valid_idx].reset_index(drop=True)
        y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
        y_fold_valid = y_train.iloc[valid_idx].reset_index(drop=True)

        repeat_id = ((fold_index - 1) // int(cv_cfg.get("n_splits", 5))) + 1
        inner_fold_id = ((fold_index - 1) % int(cv_cfg.get("n_splits", 5))) + 1

        for spec in model_specs:
            estimator = spec.builder(config, use_log_target)
            estimator.fit(X_fold_train, y_fold_train)
            predictions = estimator.predict(X_fold_valid)
            feature_count = effective_feature_count(estimator, X_fold_train)
            metrics = compute_metrics(y_fold_valid, predictions, feature_count)

            row = {
                "fold_id": fold_index,
                "repeat_id": repeat_id,
                "inner_fold_id": inner_fold_id,
                "model_key": spec.key,
                "model_label": spec.label,
                "feature_degree": spec.feature_degree,
                **metrics,
            }
            row.update(extract_model_snapshot(estimator))
            rows.append(row)

    return pd.DataFrame(rows)


def fit_and_evaluate_final_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_specs: list,
    selected_keys: list[str],
    config: dict,
    use_log_target: bool,
    model_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit selected models on the full train split and evaluate once on test."""

    metrics_rows: list[dict[str, float | str]] = []
    diagnostics_rows: list[dict[str, float | str]] = []

    for spec in model_specs:
        if spec.key not in selected_keys:
            continue

        estimator = spec.builder(config, use_log_target)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        feature_count = effective_feature_count(estimator, X_train)
        metrics = compute_metrics(y_test, predictions, feature_count)
        diagnostics = compute_residual_diagnostics(y_test, predictions)

        metrics_rows.append(
            {
                "model_key": spec.key,
                "model_label": spec.label,
                **metrics,
            }
        )
        diagnostics_rows.append(
            {
                "model_key": spec.key,
                "model_label": spec.label,
                **diagnostics,
            }
        )

        prediction_frame = pd.DataFrame(
            {
                "actual": y_test,
                "predicted": predictions,
                "residual": y_test.to_numpy() - predictions,
            }
        )
        prediction_frame.to_csv(model_dir / f"{spec.key}_test_predictions.csv", index=False)
        joblib.dump(estimator, model_dir / f"{spec.key}.joblib")

    return pd.DataFrame(metrics_rows), pd.DataFrame(diagnostics_rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(int(config.get("seed", 463)))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    run_dir = bootstrap_run("q1_regression", config, SUMMARY, CHECKLIST, EXPECTED_ARTIFACTS)
    figures_dir = ensure_dir(run_dir / "figures")
    tables_dir = ensure_dir(run_dir / "tables")
    model_dir = ensure_dir(run_dir / "models")

    data_bundle = load_dataset(config, project_path())
    use_log_target = should_log_transform_target(data_bundle.y_train, config)
    print("Loaded dataset and resolved holdout split.")

    save_dataset_overview(
        data_bundle.X_train,
        data_bundle.X_test,
        data_bundle.y_train,
        data_bundle.y_test,
        use_log_target,
        tables_dir / "dataset_overview.json",
    )
    save_feature_distributions(
        pd.concat([data_bundle.X_train, data_bundle.y_train.rename("MedHouseVal")], axis=1),
        figures_dir / "feature_distributions.png",
    )
    target_correlations = save_correlation_heatmap(
        pd.concat([data_bundle.X_train, data_bundle.y_train.rename("MedHouseVal")], axis=1),
        figures_dir / "correlation_heatmap.png",
    )
    save_pairwise_interactions(
        pd.concat([data_bundle.X_train, data_bundle.y_train.rename("MedHouseVal")], axis=1),
        target_correlations,
        figures_dir / "pairwise_interactions.png",
        sample_size=int(config.get("analysis", {}).get("eda_sample_size", 1200)),
    )
    save_outlier_summary(
        pd.concat([data_bundle.X_train, data_bundle.y_train.rename("MedHouseVal")], axis=1),
        tables_dir / "outlier_summary.csv",
        zscore_threshold=float(config.get("analysis", {}).get("zscore_threshold", 3.0)),
    )
    save_feature_space_summary(
        data_bundle.X_train,
        config.get("feature_engineering", {}).get("polynomial_degrees", [2, 3]),
        tables_dir / "feature_space_summary.csv",
    )
    save_rfe_feature_ranking(
        data_bundle.X_train,
        np.log1p(data_bundle.y_train) if use_log_target else data_bundle.y_train,
        degree=max(config.get("feature_engineering", {}).get("polynomial_degrees", [2, 3])),
        top_k=int(config.get("analysis", {}).get("rfe_top_k", 20)),
        output_path=tables_dir / "rfe_feature_ranking.csv",
    )
    print("Saved EDA, outlier, and feature-engineering artifacts.")

    model_specs = build_model_specs(config, use_log_target)
    fold_results = evaluate_models(
        data_bundle.X_train,
        data_bundle.y_train,
        model_specs,
        config,
        use_log_target,
    )
    print("Finished repeated cross-validation across all Question 1 models.")
    fold_results.to_csv(tables_dir / "cross_validation_fold_metrics.csv", index=False)
    cv_summary = save_cv_metric_summary(fold_results, tables_dir / "cross_validation_summary.csv")
    significance = save_pairwise_significance_tests(
        fold_results,
        tables_dir / "pairwise_significance_tests_rmse.csv",
    )

    best_model_key = str(cv_summary.iloc[0]["model_key"])
    selected_final_models = [best_model_key]
    if "huber_regressor" not in selected_final_models:
        selected_final_models.append("huber_regressor")

    final_metrics, diagnostics = fit_and_evaluate_final_models(
        data_bundle.X_train,
        data_bundle.X_test,
        data_bundle.y_train,
        data_bundle.y_test,
        model_specs,
        selected_final_models,
        config,
        use_log_target,
        model_dir,
    )
    print("Completed final holdout evaluation and saved trained models.")
    final_metrics.to_csv(tables_dir / "final_test_metrics.csv", index=False)
    diagnostics.to_csv(tables_dir / "residual_diagnostics.csv", index=False)

    for row in final_metrics.itertuples(index=False):
        predictions = pd.read_csv(model_dir / f"{row.model_key}_test_predictions.csv")
        save_residual_plots(
            y_true=predictions["actual"],
            y_pred=predictions["predicted"].to_numpy(),
            model_label=row.model_label,
            output_dir=figures_dir,
        )

    best_row = final_metrics.loc[final_metrics["model_key"] == best_model_key].iloc[0]
    huber_row = final_metrics.loc[final_metrics["model_key"] == "huber_regressor"].iloc[0]
    best_diag = diagnostics.loc[diagnostics["model_key"] == best_model_key].iloc[0]
    huber_diag = diagnostics.loc[diagnostics["model_key"] == "huber_regressor"].iloc[0]
    robust_note = (
        f"- Robust regression comparison: Huber RMSE was {huber_row['rmse']:.4f} versus "
        f"{best_row['rmse']:.4f} for the selected best model. "
        f"Breusch-Pagan p-values were {huber_diag['breusch_pagan_pvalue']:.4f} and "
        f"{best_diag['breusch_pagan_pvalue']:.4f}, respectively."
    )

    dataset_overview = {
        "train_rows": int(len(data_bundle.X_train)),
        "test_rows": int(len(data_bundle.X_test)),
        "feature_count": int(data_bundle.X_train.shape[1]),
        "target_skew_train": float(data_bundle.y_train.skew()),
        "log_target_transform": bool(use_log_target),
    }
    write_summary_markdown(
        run_dir / "summary.md",
        dataset_overview=dataset_overview,
        cv_summary=cv_summary,
        final_metrics=final_metrics,
        diagnostics=diagnostics,
        significance=significance,
        robust_note=robust_note,
    )
    write_json(
        run_dir / "final_selection.json",
        {
            "best_model_key": best_model_key,
            "best_model_label": str(cv_summary.iloc[0]["model_label"]),
            "selected_final_models": selected_final_models,
        },
    )

    print(f"Completed Question 1 workflow at {run_dir}")
    print(f"Best cross-validated model: {cv_summary.iloc[0]['model_label']}")
    print("Artifacts saved under figures/, tables/, and models/.")


if __name__ == "__main__":
    main()
