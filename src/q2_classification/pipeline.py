"""End-to-end classification workflow for Question 2."""

from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.common.config import load_config
from src.common.io import ensure_dir, project_path, write_json
from src.common.run import bootstrap_run
from src.common.seed import set_global_seed
from src.q2_classification.analysis import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_reliability_diagram,
    plot_roc_curve,
    render_q2_latex_section,
    save_class_balance_plot,
    save_cv_summary,
    save_dataset_overview,
    write_summary_markdown,
)
from src.q2_classification.data import load_dataset
from src.q2_classification.modeling import (
    CalibratedBinaryClassifier,
    build_candidate_specs,
    extract_probability_scores,
    fit_pipeline,
    positive_class_ratio,
)


SUMMARY = (
    "End-to-end workflow for the extreme-imbalance classification question with "
    "resampling, cost-sensitive learning, calibration, and threshold tuning."
)

CHECKLIST = [
    "Validate class imbalance ratio and stratified split strategy.",
    "Implement leakage-safe pipelines for SMOTE, ADASYN, and random undersampling.",
    "Compare Logistic Regression, Random Forest, XGBoost, and an MLP baseline.",
    "Evaluate class weights against resampling strategies.",
    "Calibrate the top two models with sigmoid or isotonic regression.",
    "Choose a recall-first threshold and discuss operational costs.",
]

EXPECTED_ARTIFACTS = [
    "imbalance summary table",
    "cross-validation metrics table",
    "precision-recall curves",
    "reliability diagram",
    "confusion matrix for final threshold",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the full Question 2 classification workflow.")
    parser.add_argument("--config", default="configs/q2_classification.yaml", help="Path to YAML config.")
    return parser.parse_args()


def compute_fold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute the required classification metrics for one prediction set."""

    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_columns: list[str],
    config: dict,
) -> tuple[pd.DataFrame, list]:
    """Run stratified cross-validation across all model-strategy candidates."""

    splitter = StratifiedKFold(
        n_splits=int(config.get("cross_validation", {}).get("n_splits", 5)),
        shuffle=bool(config.get("cross_validation", {}).get("shuffle", True)),
        random_state=int(config.get("seed", 463)),
    )
    class_ratio = positive_class_ratio(y_train)
    candidates = build_candidate_specs(config)
    rows: list[dict[str, float | int | str]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
        X_fold_valid = X_train.iloc[valid_idx].reset_index(drop=True)
        y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
        y_fold_valid = y_train.iloc[valid_idx].reset_index(drop=True)

        for candidate in candidates:
            estimator = candidate.builder(config, scale_columns, class_ratio)
            estimator = fit_pipeline(estimator, candidate, y_fold_train, X_fold_train)
            probabilities = extract_probability_scores(estimator, X_fold_valid)
            metrics = compute_fold_metrics(y_fold_valid, probabilities, threshold=0.5)
            rows.append(
                {
                    "fold_id": fold_index,
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "model_key": candidate.model_key,
                    "model_label": candidate.model_label,
                    "strategy_key": candidate.strategy_key,
                    "strategy_label": candidate.strategy_label,
                    **metrics,
                }
            )

    return pd.DataFrame(rows), candidates


def fit_calibrator(
    base_estimator,
    method: str,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> CalibratedBinaryClassifier:
    """Fit one calibration wrapper on top of a prefit estimator."""

    base_scores = np.clip(extract_probability_scores(base_estimator, X_calibration), 1e-6, 1.0 - 1e-6)
    if method == "sigmoid":
        logits = np.log(base_scores / (1.0 - base_scores)).reshape(-1, 1)
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(logits, y_calibration)
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(base_scores, y_calibration)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported calibration method: {method}")

    return CalibratedBinaryClassifier(base_estimator=base_estimator, calibrator=calibrator, method=method)


def select_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """Compute the F1-max and recall-first cost-sensitive thresholds."""

    grid_size = int(config.get("analysis", {}).get("threshold_grid_size", 500))
    unique_thresholds = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, grid_size)))
    unique_thresholds = np.unique(np.concatenate([[0.0, 0.5, 1.0], unique_thresholds]))

    rows = []
    fn_cost = float(config.get("threshold_selection", {}).get("false_negative_cost", 5.0))
    fp_cost = float(config.get("threshold_selection", {}).get("false_positive_cost", 1.0))

    for threshold in unique_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        total_cost = fn_cost * fn + fp_cost * fp
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "total_cost": float(total_cost),
            }
        )

    threshold_frame = pd.DataFrame(rows)
    f1_row = threshold_frame.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    cost_row = threshold_frame.sort_values(["total_cost", "recall", "threshold"], ascending=[True, False, True]).iloc[0]

    return pd.DataFrame(
        [
            {"threshold_policy": "f1_max", **f1_row.to_dict()},
            {"threshold_policy": "recall_first_cost_sensitive", **cost_row.to_dict()},
        ]
    )


def evaluate_holdout(
    estimator,
    candidate_label: str,
    threshold_summary: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Evaluate one calibrated candidate on the holdout set under both thresholds."""

    probabilities = estimator.predict_proba(X_test)[:, 1]
    rows = []
    prediction_rows = []
    for _, threshold_row in threshold_summary.iterrows():
        threshold = float(threshold_row["threshold"])
        predictions = (probabilities >= threshold).astype(int)
        metrics = compute_fold_metrics(y_test, probabilities, threshold=threshold)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        rows.append(
            {
                "candidate_label": candidate_label,
                "threshold_policy": threshold_row["threshold_policy"],
                "threshold": threshold,
                **metrics,
                "brier_score": float(brier_score_loss(y_test, probabilities)),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "true_positives": int(tp),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "actual": y_test,
                    "predicted_probability": probabilities,
                    "predicted_label": predictions,
                    "threshold_policy": threshold_row["threshold_policy"],
                }
            )
        )

    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True), probabilities


def main() -> None:
    """Run the full Question 2 workflow."""

    args = parse_args()
    config = load_config(args.config)
    set_global_seed(int(config.get("seed", 463)))
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    run_dir = bootstrap_run("q2_classification", config, SUMMARY, CHECKLIST, EXPECTED_ARTIFACTS)
    figures_dir = ensure_dir(run_dir / "figures")
    tables_dir = ensure_dir(run_dir / "tables")
    model_dir = ensure_dir(run_dir / "models")

    bundle = load_dataset(config, project_path())
    print("Loaded Question 2 dataset and created the holdout split.")

    save_dataset_overview(bundle, tables_dir / "dataset_overview.json")
    save_class_balance_plot(bundle, figures_dir / "class_balance.png")
    print("Saved dataset overview and imbalance artifacts.")

    fold_results, candidates = evaluate_candidates(bundle.X_train, bundle.y_train, bundle.scale_columns, config)
    fold_results.to_csv(tables_dir / "cross_validation_fold_metrics.csv", index=False)
    cv_summary = save_cv_summary(fold_results, tables_dir / "cross_validation_summary.csv")
    print("Finished stratified cross-validation for all Question 2 candidates.")

    top_two = cv_summary.head(2).copy()
    calibration_fraction = float(config.get("split", {}).get("calibration_validation_fraction", 0.15))
    X_model_train, X_calibration, y_model_train, y_calibration = train_test_split(
        bundle.X_train,
        bundle.y_train,
        test_size=calibration_fraction,
        stratify=bundle.y_train,
        random_state=int(config.get("seed", 463)),
    )

    candidate_lookup = {candidate.key: candidate for candidate in candidates}
    calibration_rows: list[dict[str, float | str | bool]] = []
    calibrated_models: dict[str, tuple[object, pd.DataFrame]] = {}

    for _, summary_row in top_two.iterrows():
        candidate = candidate_lookup[str(summary_row["candidate_key"])]
        class_ratio = positive_class_ratio(y_model_train)
        base_estimator = candidate.builder(config, bundle.scale_columns, class_ratio)
        base_estimator = fit_pipeline(base_estimator, candidate, y_model_train, X_model_train)

        best_method = None
        best_brier = None
        best_calibrated = None
        best_thresholds = None

        for method in config.get("calibration", {}).get("methods", ["sigmoid", "isotonic"]):
            calibrated = fit_calibrator(base_estimator, method, X_calibration, y_calibration)
            calibrated_probabilities = calibrated.predict_proba(X_calibration)[:, 1]
            brier = float(brier_score_loss(y_calibration, calibrated_probabilities))
            thresholds = select_thresholds(y_calibration, calibrated_probabilities, config)
            calibration_rows.append(
                {
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "calibration_method": method,
                    "brier_score": brier,
                    "pr_auc": float(average_precision_score(y_calibration, calibrated_probabilities)),
                    "roc_auc": float(roc_auc_score(y_calibration, calibrated_probabilities)),
                    "selected_for_final_model": False,
                }
            )
            if best_brier is None or brier < best_brier:
                best_brier = brier
                best_method = method
                best_calibrated = calibrated
                best_thresholds = thresholds

        assert best_method is not None and best_calibrated is not None and best_thresholds is not None
        calibrated_models[candidate.key] = (best_calibrated, best_thresholds)

        for row in calibration_rows:
            if row["candidate_key"] == candidate.key and row["calibration_method"] == best_method:
                row["selected_for_final_model"] = True

        joblib.dump(best_calibrated, model_dir / f"{candidate.key}_calibrated.joblib")

    calibration_summary = pd.DataFrame(calibration_rows).sort_values(
        ["selected_for_final_model", "brier_score"],
        ascending=[False, True],
    )
    calibration_summary.to_csv(tables_dir / "calibration_comparison.csv", index=False)
    calibration_summary[["candidate_label", "calibration_method", "brier_score", "selected_for_final_model"]].to_csv(
        tables_dir / "brier_score_summary.csv",
        index=False,
    )

    final_candidate_key = calibration_summary.loc[calibration_summary["selected_for_final_model"]].sort_values(
        ["brier_score", "pr_auc", "roc_auc"],
        ascending=[True, False, False],
    ).iloc[0]["candidate_key"]
    final_candidate_label = calibration_summary.loc[
        (calibration_summary["candidate_key"] == final_candidate_key)
        & (calibration_summary["selected_for_final_model"])
    ].iloc[0]["candidate_label"]

    final_estimator, final_thresholds = calibrated_models[str(final_candidate_key)]
    final_metrics, final_predictions, final_probabilities = evaluate_holdout(
        final_estimator,
        str(final_candidate_label),
        final_thresholds,
        bundle.X_test,
        bundle.y_test,
    )
    final_metrics.to_csv(tables_dir / "final_test_metrics.csv", index=False)
    final_predictions.to_csv(model_dir / "best_model_holdout_predictions.csv", index=False)
    final_thresholds.to_csv(tables_dir / "threshold_summary.csv", index=False)
    print("Completed calibration, threshold selection, and holdout evaluation.")

    default_policy = str(config.get("threshold_selection", {}).get("default_decision_policy", "recall_first_cost_sensitive"))
    default_metrics = final_metrics.loc[final_metrics["threshold_policy"] == default_policy].iloc[0]
    default_predictions = (final_probabilities >= float(default_metrics["threshold"])).astype(int)

    plot_precision_recall_curve(
        bundle.y_test.to_numpy(),
        final_probabilities,
        figures_dir / "best_model_precision_recall_curve.png",
        f"{final_candidate_label} - Precision-Recall Curve",
    )
    plot_roc_curve(
        bundle.y_test.to_numpy(),
        final_probabilities,
        figures_dir / "best_model_roc_curve.png",
        f"{final_candidate_label} - ROC Curve",
    )
    plot_reliability_diagram(
        bundle.y_test.to_numpy(),
        final_probabilities,
        figures_dir / "best_model_reliability.png",
        f"{final_candidate_label} - Reliability Diagram",
        n_bins=int(config.get("analysis", {}).get("calibration_curve_bins", 10)),
    )
    plot_confusion_matrix(
        bundle.y_test.to_numpy(),
        default_predictions,
        figures_dir / "best_model_confusion_matrix.png",
        f"{final_candidate_label} - Confusion Matrix ({default_policy})",
    )

    dataset_overview = {
        "train_rows": int(len(bundle.X_train)),
        "test_rows": int(len(bundle.X_test)),
        "feature_count": int(bundle.X_train.shape[1]),
        "positive_rate": float(bundle.positive_rate),
        "imbalance_ratio": float(bundle.imbalance_ratio),
    }
    write_summary_markdown(
        run_dir / "summary.md",
        dataset_overview=dataset_overview,
        cv_summary=cv_summary,
        calibration_summary=calibration_summary,
        final_metrics=final_metrics,
        threshold_summary=final_thresholds,
    )
    write_json(
        run_dir / "final_selection.json",
        {
            "best_cross_validation_candidate": str(cv_summary.iloc[0]["candidate_key"]),
            "top_two_candidates": top_two["candidate_key"].tolist(),
            "final_candidate_key": str(final_candidate_key),
            "final_candidate_label": str(final_candidate_label),
            "default_threshold_policy": default_policy,
        },
    )

    if bool(config.get("report", {}).get("write_section", True)):
        section_path = project_path(config.get("report", {}).get("section_path", "reports/sections/q2_classification.tex"))
        render_q2_latex_section(
            section_path,
            run_dir=run_dir,
            dataset_overview=dataset_overview,
            cv_summary=cv_summary,
            calibration_summary=calibration_summary,
            final_metrics=final_metrics,
            threshold_summary=final_thresholds,
        )
        print(f"Wrote Question 2 LaTeX section to {section_path}")

    print(f"Completed Question 2 workflow at {run_dir}")
    print(f"Best final candidate: {final_candidate_label}")
    print("Artifacts saved under figures/, tables/, and models/.")


if __name__ == "__main__":
    main()
