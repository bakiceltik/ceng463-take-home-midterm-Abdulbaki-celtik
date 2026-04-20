"""Analysis, plotting, and report helpers for Question 2."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

TEMP_CACHE_ROOT = Path(tempfile.gettempdir()) / "ceng463-matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.common.io import write_json, write_text


def save_dataset_overview(bundle, output_path: Path) -> None:
    """Persist dataset-level classification metadata."""

    payload = {
        "train_rows": int(len(bundle.X_train)),
        "test_rows": int(len(bundle.X_test)),
        "feature_count": int(bundle.X_train.shape[1]),
        "negative_count": int(bundle.class_counts[0]),
        "positive_count": int(bundle.class_counts[1]),
        "positive_rate": float(bundle.positive_rate),
        "imbalance_ratio": float(bundle.imbalance_ratio),
        "scale_columns": bundle.scale_columns,
        "passthrough_columns_count": int(len(bundle.passthrough_columns)),
    }
    write_json(output_path, payload)


def save_class_balance_plot(bundle, output_path: Path) -> None:
    """Plot the class counts for the imbalanced dataset."""

    sns.set_theme(style="whitegrid")
    frame = pd.DataFrame(
        {
            "Class": ["Negative (0)", "Positive (1)"],
            "Count": [bundle.class_counts[0], bundle.class_counts[1]],
        }
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=frame,
        x="Class",
        y="Count",
        hue="Class",
        dodge=False,
        legend=False,
        ax=axis,
        palette=["#4C72B0", "#DD8452"],
    )
    axis.set_title("Credit Card Fraud Class Distribution")
    axis.bar_label(axis.containers[0], fmt="%d")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_cv_summary(fold_results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Aggregate fold-level metrics into mean/std summary rows."""

    metrics = [
        "precision",
        "recall",
        "f1_macro",
        "f1_micro",
        "roc_auc",
        "pr_auc",
        "mcc",
        "balanced_accuracy",
    ]
    aggregated = (
        fold_results.groupby(
            ["candidate_key", "candidate_label", "model_key", "model_label", "strategy_key", "strategy_label"],
            as_index=False,
        )[metrics]
        .agg(["mean", "std"])
    )
    aggregated.columns = ["_".join(part).strip("_") for part in aggregated.columns.to_flat_index()]
    aggregated = aggregated.sort_values(
        ["pr_auc_mean", "mcc_mean", "recall_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    aggregated.to_csv(output_path, index=False)
    return aggregated


def plot_precision_recall_curve(y_true, probabilities, output_path: Path, title: str) -> None:
    """Save a precision-recall curve."""

    figure, axis = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axis, name=title)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curve(y_true, probabilities, output_path: Path, title: str) -> None:
    """Save an ROC curve."""

    figure, axis = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axis, name=title)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_reliability_diagram(y_true, probabilities, output_path: Path, title: str, n_bins: int) -> None:
    """Save a calibration curve / reliability diagram."""

    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy="uniform")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    axis.plot(prob_pred, prob_true, marker="o", label=title, color="#4C72B0")
    axis.set_xlabel("Predicted Probability")
    axis.set_ylabel("Observed Frequency")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    """Save the confusion matrix figure."""

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axis, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_summary_markdown(
    output_path: Path,
    dataset_overview: dict,
    cv_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    final_metrics: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> None:
    """Write a concise narrative summary for the run directory."""

    best_cv = cv_summary.iloc[0]
    best_calibrated = calibration_summary.sort_values(
        ["selected_for_final_model", "brier_score"],
        ascending=[False, True],
    ).iloc[0]

    lines = [
        "# Question 2 Summary",
        "",
        "## Dataset",
        f"- Training rows: {dataset_overview['train_rows']}",
        f"- Test rows: {dataset_overview['test_rows']}",
        f"- Positive rate: {dataset_overview['positive_rate']:.6f}",
        f"- Imbalance ratio (negative:positive): {dataset_overview['imbalance_ratio']:.2f}",
        "",
        "## Cross-Validation Ranking",
        f"- Best candidate: {best_cv['candidate_label']} with PR-AUC {best_cv['pr_auc_mean']:.4f} ± {best_cv['pr_auc_std']:.4f}",
        f"- Mean MCC: {best_cv['mcc_mean']:.4f}",
        f"- Mean recall: {best_cv['recall_mean']:.4f}",
        "",
        "## Calibration",
        f"- Best calibrated variant selected for final evaluation: {best_calibrated['candidate_label']} using {best_calibrated['calibration_method']}",
        f"- Calibration-set Brier score: {best_calibrated['brier_score']:.6f}",
        "",
        "## Final Holdout Metrics",
    ]

    for _, row in final_metrics.iterrows():
        lines.append(
            f"- {row['candidate_label']} [{row['threshold_policy']}]: Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, PR-AUC={row['pr_auc']:.4f}, MCC={row['mcc']:.4f}, "
            f"Balanced Accuracy={row['balanced_accuracy']:.4f}, Brier={row['brier_score']:.6f}"
        )

    lines.extend(["", "## Threshold Comparison"])
    for _, row in threshold_summary.iterrows():
        lines.append(
            f"- {row['threshold_policy']}: threshold={row['threshold']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, false_negatives={int(row['false_negatives'])}, false_positives={int(row['false_positives'])}"
        )

    write_text(output_path, "\n".join(lines))


def render_q2_latex_section(
    output_path: Path,
    run_dir: Path,
    dataset_overview: dict,
    cv_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    final_metrics: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> None:
    """Render the real Question 2 LaTeX section from run artifacts."""

    best_cv = cv_summary.iloc[0]
    second_cv = cv_summary.iloc[1]
    best_final = final_metrics.sort_values(["pr_auc", "recall"], ascending=[False, False]).iloc[0]
    best_cal = calibration_summary.sort_values(["selected_for_final_model", "brier_score"], ascending=[False, True]).iloc[0]
    cost_row = threshold_summary.loc[threshold_summary["threshold_policy"] == "recall_first_cost_sensitive"].iloc[0]
    f1_row = threshold_summary.loc[threshold_summary["threshold_policy"] == "f1_max"].iloc[0]

    relative_root = Path("..") / run_dir.relative_to(run_dir.parents[2])
    pr_curve_path = relative_root / "figures" / "best_model_precision_recall_curve.png"
    roc_curve_path = relative_root / "figures" / "best_model_roc_curve.png"
    reliability_path = relative_root / "figures" / "best_model_reliability.png"
    confusion_path = relative_root / "figures" / "best_model_confusion_matrix.png"

    latex = f"""
\\section{{Question 2: Classification Under Extreme Imbalance}}

\\subsection{{Problem Setup}}

For Question 2, I used the credit-card fraud dataset stored at \\texttt{{data/external/creditcard.csv}}. The task is binary fraud detection with target column \\texttt{{Class}}, where the positive class corresponds to fraudulent transactions. The dataset is extremely imbalanced, with a positive rate of {dataset_overview['positive_rate']:.6f} and an imbalance ratio of approximately {dataset_overview['imbalance_ratio']:.2f}:1 in favor of the negative class. I kept a single stratified 20\\% holdout split for final evaluation and restricted all model selection, resampling, calibration, and threshold tuning to the training portion only.

Leakage prevention was handled explicitly. The final test set was never touched during model comparison. For each cross-validation fold, resampling was performed only inside an \\texttt{{imblearn}} pipeline after preprocessing, so synthetic or undersampled observations were generated from training-fold data only. I scaled \\texttt{{Time}} and \\texttt{{Amount}} while leaving the PCA-style \\texttt{{V1}}--\\texttt{{V28}} features as passthrough numeric inputs.

\\subsection{{Models and Sampling Strategies}}

I compared four classifier families: Logistic Regression, Random Forest, XGBoost, and a feed-forward MLP classifier. Each model was evaluated under five imbalance-handling strategies: no resampling, SMOTE, ADASYN, random undersampling, and cost-sensitive learning. Cost-sensitive learning used \\texttt{{class\\_weight="balanced"}} for Logistic Regression and Random Forest, \\texttt{{scale\\_pos\\_weight}} for XGBoost, and balanced sample weights for the MLP.

Model ranking used 5-fold stratified cross-validation on the training split. Candidates were ordered primarily by PR-AUC, then MCC, then recall, since PR-AUC is more informative than ROC-AUC under severe imbalance and the downstream objective is recall-first fraud detection. The top two uncalibrated candidates were {best_cv['candidate_label']} and {second_cv['candidate_label']}.

\\begin{{table}}[htbp]
  \\centering
  \\caption{{Top cross-validation results for Question 2, ordered by PR-AUC.}}
  \\label{{tab:q2-cv}}
  \\begin{{tabular}}{{lrrrrr}}
    \\toprule
    Candidate & PR-AUC & MCC & Recall & Precision & Balanced Acc. \\\\
    \\midrule
    {best_cv['candidate_label']} & {best_cv['pr_auc_mean']:.4f} $\\pm$ {best_cv['pr_auc_std']:.4f} & {best_cv['mcc_mean']:.4f} & {best_cv['recall_mean']:.4f} & {best_cv['precision_mean']:.4f} & {best_cv['balanced_accuracy_mean']:.4f} \\\\
    {second_cv['candidate_label']} & {second_cv['pr_auc_mean']:.4f} $\\pm$ {second_cv['pr_auc_std']:.4f} & {second_cv['mcc_mean']:.4f} & {second_cv['recall_mean']:.4f} & {second_cv['precision_mean']:.4f} & {second_cv['balanced_accuracy_mean']:.4f} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\subsection{{Evaluation and Calibration}}

All candidate models were evaluated using precision, recall, macro F1, micro F1, ROC-AUC, PR-AUC, MCC, and balanced accuracy. After cross-validation, I recalibrated the best two candidates using both Platt scaling (sigmoid) and isotonic regression on a calibration-only validation slice drawn from the training partition. The better calibration method for each candidate was selected by Brier score.

The best calibrated variant overall was {best_cal['candidate_label']} with {best_cal['calibration_method']} calibration, achieving a calibration-set Brier score of {best_cal['brier_score']:.6f}. Threshold selection was then performed on the calibration split using two rules: the F1-max threshold and a recall-first cost-sensitive threshold with false-negative cost 5 and false-positive cost 1. Because missing fraud is more costly than over-flagging a legitimate transaction, I used the cost-sensitive threshold as the default operating point.

\\begin{{table}}[htbp]
  \\centering
  \\caption{{Threshold comparison for the selected final model on the holdout split.}}
  \\label{{tab:q2-thresholds}}
  \\begin{{tabular}}{{lrrrrr}}
    \\toprule
    Policy & Threshold & Precision & Recall & False Negatives & False Positives \\\\
    \\midrule
    Recall-first cost-sensitive & {cost_row['threshold']:.4f} & {cost_row['precision']:.4f} & {cost_row['recall']:.4f} & {int(cost_row['false_negatives'])} & {int(cost_row['false_positives'])} \\\\
    F1-max & {f1_row['threshold']:.4f} & {f1_row['precision']:.4f} & {f1_row['recall']:.4f} & {int(f1_row['false_negatives'])} & {int(f1_row['false_positives'])} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

The final selected model was {best_final['candidate_label']}, evaluated on the untouched holdout set. Under the default recall-first threshold, it achieved precision {best_final['precision']:.4f}, recall {best_final['recall']:.4f}, PR-AUC {best_final['pr_auc']:.4f}, MCC {best_final['mcc']:.4f}, balanced accuracy {best_final['balanced_accuracy']:.4f}, and Brier score {best_final['brier_score']:.6f}. Figures~\\ref{{fig:q2-pr-roc}} and \\ref{{fig:q2-calibration-confusion}} visualize the final decision behavior.

\\begin{{figure}}[htbp]
  \\centering
  \\begin{{minipage}}[t]{{0.48\\textwidth}}
    \\centering
    \\includegraphics[width=\\textwidth]{{{pr_curve_path.as_posix()}}}
  \\end{{minipage}}
  \\hfill
  \\begin{{minipage}}[t]{{0.48\\textwidth}}
    \\centering
    \\includegraphics[width=\\textwidth]{{{roc_curve_path.as_posix()}}}
  \\end{{minipage}}
  \\caption{{Precision-recall and ROC curves for the final calibrated Question 2 model on the holdout set.}}
  \\label{{fig:q2-pr-roc}}
\\end{{figure}}

\\begin{{figure}}[htbp]
  \\centering
  \\begin{{minipage}}[t]{{0.48\\textwidth}}
    \\centering
    \\includegraphics[width=\\textwidth]{{{reliability_path.as_posix()}}}
  \\end{{minipage}}
  \\hfill
  \\begin{{minipage}}[t]{{0.48\\textwidth}}
    \\centering
    \\includegraphics[width=\\textwidth]{{{confusion_path.as_posix()}}}
  \\end{{minipage}}
  \\caption{{Reliability diagram and confusion matrix for the final calibrated Question 2 model at the recall-first operating threshold.}}
  \\label{{fig:q2-calibration-confusion}}
\\end{{figure}}

\\subsection{{Discussion}}

The results show why PR-AUC and recall-aware thresholding matter for extreme imbalance. A model can look acceptable under ROC-AUC while still failing to identify enough fraud cases at a usable operating threshold. The final pipeline therefore emphasized PR-AUC during model ranking and then used a cost-sensitive threshold to explicitly favor recall. This choice reflects the application setting: missing a fraudulent transaction is typically more costly than reviewing an extra flagged transaction.

Calibration was also necessary because raw classifier scores are not always well aligned with true fraud probabilities. By comparing sigmoid and isotonic calibration on a dedicated validation slice, the final workflow separates ranking quality from probability quality. This improves both the reliability diagram and the downstream threshold decision. In practice, the most meaningful trade-off is between recall and operational alert volume, so the final report should interpret false positives as review workload and false negatives as direct fraud risk.
""".strip()

    write_text(output_path, latex)
