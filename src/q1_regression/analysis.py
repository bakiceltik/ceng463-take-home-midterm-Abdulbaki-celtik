"""EDA, diagnostics, and reporting helpers for Question 1."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

TEMP_CACHE_ROOT = Path(tempfile.gettempdir()) / "ceng463-matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import het_breuschpagan

from src.common.io import write_json, write_text


def save_dataset_overview(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_log_target: bool,
    output_path: Path,
) -> None:
    """Persist dataset-level metadata for the run."""

    payload = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "target_mean_train": float(y_train.mean()),
        "target_std_train": float(y_train.std()),
        "target_skew_train": float(stats.skew(y_train.to_numpy())),
        "target_mean_test": float(y_test.mean()),
        "target_std_test": float(y_test.std()),
        "log_target_transform": bool(use_log_target),
    }
    write_json(output_path, payload)


def save_feature_distributions(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot histograms for all numeric fields including the target."""

    sns.set_theme(style="whitegrid")
    columns = list(frame.columns)
    figure, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for axis, column in zip(axes, columns, strict=False):
        sns.histplot(frame[column], kde=True, ax=axis, color="#2E86AB")
        axis.set_title(column)

    for axis in axes[len(columns) :]:
        axis.axis("off")

    figure.suptitle("Question 1 Feature Distributions", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_correlation_heatmap(frame: pd.DataFrame, output_path: Path) -> pd.Series:
    """Plot the correlation heatmap and return target correlations."""

    correlation = frame.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, cmap="coolwarm", center=0.0, ax=axis)
    axis.set_title("Question 1 Correlation Heatmap")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return correlation["MedHouseVal"].drop(labels=["MedHouseVal"]).sort_values(key=np.abs, ascending=False)


def save_pairwise_interactions(
    frame: pd.DataFrame,
    ranked_target_correlations: pd.Series,
    output_path: Path,
    sample_size: int,
) -> None:
    """Save a sampled pairplot for the strongest target interactions."""

    top_features = list(ranked_target_correlations.head(4).index)
    plot_frame = frame[top_features + ["MedHouseVal"]].sample(
        n=min(sample_size, len(frame)),
        random_state=463,
    )
    pair_grid = sns.pairplot(plot_frame, corner=True, diag_kind="hist")
    pair_grid.figure.suptitle("Top Pairwise Interactions", y=1.02)
    pair_grid.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(pair_grid.figure)


def save_outlier_summary(frame: pd.DataFrame, output_path: Path, zscore_threshold: float) -> pd.DataFrame:
    """Compute IQR and Z-score outlier counts per feature."""

    rows: list[dict[str, float | str]] = []
    numeric_frame = frame.select_dtypes(include=["number"])

    for column in numeric_frame.columns:
        series = numeric_frame[column]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (series < lower) | (series > upper)
        z_scores = np.abs(stats.zscore(series.to_numpy(), nan_policy="omit"))
        z_mask = z_scores > zscore_threshold

        rows.append(
            {
                "feature": column,
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "iqr_outlier_count": int(iqr_mask.sum()),
                "iqr_outlier_pct": float(iqr_mask.mean()),
                "zscore_threshold": zscore_threshold,
                "zscore_outlier_count": int(np.nansum(z_mask)),
                "zscore_outlier_pct": float(np.nanmean(z_mask)),
            }
        )

    outlier_df = pd.DataFrame(rows).sort_values("iqr_outlier_count", ascending=False)
    outlier_df.to_csv(output_path, index=False)
    return outlier_df


def save_feature_space_summary(
    X_train: pd.DataFrame,
    polynomial_degrees: Iterable[int],
    output_path: Path,
) -> pd.DataFrame:
    """Summarize raw and polynomial feature-space sizes."""

    rows = [
        {
            "feature_space": "raw",
            "degree": 1,
            "feature_count": int(X_train.shape[1]),
        }
    ]

    for degree in polynomial_degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(X_train)
        rows.append(
            {
                "feature_space": f"polynomial_degree_{degree}",
                "degree": int(degree),
                "feature_count": int(poly.n_output_features_),
            }
        )

    feature_space_df = pd.DataFrame(rows)
    feature_space_df.to_csv(output_path, index=False)
    return feature_space_df


def save_rfe_feature_ranking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    degree: int,
    top_k: int,
    output_path: Path,
) -> pd.DataFrame:
    """Run RFE on polynomial features and save the resulting ranking."""

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    feature_names = poly.get_feature_names_out(X_train.columns)

    selector = RFE(
        estimator=Ridge(alpha=1.0),
        n_features_to_select=min(top_k, len(feature_names)),
        step=0.1,
    )
    selector.fit(X_poly, y_train)

    ranking_df = pd.DataFrame(
        {
            "feature": feature_names,
            "ranking": selector.ranking_,
            "selected": selector.support_,
        }
    ).sort_values(["ranking", "feature"], ascending=[True, True])

    ranking_df.to_csv(output_path, index=False)
    return ranking_df


def save_cv_metric_summary(fold_results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Aggregate per-fold metrics into report-ready mean/std rows."""

    metrics = ["rmse", "mae", "r2", "adjusted_r2", "mape", "explained_variance"]
    aggregated = fold_results.groupby(["model_key", "model_label"], as_index=False)[metrics].agg(["mean", "std"])
    aggregated.columns = ["_".join(part).strip("_") for part in aggregated.columns.to_flat_index()]
    aggregated = aggregated.rename(
        columns={
            "model_key": "model_key",
            "model_label": "model_label",
        }
    )
    aggregated = aggregated.sort_values(["rmse_mean", "mae_mean", "r2_mean"], ascending=[True, True, False])
    aggregated.to_csv(output_path, index=False)
    return aggregated


def save_pairwise_significance_tests(fold_results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Run paired t-tests and Wilcoxon tests on fold-level RMSE."""

    pivot = fold_results.pivot_table(index="fold_id", columns="model_label", values="rmse")
    labels = list(pivot.columns)
    rows: list[dict[str, float | str]] = []

    for idx, left in enumerate(labels):
        for right in labels[idx + 1 :]:
            paired = pivot[[left, right]].dropna()
            left_scores = paired[left].to_numpy()
            right_scores = paired[right].to_numpy()
            differences = left_scores - right_scores

            t_stat, t_pvalue = stats.ttest_rel(left_scores, right_scores)
            try:
                w_stat, w_pvalue = stats.wilcoxon(left_scores, right_scores, zero_method="wilcox")
            except ValueError:
                w_stat, w_pvalue = np.nan, np.nan

            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "model_a_mean_rmse": float(left_scores.mean()),
                    "model_b_mean_rmse": float(right_scores.mean()),
                    "winner_lower_rmse": left if left_scores.mean() < right_scores.mean() else right,
                    "paired_t_statistic": float(t_stat),
                    "paired_t_pvalue": float(t_pvalue),
                    "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else np.nan,
                    "wilcoxon_pvalue": float(w_pvalue) if not np.isnan(w_pvalue) else np.nan,
                    "mean_rmse_difference_a_minus_b": float(differences.mean()),
                }
            )

    tests_df = pd.DataFrame(rows).sort_values("paired_t_pvalue")
    tests_df.to_csv(output_path, index=False)
    return tests_df


def compute_residual_diagnostics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate quantitative residual diagnostic tests."""

    residuals = y_true.to_numpy() - y_pred
    fitted = np.asarray(y_pred)
    exog = sm.add_constant(fitted)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, exog)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)

    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "breusch_pagan_statistic": float(bp_stat),
        "breusch_pagan_pvalue": float(bp_pvalue),
        "jarque_bera_statistic": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
    }


def save_residual_plots(y_true: pd.Series, y_pred: np.ndarray, model_label: str, output_dir: Path) -> None:
    """Save fitted-vs-residual and Q-Q plots for a model."""

    residuals = y_true.to_numpy() - y_pred
    fitted = np.asarray(y_pred)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(fitted, residuals, alpha=0.45, s=20, color="#FF6B35")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Fitted Values")
    axis.set_ylabel("Residuals")
    axis.set_title(f"{model_label}: Fitted vs Residuals")
    figure.tight_layout()
    figure.savefig(output_dir / f"{model_label.lower().replace(' ', '_')}_residuals.png", dpi=200)
    plt.close(figure)

    qq_figure, qq_axis = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=qq_axis)
    qq_axis.set_title(f"{model_label}: Q-Q Plot")
    qq_figure.tight_layout()
    qq_figure.savefig(output_dir / f"{model_label.lower().replace(' ', '_')}_qq.png", dpi=200)
    plt.close(qq_figure)


def write_summary_markdown(
    output_path: Path,
    dataset_overview: dict,
    cv_summary: pd.DataFrame,
    final_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    significance: pd.DataFrame,
    robust_note: str,
) -> None:
    """Write a concise narrative summary for the run directory."""

    best_row = cv_summary.iloc[0]
    summary_lines = [
        "# Question 1 Summary",
        "",
        "## Dataset",
        f"- Training rows: {dataset_overview['train_rows']}",
        f"- Test rows: {dataset_overview['test_rows']}",
        f"- Feature count: {dataset_overview['feature_count']}",
        f"- Target skew on train split: {dataset_overview['target_skew_train']:.4f}",
        f"- Log target transform enabled: {dataset_overview['log_target_transform']}",
        "",
        "## Cross-Validation Ranking",
        f"- Best model by mean RMSE: {best_row['model_label']} ({best_row['rmse_mean']:.4f} ± {best_row['rmse_std']:.4f})",
        f"- Best model mean MAE: {best_row['mae_mean']:.4f}",
        f"- Best model mean R2: {best_row['r2_mean']:.4f}",
        "",
        "## Final Holdout Metrics",
    ]

    for _, row in final_metrics.iterrows():
        summary_lines.append(
            f"- {row['model_label']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}, "
            f"Adjusted R2={row['adjusted_r2']:.4f}, MAPE={row['mape']:.4f}, Explained Variance={row['explained_variance']:.4f}"
        )

    summary_lines.extend(
        [
            "",
            "## Residual Diagnostics",
        ]
    )

    for _, row in diagnostics.iterrows():
        summary_lines.append(
            f"- {row['model_label']}: Breusch-Pagan p={row['breusch_pagan_pvalue']:.4f}, "
            f"Jarque-Bera p={row['jarque_bera_pvalue']:.4f}"
        )

    summary_lines.extend(
        [
            "",
            "## Significance Testing",
            f"- Pairwise comparisons computed for {len(significance)} model pairs using fold-level RMSE.",
            robust_note,
        ]
    )
    write_text(output_path, "\n".join(summary_lines))
