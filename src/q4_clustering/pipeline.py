"""End-to-end clustering workflow for Question 4."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.common.config import load_config
from src.common.io import ensure_dir, project_path, write_json
from src.common.run import bootstrap_run
from src.common.seed import set_global_seed


os.environ.setdefault("MPLCONFIGDIR", "/tmp/ceng463-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ceng463-cache")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SUMMARY = (
    "End-to-end clustering workflow with K-Means, Gaussian Mixture Models, "
    "DBSCAN, Agglomerative Clustering, bootstrap stability, and a co-association ensemble."
)

CHECKLIST = [
    "Load the configured clustering dataset and standardise feature scales.",
    "Select K-Means k with elbow, silhouette, and gap-statistic diagnostics.",
    "Select GMM components with BIC/AIC and tune DBSCAN with a k-distance graph.",
    "Fit K-Means, GMM, DBSCAN, Agglomerative, and ensemble clusterers.",
    "Report internal metrics, optional external validation, and bootstrap stability.",
    "Visualise PCA projections and discuss algorithm assumptions.",
]

EXPECTED_ARTIFACTS = [
    "model-selection plots",
    "internal metrics table",
    "external validation table when labels exist",
    "stability analysis table",
    "cluster ensemble comparison",
    "cluster projection figures",
]


@dataclass(frozen=True)
class DataBundle:
    """Scaled clustering data plus optional labels."""

    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    dataset_name: str
    has_labels: bool


@dataclass(frozen=True)
class ClusteringResult:
    """Final labels and metadata for one clustering run."""

    key: str
    label: str
    labels: np.ndarray
    params: dict[str, Any]
    runtime_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the full Question 4 clustering workflow.")
    parser.add_argument("--config", default="configs/q4_clustering.yaml", help="Path to YAML config.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified sample size for faster runs.")
    parser.add_argument("--bootstrap-rounds", type=int, default=None, help="Override bootstrap stability rounds.")
    parser.add_argument("--gap-reference-datasets", type=int, default=None, help="Override gap-statistic reference count.")
    parser.add_argument("--smoke", action="store_true", help="Run a small fast validation pass.")
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply runtime overrides without mutating the loaded config."""

    resolved = deepcopy(config)
    if args.sample_size is not None:
        resolved.setdefault("dataset", {})["sample_size"] = args.sample_size
    if args.bootstrap_rounds is not None:
        resolved.setdefault("selection", {})["bootstrap_rounds"] = args.bootstrap_rounds
    if args.gap_reference_datasets is not None:
        resolved.setdefault("selection", {})["gap_reference_datasets"] = args.gap_reference_datasets

    if args.smoke:
        resolved["experiment_name"] = "q4_smoke"
        resolved.setdefault("dataset", {})["sample_size"] = 300
        resolved.setdefault("selection", {})["k_range"] = [2, 6]
        resolved.setdefault("selection", {})["bootstrap_rounds"] = 3
        resolved.setdefault("selection", {})["gap_reference_datasets"] = 3
        resolved.setdefault("dbscan", {})["eps_quantiles"] = [0.70, 0.80, 0.90]

    return resolved


def load_dataset(config: dict[str, Any]) -> DataBundle:
    """Load optdigits-style data by default, or a configured CSV if it exists."""

    dataset_cfg = config.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "optdigits")).lower()
    csv_path = dataset_cfg.get("path")
    label_column = dataset_cfg.get("label_column")

    if csv_path and project_path(str(csv_path)).exists():
        frame = pd.read_csv(project_path(str(csv_path)))
        y = frame[label_column].to_numpy() if label_column and label_column in frame.columns else None
        X_frame = frame.drop(columns=[label_column]) if label_column and label_column in frame.columns else frame
        numeric_frame = X_frame.select_dtypes(include=[np.number])
        X = numeric_frame.to_numpy(dtype=np.float64)
        feature_names = list(numeric_frame.columns)
        name = dataset_name
    else:
        digits = load_digits()
        X = digits.data.astype(np.float64)
        y = digits.target.astype(np.int64)
        feature_names = [f"pixel_{index}" for index in range(X.shape[1])]
        name = "optdigits_sklearn_digits"

    X = StandardScaler().fit_transform(X)
    y = None if y is None else np.asarray(y)
    X, y = maybe_sample(X, y, int(dataset_cfg.get("sample_size", 0)), int(config.get("seed", 463)))
    return DataBundle(X=X, y=y, feature_names=feature_names, dataset_name=name, has_labels=y is not None)


def maybe_sample(
    X: np.ndarray,
    y: np.ndarray | None,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Optionally sample rows, stratifying when labels are available."""

    if sample_size <= 0 or sample_size >= len(X):
        return X, y

    rng = np.random.default_rng(seed)
    if y is None:
        indices = rng.choice(np.arange(len(X)), size=sample_size, replace=False)
    else:
        per_class = max(1, sample_size // len(np.unique(y)))
        sampled = []
        for class_id in np.unique(y):
            class_indices = np.flatnonzero(y == class_id)
            sampled.append(rng.choice(class_indices, size=min(per_class, len(class_indices)), replace=False))
        indices = np.concatenate(sampled)
        if len(indices) < sample_size:
            remaining = np.setdiff1d(np.arange(len(X)), indices, assume_unique=False)
            indices = np.concatenate([indices, rng.choice(remaining, size=sample_size - len(indices), replace=False)])
        rng.shuffle(indices)

    return X[indices], None if y is None else y[indices]


def parse_k_range(config: dict[str, Any]) -> list[int]:
    """Expand the inclusive k range from config."""

    raw_range = config.get("selection", {}).get("k_range", [2, 10])
    return list(range(int(raw_range[0]), int(raw_range[1]) + 1))


def valid_cluster_count(labels: np.ndarray) -> int:
    """Count clusters, excluding DBSCAN noise label -1."""

    clusters = set(np.unique(labels))
    clusters.discard(-1)
    return len(clusters)


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    """Compute internal clustering metrics when at least two clusters exist."""

    n_clusters = valid_cluster_count(labels)
    noise_fraction = float(np.mean(labels == -1))
    if n_clusters < 2 or n_clusters >= len(X):
        return {
            "n_clusters": n_clusters,
            "noise_fraction": noise_fraction,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
    return {
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def external_metrics(y_true: np.ndarray | None, labels: np.ndarray) -> dict[str, float]:
    """Compute optional external validation metrics."""

    if y_true is None:
        return {}
    return {
        "adjusted_rand": float(adjusted_rand_score(y_true, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, labels)),
        "fowlkes_mallows": float(fowlkes_mallows_score(y_true, labels)),
    }


def compute_gap_statistic(
    X: np.ndarray,
    k_values: list[int],
    seed: int,
    reference_datasets: int,
) -> pd.DataFrame:
    """Compute a compact K-Means gap statistic table."""

    rng = np.random.default_rng(seed)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
        model.fit(X)
        observed_log_inertia = np.log(max(model.inertia_, 1e-12))
        reference_logs = []
        for _ in range(reference_datasets):
            reference = rng.uniform(mins, maxs, size=X.shape)
            reference_model = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 1_000_000)))
            reference_model.fit(reference)
            reference_logs.append(np.log(max(reference_model.inertia_, 1e-12)))
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "gap": float(np.mean(reference_logs) - observed_log_inertia),
                "gap_std": float(np.std(reference_logs, ddof=1)) if len(reference_logs) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def select_kmeans(X: np.ndarray, config: dict[str, Any]) -> tuple[int, pd.DataFrame]:
    """Evaluate K-Means over k and select by silhouette with gap as context."""

    seed = int(config.get("seed", 463))
    k_values = parse_k_range(config)
    reference_datasets = int(config.get("selection", {}).get("gap_reference_datasets", 10))
    gap_frame = compute_gap_statistic(X, k_values, seed, reference_datasets)
    rows = []
    for k in k_values:
        start = time.perf_counter()
        labels = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(X)
        metrics = internal_metrics(X, labels)
        rows.append({"k": k, "runtime_seconds": time.perf_counter() - start, **metrics})
    selection = pd.DataFrame(rows).merge(gap_frame, on="k", how="left")
    best = selection.sort_values(["silhouette", "gap"], ascending=[False, False]).iloc[0]
    return int(best["k"]), selection


def select_gmm(X: np.ndarray, config: dict[str, Any]) -> tuple[int, pd.DataFrame]:
    """Evaluate GMM over component counts and select by lowest BIC."""

    seed = int(config.get("seed", 463))
    rows = []
    for k in parse_k_range(config):
        start = time.perf_counter()
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5)
        labels = model.fit_predict(X)
        rows.append(
            {
                "n_components": k,
                "bic": float(model.bic(X)),
                "aic": float(model.aic(X)),
                "runtime_seconds": time.perf_counter() - start,
                **internal_metrics(X, labels),
            }
        )
    selection = pd.DataFrame(rows)
    best = selection.sort_values("bic", ascending=True).iloc[0]
    return int(best["n_components"]), selection


def select_dbscan(X: np.ndarray, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Tune DBSCAN eps from k-distance quantiles."""

    dbscan_cfg = config.get("dbscan", {})
    min_samples = int(dbscan_cfg.get("min_samples", 5))
    quantiles = [float(value) for value in dbscan_cfg.get("eps_quantiles", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])]
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    eps_values = sorted(set(float(np.quantile(k_distances, q)) for q in quantiles))
    distance_frame = pd.DataFrame({"rank": np.arange(1, len(k_distances) + 1), "k_distance": k_distances})

    rows = []
    for eps in eps_values:
        start = time.perf_counter()
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        rows.append(
            {
                "eps": eps,
                "min_samples": min_samples,
                "runtime_seconds": time.perf_counter() - start,
                **internal_metrics(X, labels),
            }
        )

    selection = pd.DataFrame(rows)
    valid = selection[selection["n_clusters"] >= 2].copy()
    if valid.empty:
        best = selection.sort_values("noise_fraction", ascending=True).iloc[0]
    else:
        best = valid.sort_values(["silhouette", "noise_fraction"], ascending=[False, True]).iloc[0]
    return {"eps": float(best["eps"]), "min_samples": min_samples}, selection, distance_frame


def select_agglomerative(X: np.ndarray, config: dict[str, Any]) -> tuple[int, pd.DataFrame]:
    """Evaluate Ward agglomerative clustering over k and select by silhouette."""

    rows = []
    for k in parse_k_range(config):
        start = time.perf_counter()
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        rows.append({"k": k, "runtime_seconds": time.perf_counter() - start, **internal_metrics(X, labels)})
    selection = pd.DataFrame(rows)
    best = selection.sort_values("silhouette", ascending=False).iloc[0]
    return int(best["k"]), selection


def fit_final_clusterers(
    X: np.ndarray,
    config: dict[str, Any],
    selections: dict[str, Any],
) -> tuple[list[ClusteringResult], dict[str, Any]]:
    """Fit all selected clustering methods on the full working dataset."""

    seed = int(config.get("seed", 463))
    results = []

    start = time.perf_counter()
    kmeans = KMeans(n_clusters=selections["kmeans_k"], n_init=50, random_state=seed)
    labels = kmeans.fit_predict(X)
    results.append(
        ClusteringResult(
            key="kmeans",
            label=f"K-Means k={selections['kmeans_k']}",
            labels=labels,
            params={"k": selections["kmeans_k"]},
            runtime_seconds=time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    gmm = GaussianMixture(n_components=selections["gmm_components"], covariance_type="full", random_state=seed, n_init=10)
    labels = gmm.fit_predict(X)
    results.append(
        ClusteringResult(
            key="gmm",
            label=f"GMM components={selections['gmm_components']}",
            labels=labels,
            params={"n_components": selections["gmm_components"]},
            runtime_seconds=time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    dbscan_params = selections["dbscan_params"]
    labels = DBSCAN(**dbscan_params).fit_predict(X)
    results.append(
        ClusteringResult(
            key="dbscan",
            label=f"DBSCAN eps={dbscan_params['eps']:.3f}",
            labels=labels,
            params=dbscan_params,
            runtime_seconds=time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    labels = AgglomerativeClustering(n_clusters=selections["agglomerative_k"], linkage="ward").fit_predict(X)
    results.append(
        ClusteringResult(
            key="agglomerative",
            label=f"Agglomerative k={selections['agglomerative_k']}",
            labels=labels,
            params={"k": selections["agglomerative_k"], "linkage": "ward"},
            runtime_seconds=time.perf_counter() - start,
        )
    )

    fitted_models = {"kmeans": kmeans, "gmm": gmm}
    return results, fitted_models


def coassociation_ensemble(
    X: np.ndarray,
    base_results: list[ClusteringResult],
    config: dict[str, Any],
) -> ClusteringResult:
    """Build a simple co-association matrix ensemble and recluster it."""

    start = time.perf_counter()
    labels_list = [result.labels for result in base_results if result.key in {"kmeans", "gmm", "dbscan"}]
    member_labels = ["kmeans", "gmm", "dbscan"]
    kmeans_result = next((result for result in base_results if result.key == "kmeans"), None)
    if kmeans_result is not None:
        selected_k = int(kmeans_result.params["k"])
        raw_range = config.get("selection", {}).get("k_range", [2, 10])
        min_k = int(raw_range[0])
        max_k = int(raw_range[1])
        variant_ks = sorted({max(min_k, selected_k - 1), selected_k, min(max_k, selected_k + 1)})
        seed = int(config.get("seed", 463))
        for variant_k in variant_ks:
            if variant_k == selected_k:
                continue
            variant = KMeans(n_clusters=variant_k, n_init=20, random_state=seed + variant_k)
            labels_list.append(variant.fit_predict(X))
            member_labels.append(f"kmeans_k_{variant_k}")

    n_samples = len(X)
    coassociation = np.zeros((n_samples, n_samples), dtype=np.float64)
    for labels in labels_list:
        same_cluster = labels[:, None] == labels[None, :]
        valid = (labels[:, None] != -1) & (labels[None, :] != -1)
        coassociation += same_cluster & valid
    coassociation /= max(len(labels_list), 1)
    np.fill_diagonal(coassociation, 1.0)

    distance = 1.0 - coassociation
    condensed = squareform(distance, checks=False)
    n_clusters = int(config.get("ensemble", {}).get("n_clusters", 0)) or infer_ensemble_k(base_results)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    labels = model.fit_predict(distance)
    return ClusteringResult(
        key="ensemble",
        label=f"Co-association ensemble k={n_clusters}",
        labels=labels,
        params={"n_clusters": n_clusters, "members": ";".join(member_labels), "condensed_distance_size": len(condensed)},
        runtime_seconds=time.perf_counter() - start,
    )


def infer_ensemble_k(results: list[ClusteringResult]) -> int:
    """Prefer the internally selected K-Means k for the ensemble cluster count."""

    kmeans_result = next((result for result in results if result.key == "kmeans"), None)
    if kmeans_result is not None:
        return int(kmeans_result.params["k"])
    counts = [valid_cluster_count(result.labels) for result in results if result.key in {"kmeans", "gmm", "dbscan"}]
    return max(2, int(np.median([count for count in counts if count > 1] or [2])))


def result_metrics(data: DataBundle, results: list[ClusteringResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build internal and external metric tables."""

    internal_rows = []
    external_rows = []
    for result in results:
        internal_rows.append(
            {
                "algorithm": result.key,
                "label": result.label,
                "runtime_seconds": result.runtime_seconds,
                **result.params,
                **internal_metrics(data.X, result.labels),
            }
        )
        if data.has_labels:
            external_rows.append({"algorithm": result.key, "label": result.label, **external_metrics(data.y, result.labels)})

    return pd.DataFrame(internal_rows), pd.DataFrame(external_rows)


def stability_analysis(
    X: np.ndarray,
    config: dict[str, Any],
    selections: dict[str, Any],
    include_ensemble: bool = True,
) -> pd.DataFrame:
    """Estimate clustering stability with 80% bootstrap/subsample ARI."""

    selection_cfg = config.get("selection", {})
    rounds = int(selection_cfg.get("bootstrap_rounds", 20))
    fraction = float(selection_cfg.get("bootstrap_fraction", 0.8))
    seed = int(config.get("seed", 463))
    rng = np.random.default_rng(seed + 31)
    n_subsample = max(10, int(len(X) * fraction))
    full_results, _ = fit_final_clusterers(X, config, selections)
    if include_ensemble:
        full_results.append(coassociation_ensemble(X, full_results, config))
    full_lookup = {result.key: result.labels for result in full_results}
    rows = []

    for round_id in range(1, rounds + 1):
        indices = np.sort(rng.choice(np.arange(len(X)), size=n_subsample, replace=False))
        X_sub = X[indices]
        sub_results, _ = fit_final_clusterers(X_sub, config, selections)
        if include_ensemble:
            sub_results.append(coassociation_ensemble(X_sub, sub_results, config))
        for result in sub_results:
            rows.append(
                {
                    "round_id": round_id,
                    "algorithm": result.key,
                    "adjusted_rand_to_full": float(adjusted_rand_score(full_lookup[result.key][indices], result.labels)),
                }
            )

    frame = pd.DataFrame(rows)
    return (
        frame.groupby("algorithm")["adjusted_rand_to_full"]
        .agg(stability_ari_mean="mean", stability_ari_std="std")
        .reset_index()
    )


def save_selection_plots(
    kmeans_selection: pd.DataFrame,
    gmm_selection: pd.DataFrame,
    dbscan_selection: pd.DataFrame,
    k_distance: pd.DataFrame,
    agglomerative_selection: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Save model-selection diagnostic plots."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(kmeans_selection["k"], kmeans_selection["inertia"], marker="o")
    axes[0].set_title("K-Means Elbow")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(kmeans_selection["k"], kmeans_selection["silhouette"], marker="o", label="silhouette")
    axes[1].plot(kmeans_selection["k"], kmeans_selection["gap"], marker="s", label="gap")
    axes[1].set_title("K-Means Silhouette and Gap")
    axes[1].set_xlabel("k")
    axes[1].legend()
    axes[2].plot(gmm_selection["n_components"], gmm_selection["bic"], marker="o", label="BIC")
    axes[2].plot(gmm_selection["n_components"], gmm_selection["aic"], marker="s", label="AIC")
    axes[2].set_title("GMM Selection")
    axes[2].set_xlabel("components")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "model_selection_kmeans_gmm.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(k_distance["rank"], k_distance["k_distance"])
    axes[0].set_title("DBSCAN k-distance Graph")
    axes[0].set_xlabel("Sorted point rank")
    axes[0].set_ylabel("k-distance")
    axes[1].plot(dbscan_selection["eps"], dbscan_selection["silhouette"], marker="o")
    axes[1].set_title("DBSCAN eps vs silhouette")
    axes[1].set_xlabel("eps")
    axes[2].plot(agglomerative_selection["k"], agglomerative_selection["silhouette"], marker="o")
    axes[2].set_title("Agglomerative k Selection")
    axes[2].set_xlabel("k")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_selection_dbscan_agglomerative.png", dpi=200)
    plt.close(fig)


def save_dendrogram(X: np.ndarray, figures_dir: Path, config: dict[str, Any]) -> None:
    """Save a truncated Ward dendrogram."""

    max_points = min(int(config.get("visualisation", {}).get("dendrogram_sample_size", 300)), len(X))
    rng = np.random.default_rng(int(config.get("seed", 463)) + 57)
    indices = rng.choice(np.arange(len(X)), size=max_points, replace=False)
    linked = linkage(X[indices], method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Ward Linkage Dendrogram (truncated)")
    ax.set_xlabel("Merged clusters")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(figures_dir / "agglomerative_dendrogram.png", dpi=200)
    plt.close(fig)


def save_projection_plots(data: DataBundle, results: list[ClusteringResult], figures_dir: Path) -> None:
    """Save PCA projection coloured by true labels and cluster assignments."""

    projection = PCA(n_components=2, random_state=463).fit_transform(data.X)
    if data.has_labels:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(projection[:, 0], projection[:, 1], c=data.y, cmap="tab10", s=10, alpha=0.75, linewidths=0)
        ax.set_title("PCA projection coloured by digit label")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / "pca_projection_ground_truth.png", dpi=200)
        plt.close(fig)

    for result in results:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(projection[:, 0], projection[:, 1], c=result.labels, cmap="tab20", s=10, alpha=0.75, linewidths=0)
        ax.set_title(result.label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / f"pca_projection_{result.key}.png", dpi=200)
        plt.close(fig)


def write_summary(
    output_path: Path,
    data: DataBundle,
    selections: dict[str, Any],
    internal: pd.DataFrame,
    external: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """Write a compact Markdown summary for report drafting."""

    best_internal = internal.sort_values("silhouette", ascending=False).iloc[0]
    lines = [
        "# Question 4 Run Summary",
        "",
        f"- Dataset: {data.dataset_name}; samples={len(data.X)}, features={data.X.shape[1]}, labels={data.has_labels}.",
        f"- Selected K-Means k: {selections['kmeans_k']}.",
        f"- Selected GMM components: {selections['gmm_components']}.",
        f"- Selected DBSCAN params: {selections['dbscan_params']}.",
        f"- Selected Agglomerative k: {selections['agglomerative_k']}.",
        f"- Best silhouette: {best_internal['label']} ({best_internal['silhouette']:.4f}).",
        "",
        "## Internal Metrics",
        "```text",
        internal.to_string(index=False),
        "```",
    ]
    if not external.empty:
        best_external = external.sort_values("adjusted_rand", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                f"- Best external ARI: {best_external['label']} ({best_external['adjusted_rand']:.4f}).",
                "",
                "## External Metrics",
                "```text",
                external.to_string(index=False),
                "```",
            ]
        )
    lines.extend(["", "## Stability", "```text", stability.to_string(index=False), "```"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full Question 4 workflow."""

    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    set_global_seed(int(config.get("seed", 463)))
    warnings.filterwarnings("ignore", category=UserWarning)

    run_dir = bootstrap_run("q4_clustering", config, SUMMARY, CHECKLIST, EXPECTED_ARTIFACTS)
    figures_dir = ensure_dir(run_dir / "figures")
    tables_dir = ensure_dir(run_dir / "tables")
    model_dir = ensure_dir(run_dir / "models")

    data = load_dataset(config)
    print(f"Loaded dataset: {data.dataset_name}, X={data.X.shape}, labels={data.has_labels}")

    kmeans_k, kmeans_selection = select_kmeans(data.X, config)
    gmm_components, gmm_selection = select_gmm(data.X, config)
    dbscan_params, dbscan_selection, k_distance = select_dbscan(data.X, config)
    agglomerative_k, agglomerative_selection = select_agglomerative(data.X, config)
    selections = {
        "kmeans_k": kmeans_k,
        "gmm_components": gmm_components,
        "dbscan_params": dbscan_params,
        "agglomerative_k": agglomerative_k,
    }
    print("Finished model selection.")

    results, fitted_models = fit_final_clusterers(data.X, config, selections)
    if bool(config.get("ensemble", {}).get("enabled", True)):
        results.append(coassociation_ensemble(data.X, results, config))
    print("Finished final clustering and ensemble.")

    internal, external = result_metrics(data, results)
    stability = stability_analysis(data.X, config, selections, include_ensemble=bool(config.get("ensemble", {}).get("enabled", True)))
    print("Finished stability analysis.")

    kmeans_selection.to_csv(tables_dir / "kmeans_selection.csv", index=False)
    gmm_selection.to_csv(tables_dir / "gmm_selection.csv", index=False)
    dbscan_selection.to_csv(tables_dir / "dbscan_selection.csv", index=False)
    k_distance.to_csv(tables_dir / "dbscan_k_distance.csv", index=False)
    agglomerative_selection.to_csv(tables_dir / "agglomerative_selection.csv", index=False)
    internal.to_csv(tables_dir / "internal_metrics.csv", index=False)
    external.to_csv(tables_dir / "external_metrics.csv", index=False)
    stability.to_csv(tables_dir / "stability_analysis.csv", index=False)
    write_json(tables_dir / "selected_hyperparameters.json", selections)

    for key, model in fitted_models.items():
        joblib.dump(model, model_dir / f"{key}.joblib")
    np.savez_compressed(model_dir / "cluster_labels.npz", **{result.key: result.labels for result in results})

    save_selection_plots(kmeans_selection, gmm_selection, dbscan_selection, k_distance, agglomerative_selection, figures_dir)
    save_dendrogram(data.X, figures_dir, config)
    save_projection_plots(data, results, figures_dir)
    write_summary(run_dir / "summary.md", data, selections, internal, external, stability)

    print(f"Saved Question 4 artifacts at {run_dir}")
    print(f"Selected K-Means k={kmeans_k}, GMM components={gmm_components}, DBSCAN={dbscan_params}, Agglomerative k={agglomerative_k}")


if __name__ == "__main__":
    main()
