"""End-to-end dimensionality reduction workflow for Question 3."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    "End-to-end dimensionality reduction workflow comparing PCA, Kernel PCA, "
    "t-SNE, UMAP, and an undercomplete autoencoder on Fashion-MNIST."
)

CHECKLIST = [
    "Load a balanced Fashion-MNIST subset and flatten images into scaled pixel vectors.",
    "Train PCA, Kernel PCA, t-SNE, UMAP, and an undercomplete autoencoder.",
    "Record reconstruction MSE for PCA, Kernel PCA, and the autoencoder.",
    "Compute trustworthiness, continuity, and Kruskal stress for all t-SNE/UMAP runs.",
    "Train a k-NN classifier on 2D embeddings and compare 5-fold accuracy.",
    "Visualise 2D latent spaces and record runtime/memory-oriented discussion notes.",
]

EXPECTED_ARTIFACTS = [
    "2D embedding figures",
    "reconstruction MSE table",
    "trustworthiness, continuity, and Kruskal stress table",
    "k-NN downstream accuracy table",
    "runtime summary table",
    "selected hyperparameters JSON",
]

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@dataclass(frozen=True)
class DataBundle:
    """Balanced train/test data for dimensionality reduction experiments."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding plus enough metadata to evaluate and report one method/config."""

    key: str
    label: str
    method: str
    train_embedding: np.ndarray
    test_embedding: np.ndarray | None
    params: dict[str, Any]
    runtime_seconds: float


class UndercompleteAutoencoder(nn.Module):
    """Simple fully connected autoencoder with a 2D or 3D bottleneck."""

    def __init__(self, input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, bottleneck_dim))

        decoder_layers: list[nn.Module] = []
        previous_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim
        decoder_layers.extend([nn.Linear(previous_dim, input_dim), nn.Sigmoid()])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the full Question 3 dimensionality reduction workflow.")
    parser.add_argument("--config", default="configs/q3_dimensionality_reduction.yaml", help="Path to YAML config.")
    parser.add_argument("--train-per-class", type=int, default=None, help="Override balanced train samples per class.")
    parser.add_argument("--test-per-class", type=int, default=None, help="Override balanced test samples per class.")
    parser.add_argument("--epochs", type=int, default=None, help="Override autoencoder training epochs.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny subset and single t-SNE/UMAP settings to validate the pipeline quickly.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply small runtime overrides without mutating the loaded config in-place."""

    resolved = deepcopy(config)
    dataset_cfg = resolved.setdefault("dataset", {})
    autoencoder_cfg = resolved.setdefault("autoencoder", {})

    if args.train_per_class is not None:
        dataset_cfg["train_per_class"] = args.train_per_class
    if args.test_per_class is not None:
        dataset_cfg["test_per_class"] = args.test_per_class
    if args.epochs is not None:
        autoencoder_cfg["epochs"] = args.epochs

    if args.smoke:
        resolved["experiment_name"] = "q3_smoke"
        dataset_cfg["train_per_class"] = 20
        dataset_cfg["test_per_class"] = 10
        resolved.setdefault("tsne", {})["perplexities"] = [5]
        resolved.setdefault("umap", {})["n_neighbors"] = [10]
        resolved.setdefault("umap", {})["min_dist"] = [0.1]
        resolved.setdefault("evaluation", {})["manifold_sample_size"] = 200
        autoencoder_cfg["epochs"] = min(int(autoencoder_cfg.get("epochs", 20)), 2)

    return resolved


def load_fashion_mnist(config: dict[str, Any]) -> DataBundle:
    """Load Fashion-MNIST and return balanced train/test subsets scaled to [0, 1]."""

    try:
        from torchvision.datasets import FashionMNIST, MNIST
    except Exception as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError("torchvision is required for Q3 dataset loading.") from exc

    dataset_cfg = config.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "fashion_mnist")).lower()
    dataset_cls = MNIST if dataset_name == "mnist" else FashionMNIST
    data_dir = project_path(str(dataset_cfg.get("data_dir", "data/raw")))
    download = bool(dataset_cfg.get("download", True))

    train_dataset = dataset_cls(root=data_dir, train=True, download=download)
    test_dataset = dataset_cls(root=data_dir, train=False, download=download)

    X_train_all = train_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train_all = train_dataset.targets.numpy().astype(np.int64)
    X_test_all = test_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test_all = test_dataset.targets.numpy().astype(np.int64)

    seed = int(config.get("seed", 463))
    X_train, y_train = balanced_subset(
        X_train_all,
        y_train_all,
        int(dataset_cfg.get("train_per_class", 500)),
        seed,
    )
    X_test, y_test = balanced_subset(
        X_test_all,
        y_test_all,
        int(dataset_cfg.get("test_per_class", 100)),
        seed + 1,
    )
    return DataBundle(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the same number of examples from each class."""

    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []
    for class_id in np.unique(y):
        class_indices = np.flatnonzero(y == class_id)
        if samples_per_class > len(class_indices):
            raise ValueError(f"Requested {samples_per_class} samples for class {class_id}, only {len(class_indices)} exist.")
        selected_indices.append(rng.choice(class_indices, size=samples_per_class, replace=False))

    indices = np.concatenate(selected_indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def run_pca(data: DataBundle, config: dict[str, Any]) -> tuple[EmbeddingResult, dict[str, float], Any]:
    """Fit linear PCA and compute reconstruction MSE."""

    n_components = int(config.get("reduction", {}).get("n_components", 2))
    start = time.perf_counter()
    model = PCA(n_components=n_components, random_state=int(config.get("seed", 463)))
    train_embedding = model.fit_transform(data.X_train)
    test_embedding = model.transform(data.X_test)
    reconstructed = model.inverse_transform(test_embedding)
    runtime_seconds = time.perf_counter() - start
    result = EmbeddingResult(
        key="pca",
        label="PCA",
        method="pca",
        train_embedding=train_embedding,
        test_embedding=test_embedding,
        params={"n_components": n_components},
        runtime_seconds=runtime_seconds,
    )
    return result, {"method": "PCA", "reconstruction_mse": mse(data.X_test, reconstructed)}, model


def run_kernel_pca(data: DataBundle, config: dict[str, Any]) -> tuple[EmbeddingResult, dict[str, float], Any]:
    """Fit RBF Kernel PCA with inverse transform enabled."""

    n_components = int(config.get("reduction", {}).get("n_components", 2))
    kernel_cfg = config.get("kernel_pca", {})
    gamma = float(kernel_cfg.get("gamma", 1.0 / data.X_train.shape[1]))
    start = time.perf_counter()
    model = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
        random_state=int(config.get("seed", 463)),
        n_jobs=int(config.get("n_jobs", -1)),
    )
    train_embedding = model.fit_transform(data.X_train)
    test_embedding = model.transform(data.X_test)
    reconstructed = model.inverse_transform(test_embedding)
    runtime_seconds = time.perf_counter() - start
    result = EmbeddingResult(
        key="kernel_pca",
        label="Kernel PCA",
        method="kernel_pca",
        train_embedding=train_embedding,
        test_embedding=test_embedding,
        params={"n_components": n_components, "kernel": "rbf", "gamma": gamma},
        runtime_seconds=runtime_seconds,
    )
    return result, {"method": "Kernel PCA", "reconstruction_mse": mse(data.X_test, reconstructed)}, model


def run_tsne_grid(data: DataBundle, config: dict[str, Any]) -> list[EmbeddingResult]:
    """Run t-SNE for every configured perplexity."""

    n_components = int(config.get("reduction", {}).get("n_components", 2))
    perplexities = config.get("tsne", {}).get("perplexities", [5, 30, 50])
    results = []
    for perplexity in perplexities:
        start = time.perf_counter()
        model = TSNE(
            n_components=n_components,
            perplexity=float(perplexity),
            init="pca",
            learning_rate="auto",
            random_state=int(config.get("seed", 463)),
        )
        embedding = model.fit_transform(data.X_train)
        results.append(
            EmbeddingResult(
                key=f"tsne_perplexity_{perplexity}",
                label=f"t-SNE perplexity={perplexity}",
                method="tsne",
                train_embedding=embedding,
                test_embedding=None,
                params={"n_components": n_components, "perplexity": float(perplexity)},
                runtime_seconds=time.perf_counter() - start,
            )
        )
    return results


def run_umap_grid(data: DataBundle, config: dict[str, Any]) -> list[EmbeddingResult]:
    """Run UMAP for every configured neighborhood/min_dist pair."""

    try:
        import umap
    except Exception as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError("umap-learn is required for Q3. Install requirements.txt before running Q3.") from exc

    n_components = int(config.get("reduction", {}).get("n_components", 2))
    umap_cfg = config.get("umap", {})
    n_neighbors_grid = umap_cfg.get("n_neighbors", [10, 15, 30])
    min_dist_grid = umap_cfg.get("min_dist", [0.0, 0.1, 0.5])
    results = []
    for n_neighbors in n_neighbors_grid:
        for min_dist in min_dist_grid:
            start = time.perf_counter()
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                metric="euclidean",
                random_state=int(config.get("seed", 463)),
                n_jobs=1,
            )
            embedding = model.fit_transform(data.X_train)
            results.append(
                EmbeddingResult(
                    key=f"umap_neighbors_{n_neighbors}_min_dist_{min_dist}",
                    label=f"UMAP n={n_neighbors}, min_dist={min_dist}",
                    method="umap",
                    train_embedding=embedding,
                    test_embedding=None,
                    params={
                        "n_components": n_components,
                        "n_neighbors": int(n_neighbors),
                        "min_dist": float(min_dist),
                    },
                    runtime_seconds=time.perf_counter() - start,
                )
            )
    return results


def run_autoencoder(
    data: DataBundle,
    config: dict[str, Any],
    model_dir: Path,
    logs_dir: Path,
) -> tuple[EmbeddingResult, dict[str, float], UndercompleteAutoencoder]:
    """Train an undercomplete autoencoder and return latent embeddings."""

    autoencoder_cfg = config.get("autoencoder", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UndercompleteAutoencoder(
        input_dim=data.X_train.shape[1],
        hidden_dims=[int(value) for value in autoencoder_cfg.get("hidden_dims", [256, 128, 64])],
        bottleneck_dim=int(autoencoder_cfg.get("bottleneck_dim", 2)),
    ).to(device)

    train_tensor = torch.tensor(data.X_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=int(autoencoder_cfg.get("batch_size", 256)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(autoencoder_cfg.get("learning_rate", 0.001)))
    loss_fn = nn.MSELoss()
    history = []
    start = time.perf_counter()

    model.train()
    for epoch in range(1, int(autoencoder_cfg.get("epochs", 20)) + 1):
        epoch_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_mse": float(np.mean(epoch_losses))})

    runtime_seconds = time.perf_counter() - start
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(logs_dir / "autoencoder_training_history.csv", index=False)
    plot_training_history(history_frame, logs_dir / "autoencoder_training_history.png")

    model.eval()
    with torch.no_grad():
        train_embedding = model.encode(torch.tensor(data.X_train, dtype=torch.float32, device=device)).cpu().numpy()
        test_tensor = torch.tensor(data.X_test, dtype=torch.float32, device=device)
        test_embedding = model.encode(test_tensor).cpu().numpy()
        reconstructed = model(test_tensor).cpu().numpy()

    torch.save(model.state_dict(), model_dir / "autoencoder.pt")
    result = EmbeddingResult(
        key="autoencoder",
        label="Undercomplete Autoencoder",
        method="autoencoder",
        train_embedding=train_embedding,
        test_embedding=test_embedding,
        params={
            "hidden_dims": autoencoder_cfg.get("hidden_dims", [256, 128, 64]),
            "bottleneck_dim": int(autoencoder_cfg.get("bottleneck_dim", 2)),
            "epochs": int(autoencoder_cfg.get("epochs", 20)),
        },
        runtime_seconds=runtime_seconds,
    )
    return result, {"method": "Autoencoder", "reconstruction_mse": mse(data.X_test, reconstructed)}, model


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return mean squared reconstruction error."""

    return float(np.mean((actual - np.clip(predicted, 0.0, 1.0)) ** 2))


def metric_sample_indices(y: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Choose a stable subset for expensive manifold-quality metrics."""

    max_size = int(config.get("evaluation", {}).get("manifold_sample_size", len(y)))
    if max_size >= len(y):
        return np.arange(len(y))
    per_class = max(1, max_size // len(np.unique(y)))
    selected = []
    rng = np.random.default_rng(int(config.get("seed", 463)) + 17)
    for class_id in np.unique(y):
        class_indices = np.flatnonzero(y == class_id)
        selected.extend(rng.choice(class_indices, size=per_class, replace=False).tolist())
    selected_array = np.array(selected, dtype=int)
    rng.shuffle(selected_array)
    return selected_array


def rank_matrix(distances: np.ndarray) -> np.ndarray:
    """Return matrix where entry [i, j] is j's rank by distance from i."""

    order = np.argsort(distances, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(distances.shape[0])[:, None]
    ranks[rows, order] = np.arange(distances.shape[0], dtype=np.int32)
    return ranks


def continuity_score(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int) -> float:
    """Compute continuity, the trustworthiness analogue from high to low space."""

    high_distances = pairwise_distances(X_high)
    low_distances = pairwise_distances(X_low)
    high_order = np.argsort(high_distances, axis=1)[:, 1 : n_neighbors + 1]
    low_order = np.argsort(low_distances, axis=1)[:, 1 : n_neighbors + 1]
    low_ranks = rank_matrix(low_distances)

    penalty = 0.0
    for row_index in range(X_high.shape[0]):
        low_neighbors = set(low_order[row_index])
        missing_from_low = [idx for idx in high_order[row_index] if idx not in low_neighbors]
        penalty += sum(float(low_ranks[row_index, idx] - n_neighbors) for idx in missing_from_low)

    n_samples = X_high.shape[0]
    denominator = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
    return float(1.0 - (2.0 / denominator) * penalty)


def kruskal_stress(X_high: np.ndarray, X_low: np.ndarray) -> float:
    """Compute scale-normalised Kruskal stress between high and low distances."""

    high_distances = pairwise_distances(X_high)
    low_distances = pairwise_distances(X_low)
    row_idx, col_idx = np.triu_indices_from(high_distances, k=1)
    high = high_distances[row_idx, col_idx]
    low = low_distances[row_idx, col_idx]
    high = high / max(float(np.linalg.norm(high)), 1e-12)
    low = low / max(float(np.linalg.norm(low)), 1e-12)
    return float(np.sqrt(np.sum((high - low) ** 2) / max(np.sum(high**2), 1e-12)))


def compute_manifold_metrics(
    data: DataBundle,
    manifold_results: list[EmbeddingResult],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute trustworthiness, continuity, and stress for t-SNE/UMAP runs."""

    indices = metric_sample_indices(data.y_train, config)
    X_metric = data.X_train[indices]
    n_neighbors = int(config.get("evaluation", {}).get("metric_neighbors", 10))
    rows = []
    for result in manifold_results:
        embedding_metric = result.train_embedding[indices]
        rows.append(
            {
                "embedding_key": result.key,
                "method": result.method,
                "label": result.label,
                "metric_sample_size": int(len(indices)),
                "trustworthiness": float(trustworthiness(X_metric, embedding_metric, n_neighbors=n_neighbors)),
                "continuity": continuity_score(X_metric, embedding_metric, n_neighbors=n_neighbors),
                "kruskal_stress": kruskal_stress(X_metric, embedding_metric),
                **result.params,
            }
        )
    return pd.DataFrame(rows)


def select_best_manifold(manifold_metrics: pd.DataFrame, method: str) -> str:
    """Select best t-SNE/UMAP config by trustworthiness, then lower stress."""

    subset = manifold_metrics[manifold_metrics["method"] == method].copy()
    best = subset.sort_values(["trustworthiness", "kruskal_stress"], ascending=[False, True]).iloc[0]
    return str(best["embedding_key"])


def evaluate_knn_embeddings(results: list[EmbeddingResult], y: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    """Run 5-fold k-NN cross-validation on each 2D embedding."""

    classifier_cfg = config.get("downstream_classifier", {})
    splitter = StratifiedKFold(
        n_splits=int(config.get("evaluation", {}).get("cv_folds", 5)),
        shuffle=True,
        random_state=int(config.get("seed", 463)),
    )
    rows = []
    for result in results:
        estimator = KNeighborsClassifier(n_neighbors=int(classifier_cfg.get("n_neighbors", 5)))
        scores = cross_val_score(estimator, result.train_embedding, y, cv=splitter, scoring="accuracy")
        rows.append(
            {
                "embedding_key": result.key,
                "method": result.method,
                "label": result.label,
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores)),
                "fold_scores": ";".join(f"{score:.6f}" for score in scores),
                **result.params,
            }
        )
    return pd.DataFrame(rows)


def runtime_summary(results: list[EmbeddingResult]) -> pd.DataFrame:
    """Create one runtime row per method/config."""

    return pd.DataFrame(
        [
            {
                "embedding_key": result.key,
                "method": result.method,
                "label": result.label,
                "runtime_seconds": result.runtime_seconds,
                **result.params,
            }
            for result in results
        ]
    )


def plot_embedding(result: EmbeddingResult, y: np.ndarray, output_path: Path) -> None:
    """Save a 2D class-coloured embedding scatter plot."""

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        result.train_embedding[:, 0],
        result.train_embedding[:, 1],
        c=y,
        cmap="tab10",
        s=7,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_title(result.label)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)
    handles, _ = scatter.legend_elements(num=10)
    ax.legend(handles, CLASS_NAMES, title="Class", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    """Save autoencoder loss curve."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_mse"], marker="o", linewidth=1.5)
    ax.set_title("Autoencoder Training Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    data: DataBundle,
    selected_keys: dict[str, str],
    reconstruction: pd.DataFrame,
    manifold_metrics: pd.DataFrame,
    knn_metrics: pd.DataFrame,
    runtimes: pd.DataFrame,
) -> None:
    """Write concise run notes for the LaTeX report."""

    best_knn = knn_metrics.sort_values("mean_accuracy", ascending=False).iloc[0]
    lines = [
        "# Question 3 Run Summary",
        "",
        f"- Train subset: {len(data.X_train)} balanced examples ({len(np.unique(data.y_train))} classes).",
        f"- Test subset: {len(data.X_test)} balanced examples.",
        f"- Selected t-SNE: `{selected_keys['tsne']}`.",
        f"- Selected UMAP: `{selected_keys['umap']}`.",
        f"- Best downstream k-NN embedding: {best_knn['label']} with accuracy {best_knn['mean_accuracy']:.4f}.",
        "",
        "## Reconstruction MSE",
        "```text",
        reconstruction.to_string(index=False),
        "```",
        "",
        "## Selected Manifold Metrics",
        "```text",
        manifold_metrics[manifold_metrics["embedding_key"].isin(selected_keys.values())].to_string(index=False),
        "```",
        "",
        "## Runtime Summary",
        "```text",
        runtimes.sort_values("runtime_seconds", ascending=False).to_string(index=False),
        "```",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full Question 3 workflow."""

    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    set_global_seed(int(config.get("seed", 463)))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")

    run_dir = bootstrap_run("q3_dimensionality_reduction", config, SUMMARY, CHECKLIST, EXPECTED_ARTIFACTS)
    figures_dir = ensure_dir(run_dir / "figures")
    tables_dir = ensure_dir(run_dir / "tables")
    logs_dir = ensure_dir(run_dir / "logs")
    model_dir = ensure_dir(run_dir / "models")

    data = load_fashion_mnist(config)
    print(f"Loaded balanced dataset: train={data.X_train.shape}, test={data.X_test.shape}")

    embedding_results: list[EmbeddingResult] = []
    reconstruction_rows = []

    pca_result, pca_reconstruction, pca_model = run_pca(data, config)
    embedding_results.append(pca_result)
    reconstruction_rows.append(pca_reconstruction)
    joblib.dump(pca_model, model_dir / "pca.joblib")
    print("Finished PCA.")

    kernel_pca_result, kernel_pca_reconstruction, kernel_pca_model = run_kernel_pca(data, config)
    embedding_results.append(kernel_pca_result)
    reconstruction_rows.append(kernel_pca_reconstruction)
    joblib.dump(kernel_pca_model, model_dir / "kernel_pca.joblib")
    print("Finished Kernel PCA.")

    tsne_results = run_tsne_grid(data, config)
    embedding_results.extend(tsne_results)
    print("Finished t-SNE grid.")

    umap_results = run_umap_grid(data, config)
    embedding_results.extend(umap_results)
    print("Finished UMAP grid.")

    autoencoder_result, autoencoder_reconstruction, _ = run_autoencoder(data, config, model_dir, logs_dir)
    embedding_results.append(autoencoder_result)
    reconstruction_rows.append(autoencoder_reconstruction)
    print("Finished autoencoder.")

    manifold_results = tsne_results + umap_results
    manifold_metrics = compute_manifold_metrics(data, manifold_results, config)
    selected_keys = {
        "tsne": select_best_manifold(manifold_metrics, "tsne"),
        "umap": select_best_manifold(manifold_metrics, "umap"),
    }
    selected_result_keys = {"pca", "kernel_pca", selected_keys["tsne"], selected_keys["umap"], "autoencoder"}
    selected_embedding_results = [result for result in embedding_results if result.key in selected_result_keys]

    reconstruction = pd.DataFrame(reconstruction_rows)
    knn_metrics = evaluate_knn_embeddings(embedding_results, data.y_train, config)
    runtimes = runtime_summary(embedding_results)

    reconstruction.to_csv(tables_dir / "reconstruction_mse.csv", index=False)
    manifold_metrics.to_csv(tables_dir / "manifold_metrics.csv", index=False)
    knn_metrics.to_csv(tables_dir / "knn_cv_accuracy.csv", index=False)
    runtimes.to_csv(tables_dir / "runtime_summary.csv", index=False)
    write_json(tables_dir / "selected_hyperparameters.json", selected_keys)

    for result in selected_embedding_results:
        plot_embedding(result, data.y_train, figures_dir / f"{result.key}_embedding.png")

    write_summary_markdown(
        run_dir / "summary.md",
        data,
        selected_keys,
        reconstruction,
        manifold_metrics,
        knn_metrics,
        runtimes,
    )

    print(f"Saved Question 3 artifacts at {run_dir}")
    print(f"Selected t-SNE: {selected_keys['tsne']}")
    print(f"Selected UMAP: {selected_keys['umap']}")


if __name__ == "__main__":
    main()
