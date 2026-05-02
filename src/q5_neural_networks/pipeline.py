"""End-to-end CIFAR-10 neural-network workflow for Question 5."""

from __future__ import annotations

import argparse
import itertools
import os
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ceng463-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ceng463-cache")

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from src.common.config import load_config
from src.common.io import ensure_dir, project_path, write_json
from src.common.run import bootstrap_run
from src.common.seed import set_global_seed


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SUMMARY = (
    "End-to-end CIFAR-10 workflow comparing a deep MLP, a CNN trained from scratch, "
    "and a ResNet18 transfer-learning baseline with interpretability and FGSM robustness."
)

CHECKLIST = [
    "Load CIFAR-10 with balanced train/validation/test subsets.",
    "Train a deep MLP with four 512-unit hidden layers, batch norm, dropout, and early stopping.",
    "Train a CNN with at least three convolutional layers, pooling, dropout, batch norm, and augmentation.",
    "Fine-tune only the last ResNet18 layers as a transfer-learning baseline.",
    "Run a compact learning-rate/dropout/weight-decay/batch-size search.",
    "Save metrics, curves, confusion matrices, misclassification grids, Grad-CAM/LIME, and FGSM results.",
]

EXPECTED_ARTIFACTS = [
    "hyperparameter search table",
    "final metrics table",
    "per-class recall table",
    "confusion matrices",
    "training curves",
    "misclassified examples",
    "Grad-CAM and LIME figures",
    "FGSM robustness table",
]

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    builder: Callable[[dict[str, Any]], nn.Module]
    family: str


@dataclass
class TrainResult:
    model_key: str
    label: str
    model: nn.Module
    params: dict[str, Any]
    history: pd.DataFrame
    best_val_accuracy: float
    runtime_seconds: float


class IndexedDataset(Dataset):
    """Dataset wrapper returning transformed tensor, label, and original dataset index."""

    def __init__(self, dataset: Dataset, indices: list[int], transform: Callable) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        original_index = self.indices[item]
        image, label = self.dataset[original_index]
        return self.transform(image), int(label), int(original_index)


class DeepMLP(nn.Module):
    """Deep MLP with four hidden 512-unit blocks."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 3 * 32 * 32
        for _ in range(4):
            layers.extend(
                [
                    nn.Linear(in_features, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = 512
        layers.append(nn.Linear(512, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.flatten(x, start_dim=1))


class SmallCnn(nn.Module):
    """CNN trained from scratch with 4 convolutional layers."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the full Question 5 CIFAR-10 workflow.")
    parser.add_argument("--config", default="configs/q5_neural_networks.yaml", help="Path to YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end validation pass.")
    parser.add_argument("--epochs", type=int, default=None, help="Override final-training epochs.")
    parser.add_argument("--search-trials", type=int, default=None, help="Override compact search trials per model.")
    parser.add_argument("--train-per-class", type=int, default=None, help="Override train samples per class.")
    parser.add_argument("--val-per-class", type=int, default=None, help="Override validation samples per class.")
    parser.add_argument("--test-per-class", type=int, default=None, help="Override test samples per class.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use random ResNet18 weights if pretrained weights are unavailable or not desired.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply runtime overrides without mutating loaded config."""

    resolved = deepcopy(config)
    dataset_cfg = resolved.setdefault("dataset", {})
    training_cfg = resolved.setdefault("training", {})
    search_cfg = resolved.setdefault("hyperparameter_optimisation", {})
    transfer_cfg = resolved.setdefault("transfer_learning", {})

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.search_trials is not None:
        search_cfg["max_trials"] = args.search_trials
    if args.train_per_class is not None:
        dataset_cfg["train_per_class"] = args.train_per_class
    if args.val_per_class is not None:
        dataset_cfg["val_per_class"] = args.val_per_class
    if args.test_per_class is not None:
        dataset_cfg["test_per_class"] = args.test_per_class
    if args.no_pretrained:
        transfer_cfg["pretrained"] = False

    if args.smoke:
        resolved["experiment_name"] = "q5_smoke"
        dataset_cfg["train_per_class"] = 12
        dataset_cfg["val_per_class"] = 6
        dataset_cfg["test_per_class"] = 6
        training_cfg["epochs"] = 1
        training_cfg["early_stopping_patience"] = 1
        search_cfg["max_trials"] = 1
        search_cfg["search_epochs"] = 1
        resolved.setdefault("interpretability", {})["lime_samples"] = 20
        resolved.setdefault("interpretability", {})["examples_per_model"] = 3
        resolved.setdefault("robustness", {})["max_batches"] = 2

    return resolved


def balanced_indices(targets: list[int], per_class: int, seed: int) -> list[int]:
    """Sample a balanced list of indices from targets."""

    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    selected: list[int] = []
    for class_id in np.unique(targets_array):
        class_indices = np.flatnonzero(targets_array == class_id)
        selected.extend(rng.choice(class_indices, size=min(per_class, len(class_indices)), replace=False).tolist())
    rng.shuffle(selected)
    return selected


def build_splits(train_dataset: datasets.CIFAR10, test_dataset: datasets.CIFAR10, config: dict[str, Any]) -> SplitIndices:
    """Build balanced train/validation/test splits from CIFAR-10."""

    dataset_cfg = config.get("dataset", {})
    seed = int(config.get("seed", 463))
    train_per_class = int(dataset_cfg.get("train_per_class", 250))
    val_per_class = int(dataset_cfg.get("val_per_class", 50))
    test_per_class = int(dataset_cfg.get("test_per_class", 50))
    rng = np.random.default_rng(seed)
    targets = np.asarray(train_dataset.targets)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        chosen = rng.choice(class_indices, size=train_per_class + val_per_class, replace=False)
        train_indices.extend(chosen[:train_per_class].tolist())
        val_indices.extend(chosen[train_per_class:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    test_indices = balanced_indices(test_dataset.targets, test_per_class, seed + 1)
    return SplitIndices(train=train_indices, val=val_indices, test=test_indices)


def transforms_for(family: str, train: bool) -> transforms.Compose:
    """Return transforms for each model family."""

    if family == "resnet":
        size = 96
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        base: list[Any] = [transforms.Resize((size, size))]
    else:
        mean, std = CIFAR_MEAN, CIFAR_STD
        base = []
    if train and family in {"cnn", "resnet"}:
        base.extend(
            [
                transforms.RandomCrop(32, padding=4) if family == "cnn" else transforms.RandomResizedCrop(96, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    base.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(base)


def make_loaders(
    family: str,
    batch_size: int,
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    splits: SplitIndices,
    num_workers: int,
) -> dict[str, DataLoader]:
    """Create train/validation/test loaders for one model family."""

    return {
        "train": DataLoader(
            IndexedDataset(train_dataset, splits.train, transforms_for(family, train=True)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            IndexedDataset(train_dataset, splits.val, transforms_for(family, train=False)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            IndexedDataset(test_dataset, splits.test, transforms_for(family, train=False)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def load_cifar10(config: dict[str, Any]) -> tuple[datasets.CIFAR10, datasets.CIFAR10, SplitIndices]:
    """Load CIFAR-10 raw PIL datasets and balanced split indices."""

    dataset_cfg = config.get("dataset", {})
    data_dir = project_path(str(dataset_cfg.get("data_dir", "data/raw")))
    download = bool(dataset_cfg.get("download", True))
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=None)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=None)
    splits = build_splits(train_dataset, test_dataset, config)
    return train_dataset, test_dataset, splits


def select_device() -> torch.device:
    """Select the best available PyTorch device for this workflow."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet18(config: dict[str, Any]) -> nn.Module:
    """Build ResNet18 and fine-tune only late layers."""

    transfer_cfg = config.get("transfer_learning", {})
    use_pretrained = bool(transfer_cfg.get("pretrained", True))
    weights = None
    if use_pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except AttributeError:  # pragma: no cover
            weights = "DEFAULT"
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def model_specs(config: dict[str, Any]) -> list[ModelSpec]:
    """Return model specs in the required comparison order."""

    return [
        ModelSpec("mlp", "Deep MLP", lambda params: DeepMLP(dropout=float(params["dropout"])), "mlp"),
        ModelSpec("cnn", "CNN from scratch", lambda params: SmallCnn(dropout=float(params["dropout"])), "cnn"),
        ModelSpec("resnet18", "ResNet18 transfer", lambda params: build_resnet18(config), "resnet"),
    ]


def search_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build compact manual grid candidates."""

    search_cfg = config.get("hyperparameter_optimisation", {})
    values = {
        "learning_rate": search_cfg.get("learning_rates", [0.001]),
        "batch_size": search_cfg.get("batch_sizes", [128]),
        "dropout": search_cfg.get("dropout_rates", [0.3]),
        "weight_decay": search_cfg.get("weight_decays", [0.0001]),
    }
    candidates = [
        dict(zip(values.keys(), combo))
        for combo in itertools.product(
            values["learning_rate"],
            values["batch_size"],
            values["dropout"],
            values["weight_decay"],
        )
    ]
    return candidates[: int(search_cfg.get("max_trials", 2))]


def train_one_model(
    spec: ModelSpec,
    params: dict[str, Any],
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    splits: SplitIndices,
    config: dict[str, Any],
    epochs: int,
    device: torch.device,
) -> TrainResult:
    """Train one model with early stopping."""

    loaders = make_loaders(
        spec.family,
        int(params["batch_size"]),
        train_dataset,
        test_dataset,
        splits,
        int(config.get("training", {}).get("num_workers", 0)),
    )
    model = spec.builder(params).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()
    patience = int(config.get("training", {}).get("early_stopping_patience", 2))
    best_state = deepcopy(model.state_dict())
    best_val_accuracy = -np.inf
    best_epoch = 0
    history_rows = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate_loss_accuracy(model, loaders["val"], loss_fn, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    return TrainResult(
        model_key=spec.key,
        label=spec.label,
        model=model,
        params=params,
        history=history,
        best_val_accuracy=float(best_val_accuracy),
        runtime_seconds=time.perf_counter() - start,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train one epoch."""

    model.train()
    losses = []
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return float(np.mean(losses)), correct / max(total, 1)


def evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate loss and accuracy."""

    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses.append(float(loss_fn(logits, labels).detach().cpu()))
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return float(np.mean(losses)), correct / max(total, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, predictions, probabilities, and original indices."""

    model.eval()
    y_true, y_pred, probabilities, indices = [], [], [], []
    with torch.no_grad():
        for images, labels, batch_indices in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(np.argmax(probs, axis=1).tolist())
            probabilities.append(probs)
            indices.extend(batch_indices.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.vstack(probabilities), np.asarray(indices)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Compute required classification metrics."""

    top5 = np.argsort(probabilities, axis=1)[:, -5:]
    top5_error = float(np.mean([true not in top5_row for true, top5_row in zip(y_true, top5)]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top5_error": top5_error,
    }


def fgsm_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
    max_batches: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> float:
    """Evaluate FGSM accuracy with epsilon measured in pixel space.

    Batches arrive normalised for the model. The attack computes gradients through
    the normalised tensor, converts each image back to [0, 1] pixel space, applies
    the signed epsilon perturbation there, clamps to valid pixels, and normalises
    again before the adversarial forward pass.
    """

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)

    for batch_id, (images, labels, _) in enumerate(loader, start=1):
        if max_batches > 0 and batch_id > max_batches:
            break
        images = images.to(device).detach()
        labels = labels.to(device)
        images.requires_grad = True
        logits = model(images)
        loss = loss_fn(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        pixel_images = torch.clamp(images.detach() * std_tensor + mean_tensor, 0.0, 1.0)
        pixel_grad_sign = (images.grad / std_tensor).sign()
        perturbed_pixels = torch.clamp(pixel_images + epsilon * pixel_grad_sign, 0.0, 1.0)
        perturbed_normalised = ((perturbed_pixels - mean_tensor) / std_tensor).detach()
        adv_logits = model(perturbed_normalised)
        correct += int((adv_logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def denormalize(tensor: torch.Tensor, family: str) -> np.ndarray:
    """Convert one normalised CHW tensor to displayable HWC image."""

    mean = IMAGENET_MEAN if family == "resnet" else CIFAR_MEAN
    std = IMAGENET_STD if family == "resnet" else CIFAR_STD
    image = tensor.detach().cpu().clone()
    for channel, (channel_mean, channel_std) in enumerate(zip(mean, std)):
        image[channel] = image[channel] * channel_std + channel_mean
    image = image.clamp(0, 1)
    return np.transpose(image.numpy(), (1, 2, 0))


def plot_history(history: pd.DataFrame, label: str, output_path: Path) -> None:
    """Save loss and accuracy curves."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="s", label="val")
    axes[0].set_title(f"{label} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], marker="o", label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], marker="s", label="val")
    axes[1].set_title(f"{label} accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, label: str, output_path: Path) -> None:
    """Save confusion matrix heatmap."""

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{label} confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(10), CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(10), CIFAR10_CLASSES, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_misclassified_grid(
    result: TrainResult,
    loader: DataLoader,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
    max_examples: int,
) -> list[int]:
    """Save up to max_examples misclassified examples and return dataset indices."""

    misclassified_indices = indices[y_true != y_pred][:max_examples].tolist()
    if not misclassified_indices:
        return []
    index_lookup = {int(idx): (int(true), int(pred)) for idx, true, pred in zip(indices, y_true, y_pred)}
    dataset = loader.dataset
    images = []
    for item in range(len(dataset)):
        image, _, original_index = dataset[item]
        if original_index in misclassified_indices:
            images.append((image, original_index, *index_lookup[original_index]))
        if len(images) >= len(misclassified_indices):
            break

    cols = min(5, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (image, original_index, true, pred) in zip(axes_array, images):
        ax.imshow(denormalize(image, result.model_key if result.model_key != "resnet18" else "resnet"))
        ax.set_title(f"T:{CIFAR10_CLASSES[true]}\nP:{CIFAR10_CLASSES[pred]}", fontsize=8)
        ax.axis("off")
    for ax in axes_array[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return misclassified_indices


class GradCam:
    """Minimal Grad-CAM implementation for CNN/ResNet layers."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module: nn.Module, _grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_id: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[:, class_id].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)


def save_gradcam_examples(
    result: TrainResult,
    loader: DataLoader,
    misclassified_indices: list[int],
    output_path: Path,
    device: torch.device,
    max_examples: int,
) -> None:
    """Save Grad-CAM overlays for selected misclassified examples."""

    if not misclassified_indices:
        return
    if result.model_key == "cnn":
        target_layer = next(layer for layer in reversed(result.model.features) if isinstance(layer, nn.Conv2d))
    else:
        target_layer = result.model.layer4[-1].conv2
    gradcam = GradCam(result.model, target_layer)
    dataset = loader.dataset
    examples = []
    result.model.eval()
    for item in range(len(dataset)):
        image, label, original_index = dataset[item]
        if original_index not in misclassified_indices:
            continue
        image_batch = image.unsqueeze(0).to(device)
        logits = result.model(image_batch)
        pred = int(logits.argmax(dim=1).item())
        cam = gradcam(image_batch, pred)
        examples.append((image, cam, int(label), pred))
        if len(examples) >= max_examples:
            break

    cols = min(3, len(examples))
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    family = "resnet" if result.model_key == "resnet18" else "cnn"
    for ax, (image, cam, true, pred) in zip(axes_array, examples):
        display = denormalize(image, family)
        ax.imshow(display)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(f"T:{CIFAR10_CLASSES[true]}\nP:{CIFAR10_CLASSES[pred]}", fontsize=8)
        ax.axis("off")
    for ax in axes_array[len(examples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_lime_examples(
    result: TrainResult,
    test_dataset: datasets.CIFAR10,
    misclassified_indices: list[int],
    output_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> bool:
    """Save limited LIME explanations for MLP predictions."""

    if not misclassified_indices:
        return False
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries, slic
    except Exception:
        return False

    max_examples = min(int(config.get("interpretability", {}).get("lime_examples", 1)), len(misclassified_indices))
    num_samples = int(config.get("interpretability", {}).get("lime_samples", 50))
    explainer = lime_image.LimeImageExplainer(random_state=int(config.get("seed", 463)))
    transform = transforms_for("mlp", train=False)

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        tensors = []
        for image_array in images:
            pil_image = Image.fromarray(np.uint8(np.clip(image_array, 0, 1) * 255))
            tensors.append(transform(pil_image))
        batch = torch.stack(tensors).to(device)
        result.model.eval()
        with torch.no_grad():
            return torch.softmax(result.model(batch), dim=1).cpu().numpy()

    fig, axes = plt.subplots(max_examples, 2, figsize=(7, 3.5 * max_examples))
    axes_array = np.asarray(axes).reshape(max_examples, 2)
    for row_axes, dataset_index in zip(axes_array, misclassified_indices[:max_examples]):
        pil_image, true_label = test_dataset[dataset_index]
        image = np.asarray(pil_image).astype(np.float32) / 255.0
        probs = classifier_fn(np.asarray([image]))
        pred = int(np.argmax(probs, axis=1)[0])
        explanation = explainer.explain_instance(
            image,
            classifier_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10, sigma=1, start_label=0),
        )
        _temp, mask = explanation.get_image_and_mask(pred, positive_only=True, num_features=5, hide_rest=False)
        selected = mask.astype(bool)
        highlighted = image * 0.25
        highlighted[selected] = image[selected]
        highlighted = mark_boundaries(highlighted, selected, color=(1, 1, 0), mode="thick")
        row_axes[0].imshow(image)
        row_axes[0].set_title(f"Original\nT:{CIFAR10_CLASSES[int(true_label)]} P:{CIFAR10_CLASSES[pred]}", fontsize=8)
        row_axes[0].axis("off")
        row_axes[1].imshow(highlighted)
        row_axes[1].set_title("LIME positive superpixels", fontsize=8)
        row_axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def write_summary(
    output_path: Path,
    metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    best_params: dict[str, dict[str, Any]],
    lime_saved: bool,
) -> None:
    """Write compact run summary for report drafting."""

    lines = [
        "# Question 5 Run Summary",
        "",
        "## Best Hyperparameters",
        "```text",
        pd.DataFrame([{"model": key, **value} for key, value in best_params.items()]).to_string(index=False),
        "```",
        "",
        "## Final Metrics",
        "```text",
        metrics.to_string(index=False),
        "```",
        "",
        "## FGSM Robustness",
        "```text",
        robustness.to_string(index=False),
        "```",
        "",
        f"- LIME figure saved: {lime_saved}.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full Q5 workflow."""

    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    set_global_seed(int(config.get("seed", 463)))
    warnings.filterwarnings("ignore", category=UserWarning)

    run_dir = bootstrap_run("q5_neural_networks", config, SUMMARY, CHECKLIST, EXPECTED_ARTIFACTS)
    figures_dir = ensure_dir(run_dir / "figures")
    tables_dir = ensure_dir(run_dir / "tables")
    logs_dir = ensure_dir(run_dir / "logs")
    model_dir = ensure_dir(run_dir / "models")

    device = select_device()
    train_dataset, test_dataset, splits = load_cifar10(config)
    print(
        f"Loaded CIFAR-10 balanced subset: train={len(splits.train)}, "
        f"val={len(splits.val)}, test={len(splits.test)}, device={device}"
    )

    specs = model_specs(config)
    candidates = search_candidates(config)
    search_epochs = int(config.get("hyperparameter_optimisation", {}).get("search_epochs", 1))
    final_epochs = int(config.get("training", {}).get("epochs", 4))
    search_rows = []
    best_params: dict[str, dict[str, Any]] = {}

    for spec in specs:
        best_score = -np.inf
        best_candidate = candidates[0]
        for candidate_id, params in enumerate(candidates, start=1):
            result = train_one_model(spec, params, train_dataset, test_dataset, splits, config, search_epochs, device)
            search_rows.append(
                {
                    "model_key": spec.key,
                    "candidate_id": candidate_id,
                    "best_val_accuracy": result.best_val_accuracy,
                    "runtime_seconds": result.runtime_seconds,
                    **params,
                }
            )
            if result.best_val_accuracy > best_score:
                best_score = result.best_val_accuracy
                best_candidate = params
        best_params[spec.key] = best_candidate
        print(f"Selected params for {spec.key}: {best_candidate}")

    search_frame = pd.DataFrame(search_rows)
    search_frame.to_csv(tables_dir / "hyperparameter_search.csv", index=False)

    final_results: list[TrainResult] = []
    metric_rows = []
    recall_rows = []
    robustness_rows = []
    lime_saved = False
    misclassified_counts = {}

    for spec in specs:
        result = train_one_model(spec, best_params[spec.key], train_dataset, test_dataset, splits, config, final_epochs, device)
        final_results.append(result)
        result.history.to_csv(logs_dir / f"{spec.key}_training_history.csv", index=False)
        plot_history(result.history, result.label, figures_dir / f"{spec.key}_training_curves.png")
        torch.save(result.model.state_dict(), model_dir / f"{spec.key}.pt")

        loaders = make_loaders(
            spec.family,
            int(result.params["batch_size"]),
            train_dataset,
            test_dataset,
            splits,
            int(config.get("training", {}).get("num_workers", 0)),
        )
        y_true, y_pred, probabilities, indices = predict(result.model, loaders["test"], device)
        metrics = compute_metrics(y_true, y_pred, probabilities)
        metric_rows.append(
            {
                "model_key": spec.key,
                "model_label": result.label,
                "best_val_accuracy": result.best_val_accuracy,
                "configured_epochs": final_epochs,
                "actual_epochs": int(len(result.history)),
                "early_stopped": bool(len(result.history) < final_epochs),
                "runtime_seconds": result.runtime_seconds,
                **metrics,
                **result.params,
            }
        )
        recalls = recall_score(y_true, y_pred, labels=np.arange(10), average=None, zero_division=0)
        for class_name, recall in zip(CIFAR10_CLASSES, recalls):
            recall_rows.append({"model_key": spec.key, "class_name": class_name, "recall": float(recall)})
        plot_confusion(y_true, y_pred, result.label, figures_dir / f"{spec.key}_confusion_matrix.png")

        max_examples = int(config.get("interpretability", {}).get("examples_per_model", 10))
        misclassified = save_misclassified_grid(
            result,
            loaders["test"],
            y_true,
            y_pred,
            indices,
            figures_dir / f"{spec.key}_misclassified_examples.png",
            max_examples,
        )
        misclassified_counts[spec.key] = len(misclassified)
        if spec.key in {"cnn", "resnet18"}:
            save_gradcam_examples(
                result,
                loaders["test"],
                misclassified,
                figures_dir / f"{spec.key}_gradcam_misclassified.png",
                device,
                max_examples=min(6, max_examples),
            )
        if spec.key == "mlp":
            lime_saved = save_lime_examples(result, test_dataset, misclassified, figures_dir / "mlp_lime_examples.png", config, device)

        epsilon = float(config.get("robustness", {}).get("fgsm_epsilon", 0.03))
        family = "resnet" if spec.family == "resnet" else "cifar"
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if family == "resnet" else (CIFAR_MEAN, CIFAR_STD)
        robustness_rows.append(
            {
                "model_key": spec.key,
                "model_label": result.label,
                "attack": "fgsm",
                "epsilon": epsilon,
                "accuracy_under_attack": fgsm_accuracy(
                    result.model,
                    loaders["test"],
                    device,
                    epsilon,
                    int(config.get("robustness", {}).get("max_batches", 0)),
                    mean,
                    std,
                ),
            }
        )
        print(f"Finished final evaluation for {spec.key}.")

    metrics_frame = pd.DataFrame(metric_rows)
    recalls_frame = pd.DataFrame(recall_rows)
    robustness_frame = pd.DataFrame(robustness_rows)
    metrics_frame.to_csv(tables_dir / "final_metrics.csv", index=False)
    recalls_frame.to_csv(tables_dir / "per_class_recall.csv", index=False)
    robustness_frame.to_csv(tables_dir / "fgsm_robustness.csv", index=False)
    write_json(
        tables_dir / "run_metadata.json",
        {
            "train_size": len(splits.train),
            "validation_size": len(splits.val),
            "test_size": len(splits.test),
            "device": str(device),
            "configured_final_epochs": final_epochs,
            "actual_epochs": {result.model_key: int(len(result.history)) for result in final_results},
            "early_stopped": {result.model_key: bool(len(result.history) < final_epochs) for result in final_results},
            "runtime_seconds_scope": "final model training only; hyperparameter-search runtimes are reported separately",
            "fgsm_epsilon_space": "pixel space [0, 1], then re-normalized before model evaluation",
            "best_params": best_params,
            "misclassified_examples_saved": misclassified_counts,
            "lime_saved": lime_saved,
        },
    )
    joblib.dump(best_params, model_dir / "best_hyperparameters.joblib")
    write_summary(run_dir / "summary.md", metrics_frame, robustness_frame, best_params, lime_saved)

    print(f"Saved Question 5 artifacts at {run_dir}")


if __name__ == "__main__":
    main()
