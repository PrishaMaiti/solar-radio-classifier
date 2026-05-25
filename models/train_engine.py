"""
Training engine for solar radio classifier.
Contains training loops, loss functions, and classification metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .architecture import CLASS_NAMES


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Support common DataLoader batch shapes without forcing one dataset API."""
    if isinstance(batch, dict):
        inputs = batch.get("image", batch.get("inputs", batch.get("x")))
        targets = batch.get("label", batch.get("target", batch.get("y")))
        if inputs is None or targets is None:
            raise KeyError("Batch dict must include image/inputs/x and label/target/y keys.")
        return inputs, targets

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError("Batch must be a tuple/list of (inputs, targets) or a supported dict.")


def build_class_weights(
    class_counts: list[int] | tuple[int, ...] | torch.Tensor,
    device: torch.device | str | None = None,
    burst_weight: float = 1.0,
    no_burst_weight: float = 1.0,
    rfi_weight: float = 1.0,
) -> torch.Tensor:
    """
    Build inverse-frequency class weights for CrossEntropyLoss.

    Args:
        class_counts: Counts ordered to match CLASS_NAMES.
        device: Optional target device.
        burst_weight: Extra multiplier for type_1 through type_8 classes.
        no_burst_weight: Extra multiplier for the no_burst class.
        rfi_weight: Extra multiplier for the rfi class.

    Returns:
        Tensor of loss weights ordered to match CLASS_NAMES.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float)
    if torch.any(counts <= 0):
        raise ValueError("All class counts must be positive to build class weights.")

    weights = counts.sum() / (len(counts) * counts)
    multipliers = torch.ones_like(weights)
    if len(weights) >= 10:
        multipliers[:8] *= burst_weight
        multipliers[8] *= no_burst_weight
        multipliers[9] *= rfi_weight
    weights = weights * multipliers
    return weights.to(device) if device is not None else weights


def make_loss_fn(
    class_weights: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> nn.CrossEntropyLoss:
    """Create the default loss function for raw logits."""
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(weight=class_weights)


def _empty_confusion_matrix(num_classes: int, device: torch.device | str) -> torch.Tensor:
    return torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Update a confusion matrix in-place.

    Rows are true classes and columns are predicted classes.
    """
    num_classes = confusion_matrix.shape[0]
    targets = targets.view(-1).to(torch.long)
    predictions = predictions.view(-1).to(torch.long)
    valid = (targets >= 0) & (targets < num_classes)
    indices = targets[valid] * num_classes + predictions[valid]
    confusion_matrix.view(-1).index_add_(
        0,
        indices,
        torch.ones_like(indices, dtype=torch.long, device=confusion_matrix.device),
    )
    return confusion_matrix


def metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """
    Calculate accuracy, precision, recall, F1, and false-negative rate.

    Rows are true classes and columns are predicted classes.
    """
    cm = confusion_matrix.detach().cpu().to(torch.float)
    true_positive = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)

    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    false_negative_rate = (support - true_positive) / support.clamp_min(1)

    total = support.sum().item()
    accuracy = true_positive.sum().item() / total if total else 0.0
    class_count = len(class_names)
    burst_indices = [idx for idx, name in enumerate(class_names) if name.startswith("type_")]

    return {
        "accuracy": accuracy,
        "macro_precision": precision[:class_count].mean().item(),
        "macro_recall": recall[:class_count].mean().item(),
        "macro_f1": f1[:class_count].mean().item(),
        "burst_macro_recall": recall[burst_indices].mean().item() if burst_indices else 0.0,
        "per_class": {
            class_names[idx]: {
                "precision": precision[idx].item(),
                "recall": recall[idx].item(),
                "f1": f1[idx].item(),
                "false_negative_rate": false_negative_rate[idx].item(),
                "support": int(support[idx].item()),
            }
            for idx in range(class_count)
        },
    }


def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: tuple[str, ...] = CLASS_NAMES,
    save_path: str | Path | None = None,
    normalize: bool = False,
) -> None:
    """Save or display a confusion matrix image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix.detach().cpu().to(torch.float)
    if normalize:
        cm = cm / cm.sum(dim=1, keepdim=True).clamp_min(1)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm.numpy(), cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col].item()
            label = f"{value:.2f}" if normalize else str(int(value))
            ax.text(col, row, label, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Train model for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training data loader.
        optimizer: Optimization algorithm.
        loss_fn: Loss function.
        device: Device to train on.

    Returns:
        dict: Average loss, accuracy, confusion matrix, and metrics.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_classes = getattr(model, "num_classes", len(CLASS_NAMES))
    confusion_matrix = _empty_confusion_matrix(num_classes, device)

    for batch in dataloader:
        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        predictions = logits.argmax(dim=1)
        update_confusion_matrix(confusion_matrix, targets, predictions)

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    return {
        "loss": total_loss / max(total_samples, 1),
        "confusion_matrix": confusion_matrix.detach().cpu(),
        **metrics,
    }


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate model on validation/test set.

    Args:
        model: The neural network model.
        dataloader: Validation/test data loader.
        loss_fn: Loss function.
        device: Device to evaluate on.

    Returns:
        dict: Loss, probabilities, confusion matrix, and metrics.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    num_classes = getattr(model, "num_classes", len(CLASS_NAMES))
    confusion_matrix = _empty_confusion_matrix(num_classes, device)
    probabilities = []
    targets_seen = []

    for batch in dataloader:
        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device).long()

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)
        update_confusion_matrix(confusion_matrix, targets, predictions)
        probabilities.append(torch.softmax(logits, dim=1).detach().cpu())
        targets_seen.append(targets.detach().cpu())

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    return {
        "loss": total_loss / max(total_samples, 1),
        "confusion_matrix": confusion_matrix.detach().cpu(),
        "probabilities": torch.cat(probabilities) if probabilities else torch.empty(0, num_classes),
        "targets": torch.cat(targets_seen) if targets_seen else torch.empty(0, dtype=torch.long),
        **metrics,
    }


def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    class_weights: torch.Tensor | None = None,
    weight_decay: float = 1e-4,
    checkpoint_path: str | Path | None = None,
    confusion_matrix_dir: str | Path | None = None,
    selection_metric: str = "burst_macro_recall",
):
    """
    Full training pipeline.

    Args:
        model: The neural network model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        device: Device to train on.
        class_weights: Optional CrossEntropyLoss class weights.
        weight_decay: AdamW weight decay.
        checkpoint_path: Optional path for the best model state dict.
        confusion_matrix_dir: Optional directory for per-epoch validation matrices.
        selection_metric: Validation metric used to identify the best model.

    Returns:
        dict: Training history and best model metadata.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = make_loss_fn(class_weights=class_weights, device=device)

    history: dict[str, Any] = {
        "train": [],
        "val": [],
        "best_epoch": None,
        "best_metric": float("-inf"),
        "selection_metric": selection_metric,
    }

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if confusion_matrix_dir is not None:
            matrix_path = Path(confusion_matrix_dir) / f"epoch_{epoch:03d}_val_confusion_matrix.png"
            plot_confusion_matrix(val_metrics["confusion_matrix"], save_path=matrix_path)

        metric_value = val_metrics.get(selection_metric)
        if metric_value is None:
            raise KeyError(f"Validation metric '{selection_metric}' was not calculated.")

        if metric_value > history["best_metric"]:
            history["best_metric"] = metric_value
            history["best_epoch"] = epoch
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "selection_metric": selection_metric,
                        "metric_value": metric_value,
                        "class_names": CLASS_NAMES,
                    },
                    checkpoint_path,
                )

    return history
