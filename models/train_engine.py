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


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Support common DataLoader batch shapes without forcing one dataset API."""
    if isinstance(batch, dict):
        inputs = batch.get("image", batch.get("inputs", batch.get("x")))
        targets = batch.get("label", batch.get("target", batch.get("y"), batch.get("solar_label")))
        rfi_targets = batch.get("rfi_label", batch.get("rfi_target"))
        if inputs is None or targets is None:
            raise KeyError("Batch dict must include image/inputs/x and label/target/y keys.")
        return inputs, targets, rfi_targets

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        rfi_targets = batch[2] if len(batch) >= 3 else None
        return batch[0], batch[1], rfi_targets

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
        burst_weight: Extra multiplier for type_* classes.
        no_burst_weight: Extra multiplier for the no_burst class.
        rfi_weight: Extra multiplier for the rfi class.

    Returns:
        Tensor of loss weights ordered to match CLASS_NAMES.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float)
    if torch.any(counts <= 0):
        raise ValueError("All class counts must be positive to build class weights.")

    weights = counts.sum() / (len(counts) * counts)
    if len(weights) != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} class counts, got {len(weights)}.")

    multipliers = torch.ones_like(weights)
    for index, class_name in enumerate(CLASS_NAMES):
        if class_name.startswith("type_"):
            multipliers[index] *= burst_weight
        elif class_name == "no_burst":
            multipliers[index] *= no_burst_weight
        elif class_name == "rfi":
            multipliers[index] *= rfi_weight
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


def make_rfi_loss_fn(
    pos_weight: torch.Tensor | float | None = None,
    device: torch.device | str | None = None,
) -> nn.BCEWithLogitsLoss:
    """Create the binary RFI loss function for raw logits."""
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(float(pos_weight), dtype=torch.float)
    if pos_weight is not None and device is not None:
        pos_weight = pos_weight.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def multitask_loss(
    outputs: dict[str, torch.Tensor] | torch.Tensor,
    solar_targets: torch.Tensor,
    rfi_targets: torch.Tensor | None,
    solar_loss_fn: nn.CrossEntropyLoss,
    rfi_loss_fn: nn.BCEWithLogitsLoss,
    rfi_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate combined solar classification and RFI detection loss."""
    if isinstance(outputs, torch.Tensor):
        solar_logits = outputs
        rfi_logits = None
    else:
        solar_logits = outputs["solar_logits"]
        rfi_logits = outputs.get("rfi_logits")

    solar_loss = solar_loss_fn(solar_logits, solar_targets)
    if rfi_targets is None or rfi_logits is None:
        rfi_loss = torch.zeros((), device=solar_loss.device)
    else:
        rfi_loss = rfi_loss_fn(rfi_logits, rfi_targets.float())

    return solar_loss + rfi_loss_weight * rfi_loss, solar_loss, rfi_loss


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


def binary_metrics_from_confusion_matrix(confusion_matrix: torch.Tensor) -> dict[str, Any]:
    """Calculate binary metrics for a 2x2 confusion matrix."""
    cm = confusion_matrix.detach().cpu().to(torch.float)
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]
    true_positive = cm[1, 1]
    total = cm.sum().item()

    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    specificity = true_negative / (true_negative + false_positive).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    return {
        "rfi_accuracy": (true_positive + true_negative).item() / total if total else 0.0,
        "rfi_precision": precision.item(),
        "rfi_recall": recall.item(),
        "rfi_specificity": specificity.item(),
        "rfi_f1": f1.item(),
        "rfi_false_negative_rate": (1.0 - recall).item(),
        "rfi_false_positive_rate": (1.0 - specificity).item(),
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


def _solar_logits(outputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else outputs["solar_logits"]


def _rfi_logits(outputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor | None:
    return None if isinstance(outputs, torch.Tensor) else outputs.get("rfi_logits")


def train_epoch(
    model,
    dataloader,
    optimizer,
    solar_loss_fn,
    device,
    rfi_loss_fn: nn.BCEWithLogitsLoss | None = None,
    rfi_loss_weight: float = 1.0,
    rfi_threshold: float = 0.5,
):
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
    total_solar_loss = 0.0
    total_rfi_loss = 0.0
    total_samples = 0
    num_classes = getattr(model, "num_classes", len(CLASS_NAMES))
    confusion_matrix = _empty_confusion_matrix(num_classes, device)
    rfi_confusion_matrix = _empty_confusion_matrix(2, device)

    for batch in dataloader:
        inputs, targets, rfi_targets = _unpack_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        if rfi_targets is not None:
            rfi_targets = rfi_targets.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if rfi_loss_fn is None:
            rfi_loss_fn = make_rfi_loss_fn(device=device)
        loss, solar_loss, rfi_loss = multitask_loss(
            outputs,
            targets,
            rfi_targets,
            solar_loss_fn,
            rfi_loss_fn,
            rfi_loss_weight,
        )
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_solar_loss += solar_loss.item() * batch_size
        total_rfi_loss += rfi_loss.item() * batch_size
        total_samples += batch_size
        predictions = _solar_logits(outputs).argmax(dim=1)
        update_confusion_matrix(confusion_matrix, targets, predictions)
        rfi_logits = _rfi_logits(outputs)
        if rfi_targets is not None and rfi_logits is not None:
            rfi_predictions = (torch.sigmoid(rfi_logits) >= rfi_threshold).long()
            update_confusion_matrix(rfi_confusion_matrix, rfi_targets.long(), rfi_predictions)

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    rfi_metrics = binary_metrics_from_confusion_matrix(rfi_confusion_matrix)
    joint_score = 0.5 * (metrics["burst_macro_recall"] + rfi_metrics["rfi_f1"])
    return {
        "loss": total_loss / max(total_samples, 1),
        "solar_loss": total_solar_loss / max(total_samples, 1),
        "rfi_loss": total_rfi_loss / max(total_samples, 1),
        "confusion_matrix": confusion_matrix.detach().cpu(),
        "rfi_confusion_matrix": rfi_confusion_matrix.detach().cpu(),
        "joint_score": joint_score,
        **metrics,
        **rfi_metrics,
    }


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    solar_loss_fn,
    device,
    rfi_loss_fn: nn.BCEWithLogitsLoss | None = None,
    rfi_loss_weight: float = 1.0,
    rfi_threshold: float = 0.5,
):
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
    total_solar_loss = 0.0
    total_rfi_loss = 0.0
    total_samples = 0
    num_classes = getattr(model, "num_classes", len(CLASS_NAMES))
    confusion_matrix = _empty_confusion_matrix(num_classes, device)
    rfi_confusion_matrix = _empty_confusion_matrix(2, device)
    probabilities = []
    rfi_probabilities = []
    targets_seen = []
    rfi_targets_seen = []
    if rfi_loss_fn is None:
        rfi_loss_fn = make_rfi_loss_fn(device=device)

    for batch in dataloader:
        inputs, targets, rfi_targets = _unpack_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        if rfi_targets is not None:
            rfi_targets = rfi_targets.to(device).float()

        outputs = model(inputs)
        loss, solar_loss, rfi_loss = multitask_loss(
            outputs,
            targets,
            rfi_targets,
            solar_loss_fn,
            rfi_loss_fn,
            rfi_loss_weight,
        )
        solar_logits = _solar_logits(outputs)
        rfi_logits = _rfi_logits(outputs)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_solar_loss += solar_loss.item() * batch_size
        total_rfi_loss += rfi_loss.item() * batch_size
        total_samples += batch_size

        predictions = solar_logits.argmax(dim=1)
        update_confusion_matrix(confusion_matrix, targets, predictions)
        probabilities.append(torch.softmax(solar_logits, dim=1).detach().cpu())
        targets_seen.append(targets.detach().cpu())
        if rfi_targets is not None and rfi_logits is not None:
            rfi_probs = torch.sigmoid(rfi_logits)
            rfi_predictions = (rfi_probs >= rfi_threshold).long()
            update_confusion_matrix(rfi_confusion_matrix, rfi_targets.long(), rfi_predictions)
            rfi_probabilities.append(rfi_probs.detach().cpu())
            rfi_targets_seen.append(rfi_targets.detach().cpu())

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    rfi_metrics = binary_metrics_from_confusion_matrix(rfi_confusion_matrix)
    joint_score = 0.5 * (metrics["burst_macro_recall"] + rfi_metrics["rfi_f1"])
    return {
        "loss": total_loss / max(total_samples, 1),
        "solar_loss": total_solar_loss / max(total_samples, 1),
        "rfi_loss": total_rfi_loss / max(total_samples, 1),
        "confusion_matrix": confusion_matrix.detach().cpu(),
        "rfi_confusion_matrix": rfi_confusion_matrix.detach().cpu(),
        "joint_score": joint_score,
        "probabilities": torch.cat(probabilities) if probabilities else torch.empty(0, num_classes),
        "rfi_probabilities": torch.cat(rfi_probabilities) if rfi_probabilities else torch.empty(0),
        "targets": torch.cat(targets_seen) if targets_seen else torch.empty(0, dtype=torch.long),
        "rfi_targets": torch.cat(rfi_targets_seen) if rfi_targets_seen else torch.empty(0, dtype=torch.float),
        **metrics,
        **rfi_metrics,
    }


def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    class_weights: torch.Tensor | None = None,
    rfi_pos_weight: torch.Tensor | float | None = None,
    rfi_loss_weight: float = 1.0,
    rfi_threshold: float = 0.5,
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
        rfi_pos_weight: Optional positive-class weight for RFI BCE loss.
        rfi_loss_weight: Multiplier for the RFI loss in total loss.
        rfi_threshold: Probability threshold for RFI-present predictions.
        weight_decay: AdamW weight decay.
        checkpoint_path: Optional path for the best model state dict.
        confusion_matrix_dir: Optional directory for per-epoch validation matrices.
        selection_metric: Validation metric used to identify the best model.

    Returns:
        dict: Training history and best model metadata.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    solar_loss_fn = make_loss_fn(class_weights=class_weights, device=device)
    rfi_loss_fn = make_rfi_loss_fn(pos_weight=rfi_pos_weight, device=device)

    history: dict[str, Any] = {
        "train": [],
        "val": [],
        "best_epoch": None,
        "best_metric": float("-inf"),
        "selection_metric": selection_metric,
    }

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            solar_loss_fn,
            device,
            rfi_loss_fn=rfi_loss_fn,
            rfi_loss_weight=rfi_loss_weight,
            rfi_threshold=rfi_threshold,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            solar_loss_fn,
            device,
            rfi_loss_fn=rfi_loss_fn,
            rfi_loss_weight=rfi_loss_weight,
            rfi_threshold=rfi_threshold,
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if confusion_matrix_dir is not None:
                matrix_path = Path(confusion_matrix_dir) / f"epoch_{epoch:03d}_val_confusion_matrix.png"
                plot_confusion_matrix(val_metrics["confusion_matrix"], save_path=matrix_path)
                rfi_matrix_path = Path(confusion_matrix_dir) / f"epoch_{epoch:03d}_val_rfi_confusion_matrix.png"
                plot_confusion_matrix(
                    val_metrics["rfi_confusion_matrix"],
                    class_names=("clean", "rfi"),
                    save_path=rfi_matrix_path,
                )

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
                        "rfi_labels": ("clean", "rfi"),
                        "rfi_threshold": rfi_threshold,
                    },
                    checkpoint_path,
                )

    return history
