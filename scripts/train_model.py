"""
Train the solar radio burst classifier on PNG spectrograms.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.architecture import CLASS_NAMES, SolarRadioClassifier
from models.train_engine import build_class_weights, train


LABEL_PATTERN = re.compile(r"_(type_\d+|no_burst)(?:_(clean|rfi))?\.png$")
LABEL_ALIASES = {
    "type_1": "type_8",
}
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class SpectrogramDataset(Dataset):
    """Dataset for filename-labeled spectrogram PNGs."""

    def __init__(self, data_dir: str | Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._discover_samples()

    def _discover_samples(self) -> list[tuple[Path, int, int]]:
        samples: list[tuple[Path, int, int]] = []
        for path in sorted(self.data_dir.glob("*.png")):
            match = LABEL_PATTERN.search(path.name)
            if match is None:
                continue
            label = match.group(1)
            rfi_state = match.group(2)
            label = LABEL_ALIASES.get(label, label)
            if label not in CLASS_TO_INDEX:
                continue
            rfi_label = 1 if rfi_state == "rfi" else 0
            samples.append((path, CLASS_TO_INDEX[label], rfi_label))

        if not samples:
            raise FileNotFoundError(f"No labeled PNG samples found in {self.data_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        path, label, rfi_label = self.samples[index]
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, rfi_label

    def class_counts(self) -> list[int]:
        counts = Counter(label for _, label, _ in self.samples)
        return [counts[index] for index in range(len(CLASS_NAMES))]

    def rfi_counts(self) -> dict[str, int]:
        counts = Counter(rfi_label for _, _, rfi_label in self.samples)
        return {"clean": counts[0], "rfi": counts[1]}


def build_transforms(image_size: tuple[int, int]) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    return train_transform, eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the solar radio burst CNN.")
    parser.add_argument("--train-dir", default="data/train", help="Directory of training PNGs.")
    parser.add_argument("--val-dir", default="data/val", help="Directory of validation PNGs.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--image-height", type=int, default=224, help="Resized image height.")
    parser.add_argument("--image-width", type=int, default=224, help="Resized image width.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. Use cpu if your GPU is not supported by the installed PyTorch build.",
    )
    parser.add_argument("--burst-weight", type=float, default=1.25, help="Extra loss weight for burst classes.")
    parser.add_argument("--no-burst-weight", type=float, default=1.0, help="Extra loss weight for no_burst.")
    parser.add_argument("--rfi-loss-weight", type=float, default=1.0, help="Multiplier for binary RFI loss.")
    parser.add_argument("--rfi-threshold", type=float, default=0.5, help="RFI-present probability threshold.")
    parser.add_argument("--selection-metric", default="joint_score", help="Validation metric for best checkpoint.")
    parser.add_argument("--checkpoint", default="models/best_model.pt", help="Best checkpoint output path.")
    parser.add_argument(
        "--confusion-matrix-dir",
        default="models/confusion_matrices",
        help="Directory for per-epoch validation confusion matrices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    image_size = (args.image_height, args.image_width)
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = SpectrogramDataset(args.train_dir, transform=train_transform)
    val_dataset = SpectrogramDataset(args.val_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SolarRadioClassifier(in_channels=1, num_classes=len(CLASS_NAMES))
    class_weights = build_class_weights(
        train_dataset.class_counts(),
        device=device,
        burst_weight=args.burst_weight,
        no_burst_weight=args.no_burst_weight,
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("Class order:", ", ".join(CLASS_NAMES))
    print("Class counts:", train_dataset.class_counts())
    print("RFI counts:", train_dataset.rfi_counts())

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        class_weights=class_weights,
        rfi_loss_weight=args.rfi_loss_weight,
        rfi_threshold=args.rfi_threshold,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint,
        confusion_matrix_dir=args.confusion_matrix_dir,
        selection_metric=args.selection_metric,
    )

    for epoch, (train_metrics, val_metrics) in enumerate(zip(history["train"], history["val"]), start=1):
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_burst_recall={val_metrics['burst_macro_recall']:.4f} "
            f"val_rfi_recall={val_metrics['rfi_recall']:.4f} "
            f"val_rfi_f1={val_metrics['rfi_f1']:.4f}"
        )

    print(
        f"Best epoch: {history['best_epoch']} "
        f"({history['selection_metric']}={history['best_metric']:.4f})"
    )
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Confusion matrices: {args.confusion_matrix_dir}")


if __name__ == "__main__":
    main()
