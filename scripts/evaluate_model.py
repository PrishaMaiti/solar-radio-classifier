"""
Evaluate a trained solar radio burst classifier checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.architecture import CLASS_NAMES, SolarRadioClassifier
from models.train_engine import evaluate, make_loss_fn, plot_confusion_matrix
from scripts.train_model import SpectrogramDataset, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a solar radio burst checkpoint.")
    parser.add_argument("--checkpoint", default="models/best_merged_model.pt", help="Checkpoint path.")
    parser.add_argument("--data-dir", default="data/test", help="Directory of labeled PNGs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image-height", type=int, default=224, help="Resized image height.")
    parser.add_argument("--image-width", type=int, default=224, help="Resized image width.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Evaluation device.")
    parser.add_argument(
        "--confusion-matrix",
        default="models/test_confusion_matrix.png",
        help="Output path for the test confusion matrix image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    _, eval_transform = build_transforms((args.image_height, args.image_width))
    dataset = SpectrogramDataset(args.data_dir, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SolarRadioClassifier(in_channels=1, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, loader, make_loss_fn(), device)
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=args.confusion_matrix)

    print(f"Test samples: {len(dataset)}")
    print("Class order:", ", ".join(CLASS_NAMES))
    print(f"test_loss={metrics['loss']:.4f}")
    print(f"test_acc={metrics['accuracy']:.4f}")
    print(f"test_burst_macro_recall={metrics['burst_macro_recall']:.4f}")
    print("Per-class metrics:")
    for class_name, values in metrics["per_class"].items():
        print(
            f"  {class_name}: "
            f"recall={values['recall']:.4f}, "
            f"precision={values['precision']:.4f}, "
            f"f1={values['f1']:.4f}, "
            f"fnr={values['false_negative_rate']:.4f}, "
            f"support={values['support']}"
        )
    print("Confusion matrix:")
    print(metrics["confusion_matrix"].numpy())
    print(f"Confusion matrix image: {args.confusion_matrix}")


if __name__ == "__main__":
    main()
