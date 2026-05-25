"""
Sweep RFI decision thresholds for a trained multitask checkpoint.
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
from models.train_engine import evaluate, make_loss_fn
from scripts.train_model import SpectrogramDataset, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep RFI thresholds for validation accuracy.")
    parser.add_argument("--checkpoint", default="models/best_multitask_model.pt", help="Checkpoint path.")
    parser.add_argument("--data-dir", default="data/val", help="Validation directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--image-height", type=int, default=224, help="Resized image height.")
    parser.add_argument("--image-width", type=int, default=224, help="Resized image width.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Evaluation device.")
    parser.add_argument("--start", type=float, default=0.05, help="First threshold.")
    parser.add_argument("--stop", type=float, default=0.95, help="Last threshold.")
    parser.add_argument("--step", type=float, default=0.05, help="Threshold step.")
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

    thresholds = []
    value = args.start
    while value <= args.stop + 1e-9:
        thresholds.append(round(value, 4))
        value += args.step

    rows = []
    for threshold in thresholds:
        metrics = evaluate(model, loader, make_loss_fn(), device, rfi_threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "rfi_accuracy": metrics["rfi_accuracy"],
                "rfi_precision": metrics["rfi_precision"],
                "rfi_recall": metrics["rfi_recall"],
                "rfi_f1": metrics["rfi_f1"],
                "solar_accuracy": metrics["accuracy"],
                "burst_macro_recall": metrics["burst_macro_recall"],
            }
        )

    rows.sort(key=lambda row: (row["rfi_accuracy"], row["solar_accuracy"]), reverse=True)
    print("Top thresholds by RFI accuracy:")
    for row in rows[:10]:
        print(
            f"threshold={row['threshold']:.4f} "
            f"rfi_acc={row['rfi_accuracy']:.4f} "
            f"rfi_precision={row['rfi_precision']:.4f} "
            f"rfi_recall={row['rfi_recall']:.4f} "
            f"rfi_f1={row['rfi_f1']:.4f} "
            f"solar_acc={row['solar_accuracy']:.4f} "
            f"burst_recall={row['burst_macro_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
