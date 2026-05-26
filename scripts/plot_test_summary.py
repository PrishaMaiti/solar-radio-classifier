"""
Create a presentation-friendly summary figure for final test results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.architecture import CLASS_NAMES, SolarRadioClassifier
from models.train_engine import evaluate, make_loss_fn
from scripts.train_model import SpectrogramDataset, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final test summary metrics.")
    parser.add_argument("--checkpoint", default="models/best_multitask_rfiw_2_0.pt")
    parser.add_argument("--data-dir", default="data/test")
    parser.add_argument("--output", default="models/final_test_summary.png")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rfi-threshold", type=float, default=0.65)
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
    metrics = evaluate(model, loader, make_loss_fn(), device, rfi_threshold=args.rfi_threshold)

    summary_labels = ["Solar Acc.", "Burst Recall", "RFI Acc.", "RFI F1"]
    summary_values = [
        metrics["accuracy"],
        metrics["burst_macro_recall"],
        metrics["rfi_accuracy"],
        metrics["rfi_f1"],
    ]

    class_recalls = [metrics["per_class"][name]["recall"] for name in CLASS_NAMES]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 10.5),
        gridspec_kw={"height_ratios": [1, 1.35]},
    )
    fig.suptitle("Final Test Performance", fontsize=24, fontweight="bold", y=0.98)

    colors = ["#2f6fbb", "#3f9c72", "#c67c2e", "#8a5fbf"]
    bars = ax_top.bar(summary_labels, summary_values, color=colors)
    ax_top.set_ylim(0, 1.16)
    ax_top.set_ylabel("Score", fontsize=15)
    ax_top.tick_params(axis="x", labelsize=14)
    ax_top.tick_params(axis="y", labelsize=13)
    ax_top.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary_values):
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )

    recall_bars = ax_bottom.bar(CLASS_NAMES, class_recalls, color="#4c7f9f")
    ax_bottom.set_ylim(0, 1.16)
    ax_bottom.set_ylabel("Recall", fontsize=15)
    ax_bottom.text(
        0.5,
        1.13,
        "Solar Class Recall",
        transform=ax_bottom.transAxes,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )
    ax_bottom.tick_params(axis="x", labelrotation=35, labelsize=13)
    ax_bottom.tick_params(axis="y", labelsize=13)
    ax_bottom.grid(axis="y", alpha=0.25)
    for bar, value in zip(recall_bars, class_recalls):
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    note = (
        f"Test samples: {len(dataset)} | RFI threshold: {args.rfi_threshold:.2f} | "
        f"RFI confusion: TN={int(metrics['rfi_confusion_matrix'][0, 0])}, "
        f"FP={int(metrics['rfi_confusion_matrix'][0, 1])}, "
        f"FN={int(metrics['rfi_confusion_matrix'][1, 0])}, "
        f"TP={int(metrics['rfi_confusion_matrix'][1, 1])}"
    )
    fig.text(0.5, 0.035, note, ha="center", fontsize=13)

    fig.subplots_adjust(hspace=0.52)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
