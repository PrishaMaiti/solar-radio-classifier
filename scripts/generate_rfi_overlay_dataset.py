"""
Generate clean and RFI-contaminated spectrogram datasets.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_gen.synthesizer import (
    generate_no_burst,
    generate_rfi,
    generate_type_2,
    generate_type_3,
    generate_type_4,
    generate_type_5,
    generate_type_6,
    generate_type_7,
    generate_type_8,
)


CLASS_GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "type_2": generate_type_2,
    "type_3": generate_type_3,
    "type_4": generate_type_4,
    "type_5": generate_type_5,
    "type_6": generate_type_6,
    "type_7": generate_type_7,
    "type_8": generate_type_8,
    "no_burst": generate_no_burst,
}


def normalize_to_uint8(spectrogram: np.ndarray) -> np.ndarray:
    """Robustly scale a spectrogram to 8-bit grayscale."""
    array = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(array, [1, 99.7])
    if high <= low:
        high = low + 1.0
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def add_rfi_overlay(spectrogram: np.ndarray, strength: float | None = None) -> np.ndarray:
    """Add RFI artifacts to an existing burst/no-burst spectrogram."""
    if strength is None:
        strength = float(np.random.uniform(0.7, 1.3))

    rfi = generate_rfi(freq_bins=spectrogram.shape[0], time_bins=spectrogram.shape[1])
    rfi_component = rfi - np.percentile(rfi, 20)
    rfi_component = np.clip(rfi_component, 0.0, None)
    return spectrogram + strength * rfi_component


def save_spectrogram(spectrogram: np.ndarray, path: Path) -> None:
    image = Image.fromarray(normalize_to_uint8(spectrogram), mode="L")
    image.save(path)


def reset_split_dir(split_dir: Path, project_root: Path) -> None:
    split_dir = split_dir.resolve()
    data_root = (project_root / "data").resolve()
    if data_root not in split_dir.parents:
        raise ValueError(f"Refusing to clear directory outside data root: {split_dir}")

    split_dir.mkdir(parents=True, exist_ok=True)
    for png_path in split_dir.glob("*.png"):
        png_path.unlink()


def generate_split(
    split_name: str,
    output_dir: Path,
    images_per_class_state: int,
    freq_bins: int,
    time_bins: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    sample_index = 0

    for class_name, generator in CLASS_GENERATORS.items():
        for rfi_present in (False, True):
            state = "rfi" if rfi_present else "clean"
            counts[f"{class_name}_{state}"] = images_per_class_state
            print(f"Generating {split_name} {class_name} {state}: {images_per_class_state}", flush=True)

            for _ in range(images_per_class_state):
                spectrogram = generator(freq_bins=freq_bins, time_bins=time_bins)
                if rfi_present:
                    spectrogram = add_rfi_overlay(spectrogram)

                filename = f"{split_name}_{sample_index:04d}_{class_name}_{state}.png"
                save_spectrogram(spectrogram, output_dir / filename)
                sample_index += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean and RFI-overlaid spectrogram data.")
    parser.add_argument("--output-root", default="data", help="Dataset output root.")
    parser.add_argument("--train-per-class-state", type=int, default=50, help="Train images per class per RFI state.")
    parser.add_argument("--val-per-class-state", type=int, default=25, help="Validation images per class per RFI state.")
    parser.add_argument("--test-per-class-state", type=int, default=25, help="Test images per class per RFI state.")
    parser.add_argument("--freq-bins", type=int, default=128, help="Generated spectrogram frequency bins.")
    parser.add_argument("--time-bins", type=int, default=256, help="Generated spectrogram time bins.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Move existing train/val/test PNGs into data/archive_before_rfi_overlay_generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    test_dir = output_root / "test"

    if args.archive_existing:
        archive_dir = output_root / "archive_before_rfi_overlay_generation"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for split_dir in (train_dir, val_dir, test_dir):
            if split_dir.exists():
                destination = archive_dir / split_dir.name
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(split_dir, destination)

    reset_split_dir(train_dir, PROJECT_ROOT)
    reset_split_dir(val_dir, PROJECT_ROOT)
    reset_split_dir(test_dir, PROJECT_ROOT)

    train_counts = generate_split("train", train_dir, args.train_per_class_state, args.freq_bins, args.time_bins)
    val_counts = generate_split("val", val_dir, args.val_per_class_state, args.freq_bins, args.time_bins)
    test_counts = generate_split("test", test_dir, args.test_per_class_state, args.freq_bins, args.time_bins)

    print(f"Classes: {', '.join(CLASS_GENERATORS)}")
    print(f"RFI states: clean, rfi")
    print(f"Train images: {sum(train_counts.values())}")
    print(f"Validation images: {sum(val_counts.values())}")
    print(f"Test images: {sum(test_counts.values())}")
    print(f"Total images: {sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())}")
    print(f"Train counts: {train_counts}")
    print(f"Validation counts: {val_counts}")
    print(f"Test counts: {test_counts}")


if __name__ == "__main__":
    main()
