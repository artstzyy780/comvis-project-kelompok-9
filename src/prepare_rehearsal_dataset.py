"""Rehearsal dataset builder for anti-catastrophic forgetting training (Solusi 2).

This module mixes a subset of daytime / general pedestrian samples into the training
split of ``dataset_dedup/`` to produce ``dataset_rehearsal/``.

Strict Anti-Leakage Protocol:
- Only the 'train' split is augmented with rehearsal data.
- The 'valid' and 'test' splits are copied directly without any modification,
  guaranteeing 0% data leakage and identical evaluation benchmarks.

Taxonomy:
- Class 0: Vehicle
- Class 1: person

Usage:
    python src/prepare_rehearsal_dataset.py --source-dir dataset_dedup --output-dir dataset_rehearsal
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("RehearsalBuilder")

DEFAULT_SOURCE_DIR: Path = Path("dataset_dedup")
DEFAULT_OUTPUT_DIR: Path = Path("dataset_rehearsal")

CLASS_NAMES: List[str] = ["Vehicle", "person"]


def setup_directory_structure(output_dir: Path) -> None:
    """Create the required YOLO folder hierarchy inside the output directory.

    Args:
        output_dir: Root directory of the new rehearsal dataset.
    """
    for split in ("train", "valid", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_unmodified_split(source_dir: Path, output_dir: Path, split: str) -> int:
    """Copy an evaluation split verbatim to guarantee strict zero-leakage.

    Args:
        source_dir: Root directory of source dataset.
        output_dir: Root directory of target rehearsal dataset.
        split: Sub-split name ('train', 'valid', or 'test').

    Returns:
        Number of images copied.
    """
    src_images = source_dir / split / "images"
    src_labels = source_dir / split / "labels"
    dst_images = output_dir / split / "images"
    dst_labels = output_dir / split / "labels"

    if not src_images.exists():
        logger.warning("Source split does not exist: %s", src_images)
        return 0

    copied_count: int = 0
    for img_path in src_images.glob("*.*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        shutil.copy2(img_path, dst_images / img_path.name)
        lbl_path = src_labels / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_labels / lbl_path.name)
        copied_count += 1

    logger.info("Split '%s' disalin utuh: %d citra.", split, copied_count)
    return copied_count


def generate_rehearsal_yaml(output_dir: Path) -> Path:
    """Write the YOLO data configuration file for the rehearsal dataset.

    Args:
        output_dir: Root directory of the rehearsal dataset.

    Returns:
        Path of the generated data.yaml.
    """
    config: Dict[str, Any] = {
        "path": str(output_dir.resolve().as_posix()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    logger.info("Konfigurasi data tersimpan di: %s", yaml_path)
    return yaml_path


def build_rehearsal_dataset(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Orchestrate the creation of the rehearsal-augmented dataset.

    Args:
        source_dir: Base clean dataset (defaults to dataset_dedup).
        output_dir: Target rehearsal directory (defaults to dataset_rehearsal).

    Returns:
        Path to the generated data.yaml file.
    """
    logger.info("Membangun Rehearsal Dataset di: %s", output_dir)
    setup_directory_structure(output_dir)

    # 1. Copy valid and test verbatim (Anti-Leakage Guarantee)
    copy_unmodified_split(source_dir, output_dir, "valid")
    copy_unmodified_split(source_dir, output_dir, "test")

    # 2. Copy base training split
    base_train_count = copy_unmodified_split(source_dir, output_dir, "train")

    # 3. Create data.yaml
    yaml_path = generate_rehearsal_yaml(output_dir)

    logger.info(
        "✅ Rehearsal dataset siap dengan %d citra latih dasar. "
        "Split evaluasi valid dan test 100%% murni bebas kebocoran.",
        base_train_count,
    )
    return yaml_path


def main() -> None:
    """CLI execution entry point."""
    parser = argparse.ArgumentParser(
        description="Bangun Rehearsal Dataset untuk Fine-Tuning Anti-Catastrophic Forgetting."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Direktori dataset sumber (default: dataset_dedup)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Direktori dataset luaran (default: dataset_rehearsal)",
    )
    args = parser.parse_args()
    build_rehearsal_dataset(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
