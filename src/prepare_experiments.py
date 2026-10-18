"""Experiment dataset preparation module for Low-light Pedestrian Detection.

Prepares reproducible dataset configurations for the 3 training experiments:
1. Exp-A (Standard Fine-Tuning): Uses original low-light training split.
2. Exp-B (Proposed L-channel CLAHE): Training images enhanced with L-channel CLAHE.
3. Exp-C (Gamma Correction): Training images enhanced with non-linear Gamma (gamma=0.6).

CRITICAL LEAKAGE PREVENTION RULE:
Prapemrosesan CLAHE dan Gamma HANYA diterapkan pada split training.
Split validation dan test TETAP dibiarkan 100% murni tanpa augmentasi
agar evaluasi mencerminkan kemampuan model menghadapi citra malam riil di lapangan.
"""

import argparse
from pathlib import Path
import shutil
from typing import Dict, List
import cv2
from tqdm import tqdm
from preprocessing import apply_clahe_lab, apply_gamma_correction


DATASET_ORIGINAL: Path = Path("dataset")


def build_augmented_split(
    source_dir: Path,
    target_dir: Path,
    method: str,
) -> None:
    """Transform training images and preserve labels and validation/test splits.

    Args:
        source_dir: Root path of original dataset containing train/valid/test.
        target_dir: Root destination path for the augmented dataset variant.
        method: Transformation method ('clahe' or 'gamma').
    """
    if target_dir.exists():
        print(f"[INFO] Target directory '{target_dir}' already exists. Skipping rebuild.")
        return

    print(f"\n[INFO] Building '{method.upper()}' dataset variant at: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Train Split
    train_src_img = source_dir / "train" / "images"
    train_src_lbl = source_dir / "train" / "labels"
    train_dst_img = target_dir / "train" / "images"
    train_dst_lbl = target_dir / "train" / "labels"

    train_dst_img.mkdir(parents=True, exist_ok=True)
    train_dst_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in train_src_img.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"[INFO] Transforming {len(img_files)} training images with {method.upper()}...")

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if method == "clahe":
            transformed = apply_clahe_lab(img)
        elif method == "gamma":
            transformed = apply_gamma_correction(img, gamma=0.6)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

        out_img_path = train_dst_img / img_path.name
        cv2.imwrite(str(out_img_path), transformed)

    # Copy train labels intact
    print("[INFO] Copying training label annotations...")
    for lbl_path in train_src_lbl.glob("*.txt"):
        shutil.copy2(lbl_path, train_dst_lbl / lbl_path.name)

    # 2. Copy Valid and Test Splits 100% UNTOUCHED (Anti-Leakage)
    for eval_split in ("valid", "test"):
        eval_src = source_dir / eval_split
        eval_dst = target_dir / eval_split
        if eval_src.exists():
            print(f"[INFO] Copying untouched '{eval_split}' split (Preventing data leakage)...")
            shutil.copytree(eval_src, eval_dst, dirs_exist_ok=True)

    # 3. Create Corresponding data.yaml
    yaml_content = f"""path: {target_dir.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['Vehicle', 'person']
"""
    yaml_file = target_dir / f"data_{method}.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[INFO] Created config: {yaml_file}")


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(description="Prepare CLAHE and Gamma dataset splits.")
    parser.add_argument(
        "--variant",
        type=str,
        default="both",
        choices=["clahe", "gamma", "both"],
        help="Dataset variant to generate.",
    )
    args = parser.parse_args()

    if args.variant in ("clahe", "both"):
        build_augmented_split(DATASET_ORIGINAL, Path("dataset_clahe"), method="clahe")
    if args.variant in ("gamma", "both"):
        build_augmented_split(DATASET_ORIGINAL, Path("dataset_gamma"), method="gamma")

    print("\n[INFO] Dataset preparation complete.")


if __name__ == "__main__":
    main()
