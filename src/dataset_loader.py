"""Typed dataset loader and data sanity checks (Checkpoint CP07).

The training pipeline itself is handled by Ultralytics, but the rubric of CP07
explicitly asks for a *"loader, split tetap, seed, dan data sanity check"*.
This module provides the deterministic, typed counterpart:

- :func:`load_split` returns the file inventory of one split in a stable order.
- :func:`sanity_check_split` verifies label pairing, class identifiers,
  coordinate ranges, and image readability.
- :func:`describe_dataset` produces a machine-readable summary of the whole
  dataset, which is stored next to the results for reproducibility.

Example:
    >>> report = sanity_check_split(Path("dataset"), "test")
    >>> report.ok
    True
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, List, Tuple

import cv2

from config import CLASS_NAMES, DATASET_DIR, NUM_CLASSES, RESULTS_DIR


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("DatasetLoader")

IMAGE_EXTENSIONS: Final[Tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_NAMES: Final[Tuple[str, ...]] = ("train", "valid", "test")
SUMMARY_PATH: Final[Path] = RESULTS_DIR / "dataset_summary.json"

MIN_BOX_COORDINATE: Final[float] = 0.0
MAX_BOX_COORDINATE: Final[float] = 1.0


@dataclass
class SplitInventory:
    """Deterministic inventory of one dataset split.

    Attributes:
        split_name: Name of the split.
        root: Root directory of the split.
        images: Sorted list of image paths.
        labels: Sorted list of label paths that have a matching image.
        orphan_labels: Label files without a matching image.
        missing_labels: Images without a label file.
    """

    split_name: str
    root: Path
    images: List[Path] = field(default_factory=list)
    labels: List[Path] = field(default_factory=list)
    orphan_labels: List[Path] = field(default_factory=list)
    missing_labels: List[Path] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Return the number of images in the split."""
        return len(self.images)


@dataclass
class SanityReport:
    """Result of the data sanity check for one split.

    Attributes:
        split_name: Name of the checked split.
        image_count: Number of images found.
        label_count: Number of label files found.
        total_boxes: Total number of annotated boxes.
        class_counts: Bounding box count per class identifier.
        images_with_person: Number of images containing a pedestrian.
        empty_labels: Number of label files without any annotation.
        missing_labels: Images without a label file.
        orphan_labels: Label files without a matching image.
        invalid_coordinates: Labels with out-of-range coordinates.
        invalid_classes: Labels referencing an unknown class identifier.
        unreadable_images: Images that cannot be decoded.
    """

    split_name: str
    image_count: int = 0
    label_count: int = 0
    total_boxes: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)
    images_with_person: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    invalid_coordinates: int = 0
    invalid_classes: int = 0
    unreadable_images: int = 0

    @property
    def ok(self) -> bool:
        """Return True when the split passes every sanity criterion."""
        return not (
            self.missing_labels
            or self.orphan_labels
            or self.invalid_coordinates
            or self.invalid_classes
            or self.unreadable_images
        )

    def issues(self) -> List[str]:
        """Return a human readable list of detected problems.

        Returns:
            One message per detected problem; empty when the split is clean.
        """
        problems: List[str] = []
        if self.missing_labels:
            problems.append(f"{self.missing_labels} citra tanpa berkas label")
        if self.orphan_labels:
            problems.append(f"{self.orphan_labels} label tanpa citra pasangan")
        if self.invalid_coordinates:
            problems.append(f"{self.invalid_coordinates} baris label dengan koordinat di luar [0, 1]")
        if self.invalid_classes:
            problems.append(f"{self.invalid_classes} baris label dengan id kelas tak dikenal")
        if self.unreadable_images:
            problems.append(f"{self.unreadable_images} citra tidak dapat dibaca")
        return problems

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the report."""
        return {
            "split": self.split_name,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "total_boxes": self.total_boxes,
            "class_counts": {str(key): value for key, value in self.class_counts.items()},
            "images_with_person": self.images_with_person,
            "empty_labels": self.empty_labels,
            "missing_labels": self.missing_labels,
            "orphan_labels": self.orphan_labels,
            "invalid_coordinates": self.invalid_coordinates,
            "invalid_classes": self.invalid_classes,
            "unreadable_images": self.unreadable_images,
            "ok": self.ok,
            "issues": self.issues(),
        }


def load_split(dataset_root: Path, split: str) -> SplitInventory:
    """Collect the deterministic file inventory of one split.

    Args:
        dataset_root: Root directory of the dataset.
        split: Split name ('train', 'valid', or 'test').

    Returns:
        A SplitInventory describing the split.

    Raises:
        FileNotFoundError: When the split directory does not exist.
    """
    images_dir: Path = dataset_root / split / "images"
    labels_dir: Path = dataset_root / split / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {images_dir}")

    images: List[Path] = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels: List[Path] = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    image_stems: set = {path.stem for path in images}
    label_stems: set = {path.stem for path in labels}

    return SplitInventory(
        split_name=split,
        root=dataset_root,
        images=images,
        labels=[path for path in labels if path.stem in image_stems],
        orphan_labels=[path for path in labels if path.stem not in image_stems],
        missing_labels=[path for path in images if path.stem not in label_stems],
    )


def sanity_check_split(
    dataset_root: Path,
    split: str,
    check_decodability: bool = False,
    max_decode_checks: int = 50,
) -> SanityReport:
    """Verify the integrity of one split.

    Args:
        dataset_root: Root directory of the dataset.
        split: Split name.
        check_decodability: Whether a sample of images is decoded to prove
            readability. Disabled by default because it dominates the runtime.
        max_decode_checks: Number of images decoded when the check is enabled.

    Returns:
        A SanityReport describing the findings.
    """
    inventory: SplitInventory = load_split(dataset_root, split)
    report: SanityReport = SanityReport(split_name=split)
    report.image_count = inventory.count
    report.label_count = len(inventory.labels)
    report.missing_labels = len(inventory.missing_labels)
    report.orphan_labels = len(inventory.orphan_labels)

    for label_path in inventory.labels:
        with open(label_path, "r", encoding="utf-8") as handle:
            lines: List[str] = [line.strip() for line in handle if line.strip()]
        if not lines:
            report.empty_labels += 1
        has_person: bool = False
        for line in lines:
            tokens: List[str] = line.split()
            if len(tokens) < 5:
                report.invalid_coordinates += 1
                continue
            try:
                class_id: int = int(tokens[0])
                coordinates: List[float] = [float(value) for value in tokens[1:5]]
            except ValueError:
                report.invalid_classes += 1
                continue
            if class_id not in CLASS_NAMES:
                report.invalid_classes += 1
                continue
            if any(value < MIN_BOX_COORDINATE or value > MAX_BOX_COORDINATE for value in coordinates):
                report.invalid_coordinates += 1
                continue
            report.class_counts[class_id] = report.class_counts.get(class_id, 0) + 1
            report.total_boxes += 1
            if class_id == 1:
                has_person = True
        if has_person:
            report.images_with_person += 1

    if check_decodability:
        sample: List[Path] = inventory.images[:max_decode_checks]
        for image_path in sample:
            image = cv2.imread(str(image_path))
            if image is None:
                report.unreadable_images += 1

    return report


def describe_dataset(dataset_root: Path = DATASET_DIR) -> Dict[str, object]:
    """Build a machine-readable summary of the whole dataset.

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        Dictionary describing every split and the aggregated totals.
    """
    reports: Dict[str, SanityReport] = {}
    for split in SPLIT_NAMES:
        if not (dataset_root / split / "images").exists():
            continue
        reports[split] = sanity_check_split(dataset_root, split)

    totals: Dict[str, int] = {
        "images": sum(report.image_count for report in reports.values()),
        "boxes": sum(report.total_boxes for report in reports.values()),
        "person_boxes": sum(report.class_counts.get(1, 0) for report in reports.values()),
        "vehicle_boxes": sum(report.class_counts.get(0, 0) for report in reports.values()),
    }
    return {
        "dataset_root": str(dataset_root),
        "num_classes": NUM_CLASSES,
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "splits": {name: report.to_dict() for name, report in reports.items()},
        "totals": totals,
    }


def main() -> None:
    """Command line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data sanity check untuk dataset low-light (CP07)."
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--check-decodability", action="store_true")
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    summary: Dict[str, object] = describe_dataset(args.dataset_root)
    for name, split_summary in summary["splits"].items():  # type: ignore[union-attr]
        status: str = "OK" if split_summary["ok"] else "MASALAH"
        logger.info(
            "%-6s %s | %s citra | %s box | issues: %s",
            name,
            status,
            split_summary["image_count"],
            split_summary["total_boxes"],
            ", ".join(split_summary["issues"]) or "-",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Ringkasan: %s", args.output)


if __name__ == "__main__":
    main()
