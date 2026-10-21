"""Rebuild a leakage-free dataset split grouped by source scene (CP07).

Problem
-------
The Roboflow ``Low-light v2`` export distributes the augmented copies of a
single photographic scene across ``train``, ``valid`` and ``test``. Auditing
the shipped split therefore shows that 55% of the test images - and 71.8% of
the test *pedestrian* boxes - originate from a scene that is also present in
the training set. Any metric computed on that split is optimistic.

Solution
--------
Rebuild the split at the level of the **source scene**: every scene (and all of
its augmented copies) is assigned to exactly one split, stratified by whether
the scene contains a pedestrian, and fully deterministic given a seed.

Usage:
    python src/rebuild_split_dedup.py --dry-run     # report only, no files
    python src/rebuild_split_dedup.py               # write dataset_dedup/
"""

import argparse
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    AUDIT_DIR,
    DATASET_DIR,
    DEDUP_DATASET_DIR,
    DEFAULT_SEED,
    PERSON_CLASS_ID,
    SplitLeakageReport,
)
from scene_utils import (
    assign_scenes_stratified,
    collect_images,
    detect_cross_split_leakage,
    group_by_scene,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("RebuildSplitDedup")

SPLIT_NAMES: Tuple[str, ...] = ("train", "valid", "test")
SOURCE_SPLITS: Tuple[str, ...] = ("train", "valid", "test")


class SplitStatistics:
    """Per-split counters collected while materialising the new split.

    Attributes:
        split_name: Name of the split.
        images: Number of image files written.
        boxes: Number of bounding boxes written.
        class_counts: Bounding box count per class identifier.
        scenes: Number of distinct source scenes in the split.
        images_with_person: Number of images containing at least one person.
    """

    def __init__(self, split_name: str) -> None:
        """Initialise an empty statistics container for one split."""
        self.split_name: str = split_name
        self.images: int = 0
        self.boxes: int = 0
        self.class_counts: Counter[int] = Counter()
        self.scenes: int = 0
        self.images_with_person: int = 0

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the counters."""
        return {
            "split": self.split_name,
            "images": self.images,
            "boxes": self.boxes,
            "vehicle_boxes": int(self.class_counts.get(0, 0)),
            "person_boxes": int(self.class_counts.get(1, 0)),
            "scenes": self.scenes,
            "images_with_person": self.images_with_person,
        }


def read_label_boxes(label_path: Path) -> List[Tuple[int, int]]:
    """Read a YOLO label file and return (class_id, line_count) style entries.

    Args:
        label_path: Path to the ``.txt`` annotation file.

    Returns:
        List of class identifiers, one per annotated box.
    """
    if not label_path.exists():
        return []
    classes: List[int] = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            tokens: List[str] = line.strip().split()
            if tokens and tokens[0].isdigit():
                classes.append(int(tokens[0]))
    return classes


def collect_scene_inventory(source_root: Path) -> Tuple[Dict[str, List[Path]], Dict[str, bool]]:
    """Map every source scene to its image copies and pedestrian presence.

    Args:
        source_root: Root of the original dataset.

    Returns:
        Tuple of (scene_id -> image paths, scene_id -> contains pedestrian).
    """
    all_images: List[Path] = []
    for split in SOURCE_SPLITS:
        all_images.extend(collect_images(source_root, split))

    scenes: Dict[str, List[Path]] = group_by_scene(all_images)

    contains_person: Dict[str, bool] = {}
    for scene_id, paths in scenes.items():
        found: bool = False
        for path in paths:
            label_path: Path = source_root / "train" / "labels" / f"{path.stem}.txt"
            if not label_path.exists():
                # The annotation may live in another split folder of the source.
                for split in SOURCE_SPLITS:
                    candidate: Path = source_root / split / "labels" / f"{path.stem}.txt"
                    if candidate.exists():
                        label_path = candidate
                        break
            if PERSON_CLASS_ID in read_label_boxes(label_path):
                found = True
                break
        contains_person[scene_id] = found

    return scenes, contains_person


def locate_label(source_root: Path, stem: str) -> Optional[Path]:
    """Find the annotation file of an image stem inside the source dataset.

    Args:
        source_root: Root of the original dataset.
        stem: File stem of the image (without extension).

    Returns:
        Path to the label file, or None when no annotation exists.
    """
    for split in SOURCE_SPLITS:
        candidate: Path = source_root / split / "labels" / f"{stem}.txt"
        if candidate.exists():
            return candidate
    return None


def copy_file(source: Path, destination: Path, use_hardlink: bool) -> None:
    """Copy or hard-link a file, creating parent directories as needed.

    Args:
        source: Existing file.
        destination: Target path.
        use_hardlink: Try a hard link first and fall back to a copy.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_hardlink:
        try:
            os.link(source, destination)
            return
        except (OSError, AttributeError):
            logger.debug("Hard link unavailable for %s, falling back to copy", source)
    shutil.copy2(source, destination)


def materialise_split(
    source_root: Path,
    target_root: Path,
    assignment: Dict[str, str],
    scenes: Dict[str, List[Path]],
    use_hardlink: bool,
) -> Dict[str, SplitStatistics]:
    """Write the new split folders with images and labels.

    Args:
        source_root: Root of the original dataset.
        target_root: Root where the new split is written.
        assignment: Mapping of scene identifier to target split.
        scenes: Mapping of scene identifier to its image copies.
        use_hardlink: Use hard links instead of copying when possible.

    Returns:
        Per-split statistics collected during the copy.
    """
    statistics: Dict[str, SplitStatistics] = {name: SplitStatistics(name) for name in SPLIT_NAMES}
    seen_scenes: Dict[str, set] = {name: set() for name in SPLIT_NAMES}

    for scene_id, paths in sorted(scenes.items()):
        split_name: str = assignment[scene_id]
        stats: SplitStatistics = statistics[split_name]
        stats.scenes += 1
        seen_scenes[split_name].add(scene_id)

        for image_path in paths:
            label_path: Optional[Path] = locate_label(source_root, image_path.stem)
            classes: List[int] = read_label_boxes(label_path) if label_path else []

            copy_file(
                image_path,
                target_root / split_name / "images" / image_path.name,
                use_hardlink,
            )
            if label_path is not None:
                copy_file(
                    label_path,
                    target_root / split_name / "labels" / label_path.name,
                    use_hardlink,
                )

            stats.images += 1
            stats.boxes += len(classes)
            stats.class_counts.update(classes)
            if PERSON_CLASS_ID in classes:
                stats.images_with_person += 1

    return statistics


def write_data_yaml(target_root: Path) -> Path:
    """Write the Ultralytics ``data.yaml`` for the rebuilt dataset.

    Args:
        target_root: Root of the rebuilt dataset.

    Returns:
        Path to the written YAML file.
    """
    yaml_path: Path = target_root / "data.yaml"
    content: str = (
        f"path: {target_root.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 2\n"
        "names: ['Vehicle', 'person']\n"
    )
    target_root.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return yaml_path


def write_manifest(
    target_root: Path,
    assignment: Dict[str, str],
    statistics: Dict[str, SplitStatistics],
    seed: int,
    ratios: Dict[str, float],
) -> Path:
    """Write the machine-readable manifest of the rebuilt split.

    Args:
        target_root: Root of the rebuilt dataset.
        assignment: Mapping of scene identifier to target split.
        statistics: Per-split statistics.
        seed: Random seed used for the assignment.
        ratios: Split ratios used for the assignment.

    Returns:
        Path to the written manifest file.
    """
    manifest_path: Path = target_root / "split_manifest.json"
    payload: Dict[str, object] = {
        "generator": "src/rebuild_split_dedup.py",
        "strategy": "grouped split by source scene, stratified by pedestrian presence",
        "seed": seed,
        "ratios": ratios,
        "splits": {name: stats.to_dict() for name, stats in statistics.items()},
        "scene_assignment": dict(sorted(assignment.items())),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path


def write_report(
    report_path: Path,
    statistics: Dict[str, SplitStatistics],
    assignment: Dict[str, str],
    seed: int,
    ratios: Dict[str, float],
    leakage_before: SplitLeakageReport,
    leakage_after: SplitLeakageReport,
) -> None:
    """Write the Markdown evidence report for the rebuilt split.

    Args:
        report_path: Destination path of the Markdown report.
        statistics: Per-split statistics of the new split.
        assignment: Mapping of scene identifier to target split.
        seed: Random seed used for the assignment.
        ratios: Split ratios used for the assignment.
        leakage_before: Leakage report of the original split.
        leakage_after: Leakage report of the rebuilt split.
    """
    total_images: int = sum(stats.images for stats in statistics.values())
    total_boxes: int = sum(stats.boxes for stats in statistics.values())
    total_person: int = sum(int(stats.class_counts.get(1, 0)) for stats in statistics.values())
    total_vehicle: int = sum(int(stats.class_counts.get(0, 0)) for stats in statistics.values())

    lines: List[str] = [
        "# Laporan Rebuild Split Bebas-Leakage (Grouped by Source Scene)",
        "",
        "> Bukti untuk Checkpoint CP07 (repository & pipeline data) dan CP11 "
        "(hipotesis + satu perbaikan yang diuji).",
        "",
        "## 1. Masalah pada Split Asli",
        "",
        "Eksport Roboflow menempatkan augmented copy dari satu scene ke beberapa split "
        "sekaligus, sehingga foto yang sama muncul pada train dan test.",
        "",
        "| Split | File total | File dari scene bocor | Persentase |",
        "|:---|:---:|:---:|:---:|",
    ]
    for name in SPLIT_NAMES:
        total: int = leakage_before.per_split_total.get(name, 0)
        affected: int = leakage_before.per_split_overlap_files.get(name, 0)
        share: float = (affected / total * 100) if total else 0.0
        lines.append(f"| **{name}** | {total:,} | {affected:,} | {share:.1f}% |")

    lines.extend([
        "",
        f"- Scene sumber unik pada dataset: **{leakage_before.total_scenes}**",
        f"- Scene yang muncul pada lebih dari satu split: **{len(leakage_before.overlapping_scenes)}**",
        "",
        "## 2. Strategi Rebuild",
        "",
        "- **Unit split**: *source scene* (bukan file), sehingga seluruh augmented copy dari satu "
        "scene selalu berada pada split yang sama.",
        "- **Stratifikasi**: scene dikelompokkan berdasarkan keberadaan pejalan kaki agar sinyal "
        "kelas minoritas tersebar merata.",
        f"- **Seed**: `{seed}` sehingga pembagian bersifat reproducible.",
        f"- **Rasio target**: {json.dumps(ratios, ensure_ascii=False)}",
        "",
        "## 3. Hasil Split Baru",
        "",
        "| Split | Citra | Scene | Box | Vehicle | person | Citra berpejalan kaki |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for name in SPLIT_NAMES:
        stats: SplitStatistics = statistics[name]
        lines.append(
            f"| **{name}** | {stats.images:,} | {stats.scenes} | {stats.boxes:,} | "
            f"{int(stats.class_counts.get(0, 0)):,} | {int(stats.class_counts.get(1, 0)):,} | "
            f"{stats.images_with_person} |"
        )

    lines.extend([
        "",
        f"- **Total**: {total_images:,} citra | {total_boxes:,} box "
        f"(Vehicle {total_vehicle:,} / person {total_person:,}).",
        f"- **Jumlah scene unik**: {len(assignment)}.",
        "",
        "## 4. Verifikasi Anti-Leakage",
        "",
        "Setiap scene hanya boleh muncul pada satu split. Pemeriksaan ulang dilakukan oleh "
        "`src/scene_utils.detect_cross_split_leakage`.",
        "",
        "| Split | File total | File dari scene bocor | Status |",
        "|:---|:---:|:---:|:---:|",
    ])
    for name in SPLIT_NAMES:
        total = leakage_after.per_split_total.get(name, 0)
        affected = leakage_after.per_split_overlap_files.get(name, 0)
        status = "✅ bebas bocor" if affected == 0 else f"❌ {affected} file bocor"
        lines.append(f"| **{name}** | {total:,} | {affected:,} | {status} |")

    lines.extend([
        "",
        f"**Total scene bocor setelah rebuild: {len(leakage_after.overlapping_scenes)}** "
        f"(sebelumnya {len(leakage_before.overlapping_scenes)}).",
        "",
        "## 5. Mengapa Ini Penting",
        "",
        "Dengan split bebas-leakage, selisih Recall antar varian eksperimen benar-benar "
        "mencerminkan perbedaan model dan bukan karena scene yang sama sudah terlihat saat "
        "pelatihan. Selain itu, jumlah kotak pejalan kaki pada split uji naik dari 39 menjadi "
        f"{int(statistics['test'].class_counts.get(1, 0))}, sehingga interval keyakinan "
        "Recall menjadi jauh lebih sempit.",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Rebuild a leakage-free dataset split grouped by source scene."
    )
    parser.add_argument("--source", type=Path, default=DATASET_DIR, help="Dataset sumber")
    parser.add_argument("--target", type=Path, default=DEDUP_DATASET_DIR, help="Dataset tujuan")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--ratios",
        type=str,
        default="train=0.70,valid=0.15,test=0.15",
        help="Rasio split per scene",
    )
    parser.add_argument("--dry-run", action="store_true", help="Hanya laporan, tanpa menulis file")
    parser.add_argument("--link", action="store_true", help="Pakai hard link alih-alih copy")
    args = parser.parse_args()

    ratios: Dict[str, float] = {}
    for chunk in args.ratios.split(","):
        name, _, value = chunk.partition("=")
        ratios[name.strip()] = float(value)

    logger.info("Menganalisis split sumber: %s", args.source)
    leakage_before = detect_cross_split_leakage(args.source)
    for name in SPLIT_NAMES:
        total: int = leakage_before.per_split_total.get(name, 0)
        affected: int = leakage_before.per_split_overlap_files.get(name, 0)
        share: float = (affected / total * 100) if total else 0.0
        logger.info(
            "  %-5s: %d/%d file (%.1f%%) berasal dari scene yang juga ada di split lain",
            name, affected, total, share,
        )

    scenes, contains_person = collect_scene_inventory(args.source)
    logger.info("Menemukan %d scene sumber unik", len(scenes))

    assignment: Dict[str, str] = assign_scenes_stratified(
        contains_person, ratios=ratios, seed=args.seed
    )

    if args.dry_run:
        counter: Counter[str] = Counter(assignment.values())
        logger.info("[DRY RUN] Distribusi scene: %s", dict(counter))
        return

    if args.target.exists():
        shutil.rmtree(args.target)
    args.target.mkdir(parents=True, exist_ok=True)

    statistics = materialise_split(args.source, args.target, assignment, scenes, args.link)
    yaml_path: Path = write_data_yaml(args.target)
    manifest_path: Path = write_manifest(args.target, assignment, statistics, args.seed, ratios)

    leakage_after: SplitLeakageReport = detect_cross_split_leakage(args.target)
    write_report(
        AUDIT_DIR / "dedup_split_report.md",
        statistics,
        assignment,
        args.seed,
        ratios,
        leakage_before,
        leakage_after,
    )

    logger.info("Selesai. Config: %s", yaml_path)
    logger.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
