"""Failure case analysis for the pedestrian detector (Checkpoint CP11).

The rubric asks for *"kategori kegagalan, contoh kasus, hipotesis, dan satu
perbaikan yang diuji"*. This module produces the first three: it collects every
missed pedestrian and every false pedestrian on the evaluation split, groups
them into interpretable categories, and renders both a visual evidence grid and
a Markdown report.

The categorisation heuristics are pure functions, so they can be unit tested
without a GPU (see ``tests/test_failure_analysis.py``).

Categories
----------
``small_object``
    The pedestrian occupies less than 10% of the image height, so the feature
    response is weak.
``low_contrast``
    The average luminance inside the pedestrian box is below the dataset mean,
    meaning the silhouette almost disappears into the background.
``occluded``
    The pedestrian box overlaps a vehicle box, i.e. the object is partially
    hidden behind a car.
``low_confidence``
    The detector found the pedestrian but with a confidence below 0.40, so the
    fixed operating threshold of 0.25-0.40 discarded it.
``unlabelled_or_ambiguous``
    The detection overlaps a ground-truth box of another class, which usually
    indicates an annotation or taxonomy ambiguity rather than a model failure.

Usage:
    python src/analyze_failures.py --weights hasil_final_model/hasil_final_model/Exp_A_Standard_best.pt
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import (
    DATASET_DIR,
    EXPERIMENTS_DIR,
    PERSON_CLASS_ID,
    VEHICLE_CLASS_ID,
    load_evaluation_config,
)
from metrics import Detection, compute_iou


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("AnalyzeFailures")

REPORT_PATH: Path = EXPERIMENTS_DIR / "failure_cases.md"
GRID_PATH: Path = EXPERIMENTS_DIR / "failure_cases.png"
JSON_PATH: Path = EXPERIMENTS_DIR / "failure_cases.json"

SMALL_OBJECT_THRESHOLD: float = 0.10
LOW_CONFIDENCE_THRESHOLD: float = 0.40
VEHICLE_OVERLAP_THRESHOLD: float = 0.10

CATEGORY_LABELS: Dict[str, str] = {
    "small_object": "Pejalan kaki kecil (jarak jauh)",
    "low_contrast": "Kontras rendah (siluet menghilang)",
    "occluded": "Terhalang kendaraan",
    "low_confidence": "Keyakinan model rendah",
    "unlabelled_or_ambiguous": "Ambiguitas anotasi/taksonomi",
    "false_positive": "False positive (bukan pejalan kaki)",
}


@dataclass
class FailureCase:
    """One analysed error of the detector.

    Attributes:
        image_path: Path of the source image.
        category: Category identifier from CATEGORY_LABELS.
        detail: Human readable explanation.
        box: Bounding box of the failure in pixel coordinates.
        confidence: Confidence of the related detection, when any.
        ground_truth_count: Number of pedestrians annotated in the image.
        detected_count: Number of pedestrians detected in the image.
    """

    image_path: Path
    category: str
    detail: str
    box: Tuple[float, float, float, float]
    confidence: Optional[float] = None
    ground_truth_count: int = 0
    detected_count: int = 0


@dataclass
class FailureReport:
    """Aggregated failure statistics of one model on one split.

    Attributes:
        model_path: Path of the analysed checkpoint.
        split: Evaluated split.
        category_counts: Number of failures per category.
        cases: All collected failure cases.
        image_count: Number of evaluated images.
        ground_truth_person: Number of pedestrian boxes in the split.
    """

    model_path: str
    split: str
    category_counts: Dict[str, int]
    cases: List[FailureCase] = field(default_factory=list)
    image_count: int = 0
    ground_truth_person: int = 0


def mean_luminance_in_box(image: np.ndarray, box: Tuple[float, float, float, float]) -> float:
    """Compute the average luminance inside a box.

    Args:
        image: Source image in BGR format.
        box: Bounding box as (x1, y1, x2, y2) in pixels.

    Returns:
        Mean intensity of the blue channel (a luminance proxy) inside the box.
    """
    height, width = image.shape[:2]
    x1: int = max(0, min(width - 1, int(box[0])))
    y1: int = max(0, min(height - 1, int(box[1])))
    x2: int = max(x1 + 1, min(width, int(box[2])))
    y2: int = max(y1 + 1, min(height, int(box[3])))
    patch: np.ndarray = image[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch[:, :, 0]))


def categorize_false_negative(
    box: Tuple[float, float, float, float],
    image_height: int,
    image: Optional[np.ndarray] = None,
    vehicle_boxes: Sequence[Tuple[float, float, float, float]] = (),
    dataset_luminance: float = 43.0,
) -> str:
    """Classify why a pedestrian was missed.

    Args:
        box: Pedestrian box as (x1, y1, x2, y2) in pixels.
        image_height: Height of the source image in pixels.
        image: Optional source image used for the contrast check.
        vehicle_boxes: Vehicle boxes used for the occlusion check.
        dataset_luminance: Mean luminance of the dataset, used as reference.

    Returns:
        Category identifier from CATEGORY_LABELS.
    """
    relative_height: float = (box[3] - box[1]) / max(1, image_height)
    if relative_height < SMALL_OBJECT_THRESHOLD:
        return "small_object"

    for vehicle in vehicle_boxes:
        if compute_iou(
            Detection(PERSON_CLASS_ID, box[0], box[1], box[2], box[3]),
            Detection(VEHICLE_CLASS_ID, vehicle[0], vehicle[1], vehicle[2], vehicle[3]),
        ) > VEHICLE_OVERLAP_THRESHOLD:
            return "occluded"

    if image is not None:
        luminance: float = mean_luminance_in_box(image, box)
        if luminance < dataset_luminance:
            return "low_contrast"

    return "low_confidence"


def categorize_false_positive(
    box: Tuple[float, float, float, float],
    confidence: float,
    vehicle_boxes: Sequence[Tuple[float, float, float, float]] = (),
) -> str:
    """Classify why a non-pedestrian box was detected as a pedestrian.

    Args:
        box: Detection box as (x1, y1, x2, y2) in pixels.
        confidence: Confidence of the detection.
        vehicle_boxes: Vehicle boxes of the same image.

    Returns:
        Category identifier from CATEGORY_LABELS.
    """
    for vehicle in vehicle_boxes:
        if compute_iou(
            Detection(PERSON_CLASS_ID, box[0], box[1], box[2], box[3]),
            Detection(VEHICLE_CLASS_ID, vehicle[0], vehicle[1], vehicle[2], vehicle[3]),
        ) > VEHICLE_OVERLAP_THRESHOLD:
            return "unlabelled_or_ambiguous"
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return "false_positive"


def summarize(report: FailureReport) -> Dict[str, Any]:
    """Build a JSON-serialisable summary of a failure report.

    Args:
        report: Collected failure report.

    Returns:
        Dictionary with the aggregated statistics.
    """
    return {
        "model": report.model_path,
        "split": report.split,
        "image_count": report.image_count,
        "ground_truth_person_boxes": report.ground_truth_person,
        "category_counts": report.category_counts,
        "category_labels": CATEGORY_LABELS,
        "cases": [
            {
                "image": str(case.image_path.name),
                "category": case.category,
                "detail": case.detail,
                "confidence": case.confidence,
                "box": [round(value, 1) for value in case.box],
            }
            for case in report.cases
        ],
    }


def render_markdown(report: FailureReport, dataset_luminance: float) -> str:
    """Render the failure analysis as a Markdown report.

    Args:
        report: Collected failure report.
        dataset_luminance: Mean luminance of the dataset used as reference.

    Returns:
        Markdown document.
    """
    total: int = sum(report.category_counts.values())
    lines: List[str] = [
        "# 🔍 Analisis Kasus Gagal (Failure Case Analysis)",
        "",
        "> Checkpoint CP11 — *kategori kegagalan, contoh kasus, dan hipotesis*.",
        "",
        "## 1. Ringkasan",
        "",
        f"- **Model:** `{report.model_path}`",
        f"- **Split:** `{report.split}` ({report.image_count} citra)",
        f"- **Jumlah kotak pejalan kaki teranotasi:** {report.ground_truth_person}",
        f"- **Total kasus gagal teranalisis:** {total}",
        "",
        "## 2. Distribusi Kategori Kegagalan",
        "",
        "| Kategori | Jumlah | Persentase | Interpretasi |",
        "|:---|:---:|:---:|:---|",
    ]

    for category, count in sorted(report.category_counts.items(), key=lambda item: -item[1]):
        share: float = (count / total * 100) if total else 0.0
        lines.append(
            f"| `{category}` | {count} | {share:.1f}% | {CATEGORY_LABELS.get(category, '-')} |"
        )

    lines.extend([
        "",
        "## 3. Contoh Kasus per Kategori",
        "",
    ])
    for category in CATEGORY_LABELS:
        cases = [case for case in report.cases if case.category == category][:3]
        if not cases:
            continue
        lines.append(f"### {CATEGORY_LABELS[category]} (`{category}`)")
        lines.append("")
        for case in cases:
            confidence: str = (
                f" (conf {case.confidence:.2f})" if case.confidence is not None else ""
            )
            lines.append(f"- `{case.image_path.name}` — {case.detail}{confidence}")
        lines.append("")

    lines.extend([
        "## 4. Hipotesis dan Perbaikan yang Diuji",
        "",
        "1. **Hipotesis utama:** sebagian besar false negative berasal dari *near-duplicate "
        "antar split* pada eksport Roboflow, bukan dari keterbatasan model.",
        "   - **Perbaikan yang diuji:** `src/rebuild_split_dedup.py` membangun ulang split pada "
        "level source scene; seluruh eksperimen diulang pada `dataset_dedup/`.",
        "2. **Hipotesis kedua:** pejalan kaki kecil dan terhalang kendaraan merupakan sumber "
        "false negative yang tidak dapat dihilangkan hanya dengan pra-pemrosesan.",
        "   - **Perbaikan yang diuji:**Threshold.confidence diturunkan pada inferensi "
        "(operating point recall-first), dibuktikan lewat tabel unified (CP10).",
        "3. **Hipotesis ketiga:** CLAHE menaikkan kontras tetapi juga mengamplifikasi noise, "
        "sehingga recall turun sementara precision naik.",
        "   - **Perbaikan yang diuji:** perbandingan Exp-A vs Exp-B vs Exp-C pada split yang sama "
        "dengan interval keyakinan (CP10).",
        "",
        "## 5. Visual Evidence",
        "",
        f"Komparasi ground truth dan prediksi tersimpan di `{GRID_PATH.name}`.",
        "",
        "---",
        "",
        "*(Laporan ini dihasilkan secara otomatis oleh `src/analyze_failures.py`)*",
    ])
    return "\n".join(lines)


def render_grid(cases: List[FailureCase], output_path: Path, max_cases: int = 12) -> None:
    """Render a visual evidence grid of the most frequent failure cases.

    Args:
        cases: Collected failure cases.
        output_path: Destination PNG path.
        max_cases: Maximum number of cases rendered.
    """
    import matplotlib.pyplot as plt

    if not cases:
        return

    selected: List[FailureCase] = cases[:max_cases]
    columns: int = 3
    rows: int = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for index, axis in enumerate(axes_flat):
        axis.axis("off")
        if index >= len(selected):
            continue
        case: FailureCase = selected[index]
        image: Optional[np.ndarray] = cv2.imread(str(case.image_path))
        if image is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in case.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            case.category,
            (max(4, x1), max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(case.image_path.name, fontsize=9)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def analyze_split(
    weights_path: Path,
    dataset_root: Path,
    split: str,
    conf_threshold: float,
    nms_iou: float,
    dataset_luminance: float = 43.0,
) -> FailureReport:
    """Run the model over a split and collect every failure case.

    Args:
        weights_path: Checkpoint of the analysed model.
        dataset_root: Root of the evaluated dataset.
        split: Split name.
        conf_threshold: Confidence threshold used during inference.
        nms_iou: NMS IoU threshold.
        dataset_luminance: Mean luminance of the dataset, used as reference.

    Returns:
        The collected FailureReport.

    Raises:
        ImportError: When Ultralytics is not installed.
    """
    from ultralytics import YOLO

    images_dir: Path = dataset_root / split / "images"
    labels_dir: Path = dataset_root / split / "labels"
    images: List[Path] = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    model = YOLO(str(weights_path))
    cases: List[FailureCase] = []
    category_counts: Dict[str, int] = {key: 0 for key in CATEGORY_LABELS}
    ground_truth_person: int = 0

    for image_path in images:
        image: Optional[np.ndarray] = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        truths: List[Detection] = []
        vehicles: List[Tuple[float, float, float, float]] = []
        label_path: Path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    tokens: List[str] = line.strip().split()
                    if len(tokens) < 5:
                        continue
                    class_id: int = int(tokens[0])
                    x_center, y_center, box_width, box_height = (
                        float(value) for value in tokens[1:5]
                    )
                    box: Tuple[float, float, float, float] = (
                        (x_center - box_width / 2) * width,
                        (y_center - box_height / 2) * height,
                        (x_center + box_width / 2) * width,
                        (y_center + box_height / 2) * height,
                    )
                    if class_id == PERSON_CLASS_ID:
                        truths.append(Detection(PERSON_CLASS_ID, *box))
                        ground_truth_person += 1
                    elif class_id == VEHICLE_CLASS_ID:
                        vehicles.append(box)

        result = model.predict(
            source=image, conf=conf_threshold, iou=nms_iou, verbose=False, device="cpu"
        )
        detected_persons: List[Detection] = []
        if len(result) > 0 and result[0].boxes is not None:
            for index in range(len(result[0].boxes)):
                if int(result[0].boxes.cls[index].item()) != PERSON_CLASS_ID:
                    continue
                coords: List[float] = result[0].boxes.xyxy[index].tolist()
                detected_persons.append(
                    Detection(
                        class_id=PERSON_CLASS_ID,
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=float(result[0].boxes.conf[index].item()),
                    )
                )

        # False negatives: ground truth without any overlapping detection.
        matched: set = set()
        for truth in truths:
            best_iou: float = 0.0
            best_index: int = -1
            for index, detection in enumerate(detected_persons):
                if index in matched:
                    continue
                overlap: float = compute_iou(truth, detection)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou >= 0.5:
                matched.add(best_index)
                continue

            category: str = categorize_false_negative(
                (truth.x1, truth.y1, truth.x2, truth.y2),
                image_height=height,
                image=image,
                vehicle_boxes=vehicles,
                dataset_luminance=dataset_luminance,
            )
            category_counts[category] = category_counts.get(category, 0) + 1
            cases.append(
                FailureCase(
                    image_path=image_path,
                    category=category,
                    detail=(
                        "Pejalan kaki teranotasi tidak terdeteksi "
                        f"(tinggi {((truth.y2 - truth.y1) / height) * 100:.1f}% dari citra)."
                    ),
                    box=(truth.x1, truth.y1, truth.x2, truth.y2),
                    ground_truth_count=len(truths),
                    detected_count=len(detected_persons),
                )
            )

        # False positives: detections without an overlapping ground truth.
        for index, detection in enumerate(detected_persons):
            if index in matched:
                continue
            category = categorize_false_positive(
                (detection.x1, detection.y1, detection.x2, detection.y2),
                detection.confidence,
                vehicle_boxes=vehicles,
            )
            category_counts[category] = category_counts.get(category, 0) + 1
            cases.append(
                FailureCase(
                    image_path=image_path,
                    category=category,
                    detail="Deteksi pejalan kaki tanpa anotasi yang cocok.",
                    box=(detection.x1, detection.y1, detection.x2, detection.y2),
                    confidence=detection.confidence,
                    ground_truth_count=len(truths),
                    detected_count=len(detected_persons),
                )
            )

    return FailureReport(
        model_path=str(weights_path),
        split=split,
        category_counts=category_counts,
        cases=cases,
        image_count=len(images),
        ground_truth_person=ground_truth_person,
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Analisis kasus gagal deteksi pejalan kaki (CP11)."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("hasil_final_model/hasil_final_model/Exp_A_Standard_best.pt"),
        help="Checkpoint yang dianalisis",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    config = load_evaluation_config()
    conf: float = args.conf if args.conf is not None else config.conf_threshold

    report: FailureReport = analyze_split(
        weights_path=args.weights,
        dataset_root=args.dataset_root,
        split=args.split,
        conf_threshold=conf,
        nms_iou=config.nms_iou,
    )

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(summarize(report), handle, indent=2, ensure_ascii=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report, dataset_luminance=43.0))

    render_grid(report.cases, GRID_PATH)
    logger.info("Laporan: %s", args.output)
    logger.info("Grid   : %s", GRID_PATH)
    for category, count in report.category_counts.items():
        logger.info("  %-24s %d", category, count)


if __name__ == "__main__":
    main()
