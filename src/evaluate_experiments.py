"""Unified, single-split evaluation of every trained variant (Checkpoint CP10).

The rubric of CP10 demands *"tabel hasil pada split sama, metrik sesuai tugas,
dan catatan perubahan satu per satu"*. This module therefore produces exactly
one table, computed on exactly one split, with per-class metrics (not macro
averages) and a statistical verdict for every controlled change.

Two modes are available:

1. ``--split test`` runs the models and computes the table (needs Ultralytics).
2. ``--from-json <file>`` only re-renders the report from a stored result file,
   which keeps the reporting logic testable without a GPU.

Usage:
    python src/evaluate_experiments.py --split test
    python src/evaluate_experiments.py --from-json results/experiments/unified_results.json
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import (
    CLASS_NAMES,
    DATASET_DIR,
    EXPERIMENTS_DIR,
    PERSON_CLASS_ID,
    RESULTS_DIR,
    load_evaluation_config,
)
from metrics import (
    Detection,
    evaluate_detections,
    macro_average,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("EvaluateExperiments")

COCO_TO_TARGET_MAPPING: Dict[int, int] = {
    0: 1,  # person -> person
    2: 0,  # car -> Vehicle
    3: 0,  # motorcycle -> Vehicle
    5: 0,  # bus -> Vehicle
    7: 0,  # truck -> Vehicle
}

RESULT_JSON_PATH: Path = EXPERIMENTS_DIR / "unified_results.json"
RESULT_MARKDOWN_PATH: Path = EXPERIMENTS_DIR / "unified_results.md"


# ----------------------------------------------------------------------------
# Model Discovery
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelEntry:
    """A model that participates in the unified evaluation.

    Attributes:
        model_id: Identifier used in the report.
        display_name: Human readable name.
        weights_path: Path of the checkpoint; None for a zero-shot model.
        is_zero_shot: Whether the model uses the stock COCO taxonomy.
        independent_variable: The single factor this variant changes.
    """

    model_id: str
    display_name: str
    weights_path: Optional[Path]
    is_zero_shot: bool
    independent_variable: str


def discover_models(results_root: Path = RESULTS_DIR) -> List[ModelEntry]:
    """Locate every checkpoint that must appear in the unified table.

    Newer runs (``results/experiments/runs/<id>/weights/best.pt``) take
    precedence over the checkpoints shipped in ``hasil_final_model``.

    Returns:
        List of ModelEntry ordered as B0, Exp-A, Exp-B, Exp-C.
    """
    definitions: List[Tuple[str, str, str, bool]] = [
        ("B0_ZeroShot", "B0: Zero-Shot Baseline (COCO Pretrained)", "yolov8n.pt", True),
        ("Exp_A_Standard", "Exp-A: Fine-Tuned Standard Low-Light", "Exp_A_Standard_best.pt", False),
        ("Exp_B_CLAHE", "Exp-B: Fine-Tuned + L-channel CLAHE", "Exp_B_CLAHE_best.pt", False),
        ("Exp_C_Gamma", "Exp-C: Fine-Tuned + Gamma Correction", "Exp_C_Gamma_best.pt", False),
    ]
    variables: Dict[str, str] = {
        "B0_ZeroShot": "Tanpa fine-tuning (acuan mutlak).",
        "Exp_A_Standard": "Fine-tuning domain (control group, tanpa augmentasi pencahayaan).",
        "Exp_B_CLAHE": "Tambahan: CLAHE L-channel pada data latih.",
        "Exp_C_Gamma": "Tambahan: Gamma Correction (gamma=0.6) pada data latih.",
    }

    entries: List[ModelEntry] = []
    for model_id, display_name, filename, is_zero_shot in definitions:
        weights: Optional[Path] = None
        candidates: List[Path] = [
            results_root / "experiments" / "runs" / model_id / "weights" / "best.pt",
            results_root / "hasil_final_model" / "hasil_final_model" / filename,
            results_root / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                weights = candidate
                break
        entries.append(
            ModelEntry(
                model_id=model_id,
                display_name=display_name,
                weights_path=weights,
                is_zero_shot=is_zero_shot,
                independent_variable=variables[model_id],
            )
        )
    return entries


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def parse_ground_truth(label_path: Path, width: int, height: int) -> List[Detection]:
    """Read a YOLO label file into absolute pixel boxes.

    Args:
        label_path: Path of the label file.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        List of Detection objects for the ground truth.
    """
    detections: List[Detection] = []
    if not label_path.exists():
        return detections
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            tokens: List[str] = line.strip().split()
            if len(tokens) < 5:
                continue
            class_id: int = int(tokens[0])
            x_center, y_center, box_width, box_height = (float(value) for value in tokens[1:5])
            detections.append(
                Detection(
                    class_id=class_id,
                    x1=(x_center - box_width / 2) * width,
                    y1=(y_center - box_height / 2) * height,
                    x2=(x_center + box_width / 2) * width,
                    y2=(y_center + box_height / 2) * height,
                    confidence=1.0,
                )
            )
    return detections


def run_model_on_split(
    model: Any,
    entry: ModelEntry,
    images: Sequence[Path],
    labels_dir: Path,
    conf_threshold: float,
    nms_iou: float,
) -> Tuple[List[List[Detection]], List[List[Detection]], float]:
    """Run one model over a split and collect ground truth plus predictions.

    Args:
        model: Loaded Ultralytics model.
        entry: Model entry (used to decide the class remapping).
        images: Image paths of the split.
        labels_dir: Directory holding the annotations of the split.
        conf_threshold: Confidence threshold for detections.
        nms_iou: IoU threshold used by the NMS of the detector.

    Returns:
        Tuple of (ground truth per image, predictions per image, mean latency ms).
    """
    truths: List[List[Detection]] = []
    predictions: List[List[Detection]] = []
    latencies: List[float] = []

    for image_path in images:
        image: Optional[np.ndarray] = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truths.append(parse_ground_truth(labels_dir / f"{image_path.stem}.txt", width, height))

        result = model.predict(
            source=image,
            conf=conf_threshold,
            iou=nms_iou,
            verbose=False,
            device="cpu",
        )
        detected: List[Detection] = []
        if len(result) > 0 and result[0].boxes is not None:
            for index in range(len(result[0].boxes)):
                raw_class: int = int(result[0].boxes.cls[index].item())
                confidence: float = float(result[0].boxes.conf[index].item())
                coords: List[float] = result[0].boxes.xyxy[index].tolist()
                if entry.is_zero_shot:
                    if raw_class not in COCO_TO_TARGET_MAPPING:
                        continue
                    target_class: int = COCO_TO_TARGET_MAPPING[raw_class]
                else:
                    if raw_class not in CLASS_NAMES:
                        continue
                    target_class = raw_class
                detected.append(
                    Detection(
                        class_id=target_class,
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=confidence,
                    )
                )
        predictions.append(detected)

    mean_latency: float = float(np.mean(latencies)) if latencies else 0.0
    return truths, predictions, mean_latency


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _ci_text(interval: Optional[Tuple[float, float]]) -> str:
    """Format a confidence interval for the Markdown table.

    Args:
        interval: Tuple of (lower, upper) in percent, or None.

    Returns:
        Formatted interval string, or an empty string when unavailable.
    """
    if not interval:
        return "-"
    return f"[{interval[0]:.1f}, {interval[1]:.1f}]"


def _verdict(
    baseline: Optional[Tuple[float, float]],
    candidate: Optional[Tuple[float, float]],
) -> str:
    """Compare two recalls using overlapping confidence intervals.

    Args:
        baseline: Confidence interval of the control group.
        candidate: Confidence interval of the variant.

    Returns:
        Human readable verdict string.
    """
    if not baseline or not candidate:
        return "tidak dapat dinilai"
    overlap: bool = not (candidate[1] < baseline[0] or candidate[0] > baseline[1])
    if overlap:
        return "tidak signifikan (CI beririsan)"
    if candidate[0] > baseline[1]:
        return "meninggal secara signifikan"
    return "menurun secara signifikan"


def render_report(payload: Dict[str, Any]) -> str:
    """Render the unified evaluation report as Markdown.

    Args:
        payload: Serialised evaluation results.

    Returns:
        Markdown document with the unified table and the change log.
    """
    split: str = payload.get("split", "test")
    dataset: str = payload.get("dataset", "dataset")
    conf: float = payload.get("conf_threshold", 0.25)
    iou: float = payload.get("iou_threshold", 0.5)
    models: List[Dict[str, Any]] = payload.get("models", [])
    control_id: str = payload.get("control_model_id", "Exp_A_Standard")

    control: Optional[Dict[str, Any]] = next(
        (m for m in models if m.get("model_id") == control_id), None
    )
    control_ci: Optional[Tuple[float, float]] = None
    if control:
        for metric in control.get("class_metrics", []):
            if metric.get("class_id") == PERSON_CLASS_ID and metric.get("recall_ci"):
                control_ci = tuple(metric["recall_ci"])  # type: ignore[assignment]

    lines: List[str] = [
        "# 📊 Unified Evaluation Results (Single Split, Per-Class Metrics)",
        "",
        f"- **Split evaluasi:** `{split}` (dataset `{dataset}`) — *satu split untuk seluruh model*",
        f"- **Confidence threshold:** `{conf}` | **IoU matching:** `{iou}`",
        "- **Metrik:** per-kelas (bukan macro-average), dilengkapi CI 95% (Wilson).",
        "",
        "## 1. Tabel Hasil Terpadu",
        "",
        "| Model | Recall person (CI 95%) | Precision person | F1 person | Recall vehicle | Macro F1 | mAP@50 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for model in models:
        metrics: List[Dict[str, Any]] = model.get("class_metrics", [])
        person = next((m for m in metrics if m.get("class_id") == PERSON_CLASS_ID), {})
        vehicle = next((m for m in metrics if m.get("class_id") == 0), {})
        map50: Optional[float] = model.get("map50")
        lines.append(
            f"| **{model.get('model_id')}** | "
            f"**{person.get('recall', 0.0):.2f}%** {_ci_text(person.get('recall_ci'))} | "
            f"{person.get('precision', 0.0):.2f}% | {person.get('f1_score', 0.0):.2f}% | "
            f"{vehicle.get('recall', 0.0):.2f}% | {model.get('macro_f1', 0.0):.2f}% | "
            f"{('%.2f%%' % map50) if map50 is not None else '-'} |"
        )

    lines.extend([
        "",
        "## 2. Catatan Perubahan Satu per Satu",
        "",
        f"Kontrol grup = **{control_id}** (fine-tuning tanpa augmentasi pencahayaan).",
        "Setiap baris menguji **satu variabel bebas** terhadap kontrol tersebut.",
        "",
        "| Eksperimen | Variabel bebas yang diuji | Recall person (CI 95%) | Δ vs kontrol | Verdict statistik |",
        "|:---|:---|:---:|:---:|:---|",
    ])

    for model in models:
        model_id: str = model.get("model_id", "")
        if model_id == control_id:
            continue
        metrics = model.get("class_metrics", [])
        person = next((m for m in metrics if m.get("class_id") == PERSON_CLASS_ID), {})
        candidate_ci = tuple(person["recall_ci"]) if person.get("recall_ci") else None
        delta: Optional[float] = None
        if control is not None and control_ci and candidate_ci:
            delta = person.get("recall", 0.0) - next(
                (m.get("recall", 0.0) for m in control.get("class_metrics", [])
                 if m.get("class_id") == PERSON_CLASS_ID),
                0.0,
            )
        lines.append(
            f"| **{model_id}** | {model.get('independent_variable', '-')} | "
            f"{person.get('recall', 0.0):.2f}% {_ci_text(candidate_ci)} | "
            f"{('{:+.2f} pp'.format(delta)) if delta is not None else '-'} | "
            f"{_verdict(control_ci, candidate_ci)} |"
        )

    lines.extend([
        "",
        "## 3. Catatan Metodologis",
        "",
        "1. **Per-kelas, bukan macro.** Recall perolehan diambil dari array per-kelas "
        "(`results.box.r`) sehingga tidak lagi tercampur dengan rata-rata antar kelas.",
        "2. **Interval keyakinan 95%.** Karena kelas `person` jarang (puluhan box), selisih "
        "beberapa poin persentase belum tentu signifikan secara statistik.",
        "3. **Split tunggal.** Seluruh model dievaluasi pada split yang sama agar perbandingan "
        "adil (syarat CP10).",
        "4. **Baseline B0 memakai taksonomi COCO** yang dipetakan ke 2 kelas, sehingga "
        "perbandingan mAP dengan model fine-tuned tidak sepenuhnya setara dan hanya "
        "disajikan sebagai acuan kasar.",
        "",
    ])
    return "\n".join(lines)


def compute_results(
    dataset_root: Path,
    split: str,
    conf_threshold: float,
    iou_threshold: float,
    nms_iou: float,
) -> Dict[str, Any]:
    """Evaluate every discovered model on one split.

    Args:
        dataset_root: Root of the dataset to evaluate.
        split: Split name ('test' or 'valid').
        conf_threshold: Confidence threshold.
        iou_threshold: IoU threshold for matching.
        nms_iou: IoU threshold used by the detector NMS.

    Returns:
        Serialisable payload with the per-model results.
    """
    from ultralytics import YOLO

    images_dir: Path = dataset_root / split / "images"
    labels_dir: Path = dataset_root / split / "labels"
    images: List[Path] = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    logger.info("Mengevaluasi %d citra dari split '%s'", len(images), split)

    models_payload: List[Dict[str, Any]] = []
    for entry in discover_models():
        if entry.weights_path is None:
            logger.warning("Checkpoint tidak ditemukan untuk %s, dilewati", entry.model_id)
            continue

        logger.info("Model %s <- %s", entry.model_id, entry.weights_path)
        model = YOLO(str(entry.weights_path))
        truths, predictions, _ = run_model_on_split(
            model, entry, images, labels_dir, conf_threshold, nms_iou
        )

        class_metrics = evaluate_detections(
            truths, predictions, iou_threshold=iou_threshold
        )
        precision, recall, f1 = macro_average(class_metrics)

        map50: Optional[float] = None
        map50_95: Optional[float] = None
        try:
            validation = model.val(
                data=str(dataset_root / "data.yaml"), split=split, device="cpu", verbose=False
            )
            results_dict: Dict[str, Any] = getattr(validation, "results_dict", {}) or {}
            if "metrics/mAP50(B)" in results_dict:
                map50 = float(results_dict["metrics/mAP50(B)"]) * 100
            if "metrics/mAP50-95(B)" in results_dict:
                map50_95 = float(results_dict["metrics/mAP50-95(B)"]) * 100
        except Exception as error:  # pragma: no cover - depends on the environment
            logger.warning("Validasi mAP gagal untuk %s: %s", entry.model_id, error)

        models_payload.append(
            {
                "model_id": entry.model_id,
                "display_name": entry.display_name,
                "weights_path": str(entry.weights_path),
                "independent_variable": entry.independent_variable,
                "is_zero_shot": entry.is_zero_shot,
                "class_metrics": [asdict(metric) for metric in class_metrics],
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "map50": map50,
                "map50_95": map50_95,
            }
        )

    return {
        "split": split,
        "dataset": dataset_root.name,
        "dataset_root": str(dataset_root),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "image_count": len(images),
        "control_model_id": "Exp_A_Standard",
        "models": models_payload,
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluasi terpadu seluruh model pada satu split (CP10)."
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid", "train"])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--from-json", type=Path, default=None, help="Render ulang laporan dari JSON")
    parser.add_argument("--output", type=Path, default=RESULT_MARKDOWN_PATH)
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        config = load_evaluation_config()
        conf = args.conf if args.conf is not None else config.conf_threshold
        iou = args.iou if args.iou is not None else config.iou_threshold
        payload = compute_results(
            dataset_root=args.dataset_root,
            split=args.split,
            conf_threshold=conf,
            iou_threshold=iou,
            nms_iou=config.nms_iou,
        )
        RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULT_JSON_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        logger.info("JSON hasil: %s", RESULT_JSON_PATH)

    report: str = render_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    logger.info("Laporan: %s", args.output)


if __name__ == "__main__":
    main()
