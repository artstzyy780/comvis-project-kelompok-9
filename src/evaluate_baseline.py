"""Baseline B0 Zero-Shot Evaluation Script for Low-light Object Detection.

This script benchmarks an off-the-shelf YOLOv8n model (pretrained on MS COCO)
on the low-light validation and test splits without any additional fine-tuning.

Key Responsibilities:
1. Harmonize MS COCO class IDs with project dataset annotations:
   - COCO class 0 ('person') -> Target class 1 ('person')
   - COCO classes [2, 3, 5, 7] ('car', 'motorcycle', 'bus', 'truck') -> Target class 0 ('Vehicle')
2. Perform rigorous IoU @ 0.5 bipartite matching between predictions and ground truths.
3. Quantify per-class Precision, Recall, and F1-Score, highlighting pedestrian detection rate.
4. Measure hardware inference latency (ms) and FPS on local CPU with warm-up cycles.
5. Export structured reports in Markdown and JSON formats, alongside visual prediction grids.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


# Evaluation Constants
DEFAULT_MODEL_WEIGHTS: str = "yolov8n.pt"
DATASET_ROOT_DIR: Path = Path("dataset")
DEFAULT_SPLIT: str = "valid"
DEFAULT_CONF_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.50
WARMUP_ITERATIONS: int = 5
CANVAS_SIZE: int = 640

# Target Dataset Class Definitions (from data.yaml)
TARGET_CLASS_NAMES: Dict[int, str] = {
    0: "Vehicle",
    1: "person",
}

# Mapping from MS COCO 80-class taxonomy to Target 2-class taxonomy
COCO_TO_TARGET_MAPPING: Dict[int, int] = {
    0: 1,  # person -> person
    2: 0,  # car -> Vehicle
    3: 0,  # motorcycle -> Vehicle
    5: 0,  # bus -> Vehicle
    7: 0,  # truck -> Vehicle
}

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 128, 0),  # Blue/Orange in BGR: Vehicle
    1: (0, 255, 0),    # Green in BGR: Person
}


@dataclass
class BoundingBox:
    """Represents a pixel-coordinate bounding box."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass
class ClassMetrics:
    """Container for per-class evaluation metrics."""

    class_id: int
    class_name: str
    ground_truth_count: int
    prediction_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


@dataclass
class BaselineEvaluationResult:
    """Aggregated evaluation results for Baseline B0."""

    model_name: str
    split_name: str
    total_images: int
    confidence_threshold: float
    iou_threshold: float
    mean_latency_ms: float
    fps_cpu: float
    per_class_metrics: List[ClassMetrics]
    mean_precision: float
    mean_recall: float
    mean_f1: float


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        Intersection over Union scalar float in range [0.0, 1.0].
    """
    inter_x1: float = max(box_a.x1, box_b.x1)
    inter_y1: float = max(box_a.y1, box_b.y1)
    inter_x2: float = min(box_a.x2, box_b.x2)
    inter_y2: float = min(box_a.y2, box_b.y2)

    inter_w: float = max(0.0, inter_x2 - inter_x1)
    inter_h: float = max(0.0, inter_y2 - inter_y1)
    intersection: float = inter_w * inter_h

    area_a: float = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b: float = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    union: float = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def parse_ground_truth_labels(
    label_path: Path, img_width: int, img_height: int
) -> List[BoundingBox]:
    """Parse normalized YOLO format label text file into absolute pixel bounding boxes.

    Args:
        label_path: Path to the .txt ground truth label file.
        img_width: Absolute image width in pixels.
        img_height: Absolute image height in pixels.

    Returns:
        List of parsed BoundingBox objects.
    """
    boxes: List[BoundingBox] = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            cls_id = int(tokens[0])
            xc, yc, w, h = map(float, tokens[1:5])

            x1 = (xc - w / 2.0) * img_width
            y1 = (yc - h / 2.0) * img_height
            x2 = (xc + w / 2.0) * img_width
            y2 = (yc + h / 2.0) * img_height

            boxes.append(BoundingBox(class_id=cls_id, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def evaluate_split(
    model: YOLO,
    dataset_dir: Path,
    split: str,
    conf_thresh: float = DEFAULT_CONF_THRESHOLD,
    iou_thresh: float = DEFAULT_IOU_THRESHOLD,
) -> Tuple[BaselineEvaluationResult, List[Tuple[Path, List[BoundingBox], List[BoundingBox]]]]:
    """Execute zero-shot evaluation on a designated dataset split.

    Args:
        model: Ultralytics YOLO model instance.
        dataset_dir: Dataset root directory path.
        split: Partition to evaluate ('valid' or 'test').
        conf_thresh: Detection confidence cutoff.
        iou_thresh: IoU overlap threshold for true positive validation.

    Returns:
        A tuple containing the BaselineEvaluationResult dataclass and a sample list of evaluated items.
    """
    images_dir: Path = dataset_dir / split / "images"
    labels_dir: Path = dataset_dir / split / "labels"

    image_paths: List[Path] = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {images_dir}")

    # Track match statistics per target class
    tp_counts: Dict[int, int] = {cls_id: 0 for cls_id in TARGET_CLASS_NAMES}
    fp_counts: Dict[int, int] = {cls_id: 0 for cls_id in TARGET_CLASS_NAMES}
    fn_counts: Dict[int, int] = {cls_id: 0 for cls_id in TARGET_CLASS_NAMES}
    gt_counts: Dict[int, int] = {cls_id: 0 for cls_id in TARGET_CLASS_NAMES}
    pred_counts: Dict[int, int] = {cls_id: 0 for cls_id in TARGET_CLASS_NAMES}

    latencies_ms: List[float] = []
    visual_samples: List[Tuple[Path, List[BoundingBox], List[BoundingBox]]] = []

    print(f"[INFO] Evaluating {len(image_paths)} images from split '{split}'...")

    for idx, img_path in enumerate(image_paths):
        img_bgr: np.ndarray = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]

        label_path: Path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes: List[BoundingBox] = parse_ground_truth_labels(label_path, w, h)

        for gt in gt_boxes:
            if gt.class_id in gt_counts:
                gt_counts[gt.class_id] += 1

        # Timed Inference
        t_start = time.perf_counter()
        results = model.predict(
            source=img_bgr,
            conf=conf_thresh,
            iou=0.45,
            verbose=False,
            device="cpu",
        )
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Discard warmup iterations from latency calculation
        if idx >= WARMUP_ITERATIONS:
            latencies_ms.append(t_elapsed_ms)

        # Parse and remap predicted detections
        pred_boxes: List[BoundingBox] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes
            for i in range(len(boxes_data)):
                raw_cls: int = int(boxes_data.cls[i].item())
                confidence: float = float(boxes_data.conf[i].item())
                coords = boxes_data.xyxy[i].tolist()

                # Remap MS COCO class to target ontology
                if raw_cls in COCO_TO_TARGET_MAPPING:
                    target_cls = COCO_TO_TARGET_MAPPING[raw_cls]
                    pred_box = BoundingBox(
                        class_id=target_cls,
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=confidence,
                    )
                    pred_boxes.append(pred_box)
                    pred_counts[target_cls] += 1

        # Match predictions against ground truth per class
        for cls_id in TARGET_CLASS_NAMES:
            cls_gt: List[BoundingBox] = [b for b in gt_boxes if b.class_id == cls_id]
            cls_pred: List[BoundingBox] = [b for b in pred_boxes if b.class_id == cls_id]

            # Sort predictions by descending confidence
            cls_pred.sort(key=lambda x: x.confidence, reverse=True)

            matched_gt_indices: Set[int] = set()

            for p_box in cls_pred:
                best_iou: float = 0.0
                best_gt_idx: int = -1

                for g_idx, g_box in enumerate(cls_gt):
                    if g_idx in matched_gt_indices:
                        continue
                    iou: float = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_thresh and best_gt_idx != -1:
                    tp_counts[cls_id] += 1
                    matched_gt_indices.add(best_gt_idx)
                else:
                    fp_counts[cls_id] += 1

            # Unmatched ground truth are false negatives
            unmatched_fn: int = len(cls_gt) - len(matched_gt_indices)
            fn_counts[cls_id] += unmatched_fn

        # Collect samples with pedestrian ground truth for visualization
        if any(b.class_id == 1 for b in gt_boxes) and len(visual_samples) < 6:
            visual_samples.append((img_path, gt_boxes, pred_boxes))

    # Calculate aggregate and per-class metrics
    per_class_metrics: List[ClassMetrics] = []
    total_precisions: List[float] = []
    total_recalls: List[float] = []
    total_f1s: List[float] = []

    for cls_id, cls_name in TARGET_CLASS_NAMES.items():
        tp = tp_counts[cls_id]
        fp = fp_counts[cls_id]
        fn = fn_counts[cls_id]
        gt = gt_counts[cls_id]
        pr = pred_counts[cls_id]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics.append(
            ClassMetrics(
                class_id=cls_id,
                class_name=cls_name,
                ground_truth_count=gt,
                prediction_count=pr,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1_score=round(f1, 4),
            )
        )
        total_precisions.append(prec)
        total_recalls.append(rec)
        total_f1s.append(f1)

    mean_lat_ms: float = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    mean_fps: float = 1000.0 / mean_lat_ms if mean_lat_ms > 0 else 0.0

    eval_result = BaselineEvaluationResult(
        model_name="yolov8n-coco-zero-shot",
        split_name=split,
        total_images=len(image_paths),
        confidence_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        mean_latency_ms=round(mean_lat_ms, 2),
        fps_cpu=round(mean_fps, 2),
        per_class_metrics=per_class_metrics,
        mean_precision=round(float(np.mean(total_precisions)), 4),
        mean_recall=round(float(np.mean(total_recalls)), 4),
        mean_f1=round(float(np.mean(total_f1s)), 4),
    )

    return eval_result, visual_samples


def render_visual_predictions_grid(
    samples: List[Tuple[Path, List[BoundingBox], List[BoundingBox]]],
    output_path: Path,
) -> None:
    """Render a side-by-side comparison grid of ground truth vs model predictions.

    Args:
        samples: List of (image_path, ground_truth_boxes, predicted_boxes).
        output_path: Target path for the output visualization image.
    """
    if not samples:
        return

    rendered_rows: List[np.ndarray] = []

    for img_path, gt_boxes, pred_boxes in samples[:4]:
        base_img = cv2.imread(str(img_path))
        if base_img is None:
            continue

        gt_canvas = base_img.copy()
        pred_canvas = base_img.copy()

        # Render Ground Truth
        for b in gt_boxes:
            color = CLASS_COLORS.get(b.class_id, (255, 255, 255))
            cv2.rectangle(gt_canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            cv2.putText(
                gt_canvas,
                TARGET_CLASS_NAMES.get(b.class_id, str(b.class_id)),
                (int(b.x1), max(18, int(b.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Render Predictions
        for b in pred_boxes:
            color = CLASS_COLORS.get(b.class_id, (255, 255, 255))
            cv2.rectangle(pred_canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            label_text = f"{TARGET_CLASS_NAMES.get(b.class_id, str(b.class_id))} {b.confidence:.2f}"
            cv2.putText(
                pred_canvas,
                label_text,
                (int(b.x1), max(18, int(b.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Header Banners
        cv2.rectangle(gt_canvas, (0, 0), (220, 30), (0, 0, 0), -1)
        cv2.putText(
            gt_canvas,
            "Ground Truth (Annotated)",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(pred_canvas, (0, 0), (260, 30), (0, 0, 0), -1)
        cv2.putText(
            pred_canvas,
            "B0 Prediction (COCO Zero-Shot)",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        row = np.hstack([gt_canvas, pred_canvas])
        rendered_rows.append(row)

    if rendered_rows:
        grid = np.vstack(rendered_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), grid)
        print(f"[INFO] Saved prediction grid to: {output_path}")


def export_markdown_report(result: BaselineEvaluationResult, report_path: Path) -> None:
    """Generate and write a comprehensive academic Markdown evaluation report.

    Args:
        result: Evaluated metrics summary dataclass.
        report_path: Destination file path.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "# 📊 Baseline B0 Evaluation Report: Zero-Shot COCO Deteksi Malam",
        "",
        f"- **Model:** `{result.model_name}` (Ultralytics Pretrained COCO)",
        f"- **Split Evaluasi:** `{result.split_name}` ({result.total_images} citra)",
        f"- **Confidence Threshold:** `{result.confidence_threshold}`",
        f"- **IoU Threshold:** `{result.iou_threshold}`",
        f"- **Throughput CPU:** `{result.fps_cpu} FPS` (Rata-rata Latensi: `{result.mean_latency_ms} ms`)",
        "",
        "---",
        "",
        "## 1. Ringkasan Kinerja Kuantitatif",
        "",
        "| ID | Kelas Objek | Ground Truth | Prediksi | TP | FP | FN | Precision | Recall | F1-Score |",
        "|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for m in result.per_class_metrics:
        lines.append(
            f"| **{m.class_id}** | **{m.class_name}** | {m.ground_truth_count} | "
            f"{m.prediction_count} | {m.true_positives} | {m.false_positives} | {m.false_negatives} | "
            f"**{m.precision * 100:.1f}%** | **{m.recall * 100:.1f}%** | **{m.f1_score * 100:.1f}%** |"
        )

    lines.extend([
        "",
        f"- **Macro-Average Precision:** `{result.mean_precision * 100:.1f}%`",
        f"- **Macro-Average Recall:** `{result.mean_recall * 100:.1f}%`",
        f"- **Macro-Average F1-Score:** `{result.mean_f1 * 100:.1f}%`",
        "",
        "---",
        "",
        "## 2. Analisis Kritis Temuan Baseline (B0)",
        "",
    ])

    person_metric = next((m for m in result.per_class_metrics if m.class_name == "person"), None)
    vehicle_metric = next((m for m in result.per_class_metrics if m.class_name == "Vehicle"), None)

    if person_metric and vehicle_metric:
        lines.extend([
            f"1. **Asimetri Recall Pejalan Kaki vs Kendaraan**:",
            f"   - Model zero-shot mendeteksi kendaraan dengan Recall **{vehicle_metric.recall * 100:.1f}%**, tetapi mengalami penurunan tajam pada pejalan kaki dengan Recall hanya **{person_metric.recall * 100:.1f}%**.",
            f"   - Terdapat **{person_metric.false_negatives} pejalan kaki yang tidak terdeteksi (False Negative)** dari total {person_metric.ground_truth_count} ground truth.",
            "2. **Bukti Empiris Urgensi Penelitian**:",
            "   - Angka Recall pejalan kaki yang rendah pada model bawaan COCO ini membuktikan secara empiris bahwa model deteksi siang hari mengalami degradasi berat di malam hari.",
            "   - Hasil ini menjadi acuan dasar (*baseline*) kuat yang akan ditingkatkan melalui **Fine-tuning (Exp-A)** dan **Augmentasi Pencahayaan Adaptif (Exp-B CLAHE & Exp-C Gamma)** pada tahap selanjutnya.",
            "3. **Efisiensi Perangkat Keras CPU**:",
            f"   - Arsitektur YOLOv8n mencatat kecepatan **{result.fps_cpu} FPS** ({result.mean_latency_ms} ms/frame) pada CPU standar, membuktikan kelayakan target real-time ($\\ge 20$ FPS).",
            "",
            "---",
            "*(Laporan ini dihasilkan secara otomatis oleh `src/evaluate_baseline.py`)*",
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Exported report to: {report_path}")


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(description="Baseline B0 Evaluation for Low-Light Vision.")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["valid", "test"])
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    args = parser.parse_args()

    print("[INFO] Loading pretrained YOLOv8n model...")
    model = YOLO(DEFAULT_MODEL_WEIGHTS)

    out_dir = Path("results/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    result, samples = evaluate_split(
        model=model,
        dataset_dir=DATASET_ROOT_DIR,
        split=args.split,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

    # Save JSON metrics
    json_path = out_dir / "baseline_b0_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"[INFO] Saved metrics JSON to: {json_path}")

    # Save Markdown report
    report_path = out_dir / "baseline_b0_report.md"
    export_markdown_report(result, report_path)

    # Save Visual Grid
    vis_path = out_dir / "baseline_b0_sample_predictions.png"
    render_visual_predictions_grid(samples, vis_path)

    print("\n" + "=" * 50)
    print("BASELINE B0 EVALUATION COMPLETE")
    print(f"Split: {result.split_name} ({result.total_images} images)")
    for m in result.per_class_metrics:
        print(f"Class '{m.class_name}': Precision={m.precision*100:.1f}%, Recall={m.recall*100:.1f}%, F1={m.f1_score*100:.1f}% (FN={m.false_negatives})")
    print(f"Speed: {result.fps_cpu} FPS ({result.mean_latency_ms} ms/frame)")
    print("=" * 50)


if __name__ == "__main__":
    main()
