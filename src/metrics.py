"""Single source of truth for the quantitative evaluation of the project (CP04, CP10).

Why this module exists
----------------------
The project reports *pedestrian* recall as its primary metric, yet the first
version of the training notebook extracted ``results_dict["metrics/recall(B)"]``
from Ultralytics. That key is the **mean over classes**, not the recall of the
``person`` class, so the headline table silently compared a per-class value
against a macro average. This module removes that ambiguity:

- :func:`per_class_metrics_from_ultralytics` returns the true per-class
  precision/recall/mAP from an Ultralytics validation result.
- :func:`evaluate_detections` computes per-class P/R/F1 from raw predictions
  with an explicit IoU matching rule, independent of any framework.
- :func:`wilson_interval` and :func:`bootstrap_interval` quantify the sampling
  uncertainty, which matters because the pedestrian class is very rare.

All functions are framework agnostic except the explicit ``ultralytics``
helpers, which import the library lazily so the module can be unit tested on a
machine without PyTorch installed.

Example:
    >>> from metrics import wilson_interval
    >>> lower, upper = wilson_interval(successes=26, total=39)
    >>> round(lower, 3), round(upper, 3)
    (0.507, 0.812)
"""

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import CLASS_NAMES, PERSON_CLASS_ID


# ----------------------------------------------------------------------------
# Data Structures
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    """A single predicted bounding box in absolute pixel coordinates.

    Attributes:
        class_id: Class identifier of the prediction.
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
        confidence: Prediction confidence in the range [0, 1].
    """

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall and F1 for one class at one IoU threshold.

    Attributes:
        class_id: Class identifier.
        class_name: Human readable class name.
        ground_truth: Number of ground-truth boxes.
        predictions: Number of predicted boxes.
        true_positives: Matched predictions.
        false_positives: Unmatched predictions.
        false_negatives: Unmatched ground-truth boxes.
        precision: Precision in percent.
        recall: Recall in percent.
        f1_score: F1-score in percent.
        map50: Per-class mAP@50 in percent, when available.
        map50_95: Per-class mAP@50:95 in percent, when available.
        recall_ci: Confidence interval of the recall, when available.
        precision_ci: Confidence interval of the precision, when available.
    """

    class_id: int
    class_name: str
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    recall_ci: Optional[Tuple[float, float]] = None
    precision_ci: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated result of evaluating one model on one split.

    Attributes:
        model_id: Identifier of the evaluated model.
        split_name: Evaluated split.
        image_count: Number of evaluated images.
        class_metrics: Per-class metrics.
        macro_precision: Macro-average precision in percent.
        macro_recall: Macro-average recall in percent.
        macro_f1: Macro-average F1 in percent.
        map50: mAP@50 in percent, when available.
        map50_95: mAP@50:95 in percent, when available.
    """

    model_id: str
    split_name: str
    image_count: int
    class_metrics: List[ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    map50: Optional[float] = None
    map50_95: Optional[float] = None

    def get_class(self, class_id: int) -> Optional[ClassMetrics]:
        """Return the metrics of one class.

        Args:
            class_id: Class identifier to look up.

        Returns:
            The matching ClassMetrics, or None when the class is absent.
        """
        for metric in self.class_metrics:
            if metric.class_id == class_id:
                return metric
        return None

    @property
    def person_recall(self) -> Optional[float]:
        """Return the pedestrian recall in percent, or None when unavailable."""
        metric: Optional[ClassMetrics] = self.get_class(PERSON_CLASS_ID)
        return metric.recall if metric else None

    @property
    def person_precision(self) -> Optional[float]:
        """Return the pedestrian precision in percent, or None when unavailable."""
        metric: Optional[ClassMetrics] = self.get_class(PERSON_CLASS_ID)
        return metric.precision if metric else None


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def compute_iou(box_a: Detection, box_b: Detection) -> float:
    """Compute the intersection over union of two boxes.

    Args:
        box_a: First box.
        box_b: Second box.

    Returns:
        IoU value in the range [0, 1].
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


# ----------------------------------------------------------------------------
# Confidence Intervals
# ----------------------------------------------------------------------------
def wilson_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    The Wilson interval is preferred over the normal approximation because the
    pedestrian class is rare (few dozen boxes), where the normal approximation
    would produce invalid bounds.

    Args:
        successes: Number of successful outcomes (e.g. true positives).
        total: Number of trials (e.g. ground-truth boxes).
        confidence_level: Two-sided confidence level in the range (0, 1).

    Returns:
        Tuple of (lower bound, upper bound) as ratios in the range [0, 1].

    Raises:
        ValueError: If total is negative or successes exceeds total.
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(f"Invalid counts: successes={successes}, total={total}")

    if total == 0:
        return (0.0, 0.0)

    z: float = _z_score(confidence_level)
    proportion: float = successes / total
    denominator: float = 1.0 + (z ** 2) / total
    centre: float = proportion + (z ** 2) / (2 * total)
    margin: float = z * np.sqrt(
        (proportion * (1 - proportion) / total) + (z ** 2) / (4 * total ** 2)
    )
    lower: float = max(0.0, (centre - margin) / denominator)
    upper: float = min(1.0, (centre + margin) / denominator)
    return (float(lower), float(upper))


def _z_score(confidence_level: float) -> float:
    """Return the two-sided z-score for a confidence level.

    Args:
        confidence_level: Confidence level such as 0.95.

    Returns:
        The z-score obtained from the standard normal quantile function.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    return float(NormalDist().inv_cdf((1.0 + confidence_level) / 2.0))


def bootstrap_interval(
    values: Sequence[float],
    statistic: str = "mean",
    samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Estimate a confidence interval with the bootstrap resampling method.

    Args:
        values: Sample used to resample.
        statistic: Either 'mean' or 'median'.
        samples: Number of bootstrap resamples.
        confidence_level: Two-sided confidence level.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower bound, upper bound).

    Raises:
        ValueError: If the sample is empty or the statistic is unknown.
    """
    if not values:
        raise ValueError("Cannot bootstrap an empty sample.")
    if statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported statistic: {statistic}")

    array: np.ndarray = np.asarray(values, dtype=float)
    rng: np.random.default_rng = np.random.default_rng(seed)
    indices: np.ndarray = rng.integers(0, len(array), size=(samples, len(array)))

    if statistic == "mean":
        replicates: np.ndarray = array[indices].mean(axis=1)
    else:
        replicates: np.ndarray = np.median(array[indices], axis=1)

    alpha: float = (1.0 - confidence_level) / 2.0
    lower: float = float(np.percentile(replicates, alpha * 100))
    upper: float = float(np.percentile(replicates, (1 - alpha) * 100))
    return (lower, upper)


# ----------------------------------------------------------------------------
# Evaluation From Raw Detections
# ----------------------------------------------------------------------------
def match_class_detections(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """Match predictions to ground truth greedily by descending confidence.

    Args:
        ground_truth: Ground-truth boxes of one class.
        predictions: Predicted boxes of one class.
        iou_threshold: Minimum IoU for a match.

    Returns:
        Tuple of (true positives, false positives, false negatives).
    """
    ordered: List[Detection] = sorted(predictions, key=lambda det: det.confidence, reverse=True)
    matched: set = set()
    true_positives: int = 0

    for prediction in ordered:
        best_iou: float = 0.0
        best_index: int = -1
        for index, truth in enumerate(ground_truth):
            if index in matched:
                continue
            # A prediction may only ever match a ground truth of the same class.
            if truth.class_id != prediction.class_id:
                continue
            overlap: float = compute_iou(prediction, truth)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            true_positives += 1

    false_positives: int = len(ordered) - true_positives
    false_negatives: int = len(ground_truth) - len(matched)
    return true_positives, false_positives, false_negatives


def evaluate_detections(
    per_image_ground_truth: Sequence[Sequence[Detection]],
    per_image_predictions: Sequence[Sequence[Detection]],
    iou_threshold: float = 0.5,
    confidence_level: float = 0.95,
) -> List[ClassMetrics]:
    """Compute per-class precision, recall and F1 with confidence intervals.

    Args:
        per_image_ground_truth: Ground-truth boxes grouped per image.
        per_image_predictions: Predicted boxes grouped per image (same order).
        iou_threshold: IoU threshold used for matching.
        confidence_level: Confidence level of the reported intervals.

    Returns:
        List of ClassMetrics ordered by class identifier.
    """
    true_positives: Dict[int, int] = {cls: 0 for cls in CLASS_NAMES}
    false_positives: Dict[int, int] = {cls: 0 for cls in CLASS_NAMES}
    false_negatives: Dict[int, int] = {cls: 0 for cls in CLASS_NAMES}
    totals: Dict[int, int] = {cls: 0 for cls in CLASS_NAMES}
    predictions: Dict[int, int] = {cls: 0 for cls in CLASS_NAMES}

    for truths, predictions_list in zip(per_image_ground_truth, per_image_predictions):
        for cls_id in CLASS_NAMES:
            cls_truth: List[Detection] = [det for det in truths if det.class_id == cls_id]
            cls_pred: List[Detection] = [det for det in predictions_list if det.class_id == cls_id]
            totals[cls_id] += len(cls_truth)
            predictions[cls_id] += len(cls_pred)

            tp, fp, fn = match_class_detections(cls_truth, cls_pred, iou_threshold)
            true_positives[cls_id] += tp
            false_positives[cls_id] += fp
            false_negatives[cls_id] += fn

    metrics: List[ClassMetrics] = []
    for cls_id, cls_name in CLASS_NAMES.items():
        tp = true_positives[cls_id]
        fp = false_positives[cls_id]
        fn = false_negatives[cls_id]
        gt = totals[cls_id]

        precision: float = (tp / (tp + fp) * 100) if (tp + fp) else 0.0
        recall: float = (tp / gt * 100) if gt else 0.0
        f1: float = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        recall_ci: Optional[Tuple[float, float]] = None
        if gt:
            lower, upper = wilson_interval(tp, gt, confidence_level)
            recall_ci = (lower * 100, upper * 100)

        precision_ci: Optional[Tuple[float, float]] = None
        if (tp + fp):
            lower, upper = wilson_interval(tp, tp + fp, confidence_level)
            precision_ci = (lower * 100, upper * 100)

        metrics.append(
            ClassMetrics(
                class_id=cls_id,
                class_name=cls_name,
                ground_truth=gt,
                predictions=predictions[cls_id],
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=round(precision, 2),
                recall=round(recall, 2),
                f1_score=round(f1, 2),
                recall_ci=recall_ci,
                precision_ci=precision_ci,
            )
        )

    return metrics


# ----------------------------------------------------------------------------
# Ultralytics Bridge (lazy import, optional dependency)
# ----------------------------------------------------------------------------
def per_class_metrics_from_ultralytics(results: Any) -> List[ClassMetrics]:
    """Extract true per-class metrics from an Ultralytics validation result.

    ``results_dict["metrics/recall(B)"]`` is the **mean over classes**. This
    helper reads ``results.box.p``, ``results.box.r`` and ``results.box.ap``
    instead, which are the per-class arrays, and therefore removes the
    macro/per-class confusion that affected the first version of the report.

    Args:
        results: The ``Results`` object returned by ``model.val()`` or
            ``model.train()``.

    Returns:
        List of ClassMetrics ordered by class identifier.
    """
    metrics: List[ClassMetrics] = []
    box: Any = getattr(results, "box", None)
    if box is None:
        return metrics

    names: Dict[int, str] = dict(getattr(results, "names", CLASS_NAMES))
    precisions: Any = getattr(box, "p", [])
    recalls: Any = getattr(box, "r", [])
    maps: Any = getattr(box, "maps", [])
    ap_class: Any = getattr(box, "ap50", getattr(box, "ap", None))

    for index, class_id in enumerate(sorted(names.keys())):
        precision: float = float(precisions[index]) * 100 if index < len(precisions) else 0.0
        recall: float = float(recalls[index]) * 100 if index < len(recalls) else 0.0
        f1: float = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        map50: float = float(ap_class[index]) * 100 if ap_class is not None and index < len(ap_class) else 0.0
        map50_95: float = float(maps[index]) * 100 if index < len(maps) else 0.0
        metrics.append(
            ClassMetrics(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                ground_truth=0,
                predictions=0,
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                precision=round(precision, 2),
                recall=round(recall, 2),
                f1_score=round(f1, 2),
                map50=round(map50, 2),
                map50_95=round(map50_95, 2),
            )
        )
    return metrics


def ultralytics_map_metrics(results: Any) -> Tuple[Optional[float], Optional[float]]:
    """Extract mAP@50 and mAP@50:95 from an Ultralytics result object.

    Args:
        results: Ultralytics ``Results`` or validation result object.

    Returns:
        Tuple of (map50 in percent, map50_95 in percent); None when absent.
    """
    results_dict: Dict[str, Any] = getattr(results, "results_dict", {}) or {}
    map50: Optional[float] = results_dict.get("metrics/mAP50(B)")
    map50_95: Optional[float] = results_dict.get("metrics/mAP50-95(B)")
    if map50 is not None:
        map50 = float(map50) * 100
    if map50_95 is not None:
        map50_95 = float(map50_95) * 100
    return (map50, map50_95)


def macro_average(metrics: Sequence[ClassMetrics]) -> Tuple[float, float, float]:
    """Compute the macro-average precision, recall and F1 in percent.

    Args:
        metrics: Per-class metrics.

    Returns:
        Tuple of (macro precision, macro recall, macro F1) in percent.
    """
    if not metrics:
        return (0.0, 0.0, 0.0)
    precision: float = float(np.mean([m.precision for m in metrics]))
    recall: float = float(np.mean([m.recall for m in metrics]))
    f1: float = float(np.mean([m.f1_score for m in metrics]))
    return (round(precision, 2), round(recall, 2), round(f1, 2))


if __name__ == "__main__":
    lower, upper = wilson_interval(26, 39)
    print(f"Recall 26/39 = {26/39*100:.2f}%  CI95% = [{lower*100:.1f}%, {upper*100:.1f}%]")
    lower, upper = wilson_interval(25, 39)
    print(f"Recall 25/39 = {25/39*100:.2f}%  CI95% = [{lower*100:.1f}%, {upper*100:.1f}%]")
    lower, upper = wilson_interval(19, 39)
    print(f"Recall 19/39 = {19/39*100:.2f}%  CI95% = [{lower*100:.1f}%, {upper*100:.1f}%]")
