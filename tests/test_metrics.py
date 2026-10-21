"""Unit tests for the metric primitives (Checkpoint CP04, CP10).

The tests pin down the behaviour that the first version of the project got
wrong: per-class metrics must never be silently replaced by macro averages,
and recall on a rare class must carry an honest confidence interval.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import (  # noqa: E402
    Detection,
    compute_iou,
    evaluate_detections,
    macro_average,
    match_class_detections,
    per_class_metrics_from_ultralytics,
    wilson_interval,
)


def _box(class_id: int, x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> Detection:
    """Build a Detection with a compact constructor call."""
    return Detection(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


class TestComputeIou:
    """Behaviour of the IoU helper."""

    def test_identical_boxes_have_unit_iou(self) -> None:
        """Two identical boxes must produce IoU = 1.0."""
        box = _box(1, 0, 0, 10, 10)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_zero_iou(self) -> None:
        """Two disjoint boxes must produce IoU = 0.0."""
        assert compute_iou(_box(1, 0, 0, 10, 10), _box(1, 20, 20, 30, 30)) == 0.0

    def test_half_overlap(self) -> None:
        """Half-overlapping boxes must produce IoU = 1/3."""
        a = _box(1, 0, 0, 10, 10)
        b = _box(1, 5, 0, 15, 10)
        assert compute_iou(a, b) == pytest.approx(1 / 3, rel=1e-6)


class TestMatching:
    """Greedy confidence-ordered matching."""

    def test_single_exact_match(self) -> None:
        """One prediction on one truth yields (1, 0, 0)."""
        tp, fp, fn = match_class_detections(
            [_box(1, 0, 0, 10, 10)], [_box(1, 0, 0, 10, 10)], iou_threshold=0.5
        )
        assert (tp, fp, fn) == (1, 0, 0)

    def test_duplicate_prediction_is_false_positive(self) -> None:
        """A second prediction on an already matched truth is a false positive."""
        tp, fp, fn = match_class_detections(
            [_box(1, 0, 0, 10, 10)],
            [_box(1, 0, 0, 10, 10, 0.9), _box(1, 0, 0, 10, 10, 0.8)],
            iou_threshold=0.5,
        )
        assert (tp, fp, fn) == (1, 1, 0)

    def test_missed_truth_is_false_negative(self) -> None:
        """An unmatched ground truth is counted as a false negative."""
        tp, fp, fn = match_class_detections([_box(1, 0, 0, 10, 10)], [], iou_threshold=0.5)
        assert (tp, fp, fn) == (0, 0, 1)

    def test_class_mismatch_does_not_match(self) -> None:
        """Boxes of different classes must never be matched together."""
        tp, fp, fn = match_class_detections(
            [_box(1, 0, 0, 10, 10)], [_box(0, 0, 0, 10, 10)], iou_threshold=0.5
        )
        assert (tp, fp, fn) == (0, 1, 1)


class TestEvaluateDetections:
    """Per-class aggregation."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions yield 100% precision and recall for both classes."""
        truths = [[_box(0, 0, 0, 10, 10), _box(1, 20, 20, 30, 30)]]
        preds = [[_box(0, 0, 0, 10, 10), _box(1, 20, 20, 30, 30)]]
        metrics = evaluate_detections(truths, preds)
        assert all(m.precision == 100.0 for m in metrics)
        assert all(m.recall == 100.0 for m in metrics)

    def test_macro_average_is_not_used_as_class_value(self) -> None:
        """A class with zero ground truth must not inflate the other class."""
        truths = [[_box(1, 0, 0, 10, 10)]]
        preds = [[_box(1, 0, 0, 10, 10)]]
        metrics = evaluate_detections(truths, preds)
        vehicle = next(m for m in metrics if m.class_id == 0)
        person = next(m for m in metrics if m.class_id == 1)
        assert person.recall == 100.0
        # The vehicle class has no ground truth at all: recall is undefined, not 50%.
        assert vehicle.ground_truth == 0
        assert vehicle.recall == 0.0

    def test_recall_confidence_interval_present(self) -> None:
        """A class with ground truth must expose a confidence interval."""
        truths = [[_box(1, 0, 0, 10, 10)] for _ in range(10)]
        preds = [[_box(1, 0, 0, 10, 10)] for _ in range(7)] + [[] for _ in range(3)]
        metrics = evaluate_detections(truths, preds)
        person = next(m for m in metrics if m.class_id == 1)
        assert person.recall == pytest.approx(70.0)
        assert person.recall_ci is not None
        lower, upper = person.recall_ci
        assert lower < person.recall < upper


class TestWilsonInterval:
    """Statistical helper used for the rare pedestrian class."""

    def test_bounds_contain_estimate(self) -> None:
        """The interval must bracket the observed proportion."""
        lower, upper = wilson_interval(26, 39)
        assert lower < 26 / 39 < upper

    def test_zero_trials(self) -> None:
        """No trials yields a degenerate interval."""
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_invalid_counts(self) -> None:
        """Impossible counts raise an error."""
        with pytest.raises(ValueError):
            wilson_interval(5, 3)

    def test_interval_narrows_with_more_samples(self) -> None:
        """More observations must produce a narrower interval."""
        narrow = wilson_interval(130, 195)
        wide = wilson_interval(13, 39)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


class TestUltralyticsBridge:
    """Regression guard against the macro/per-class confusion."""

    def test_reads_per_class_arrays(self) -> None:
        """The helper must read the per-class arrays, not the macro dictionary."""

        class _Box:
            p = [0.53, 0.90]
            r = [0.79, 0.66]
            maps = [0.50, 0.45]
            ap50 = [0.74, 0.65]

        class _Results:
            box = _Box()
            names = {0: "Vehicle", 1: "person"}
            results_dict = {"metrics/recall(B)": 0.7275, "metrics/mAP50(B)": 0.695}

        metrics = per_class_metrics_from_ultralytics(_Results())
        assert len(metrics) == 2
        person = next(m for m in metrics if m.class_id == 1)
        vehicle = next(m for m in metrics if m.class_id == 0)
        assert person.recall == pytest.approx(66.0)
        assert vehicle.recall == pytest.approx(79.0)
        # The macro value (72.75%) must not leak into the per-class entries.
        assert person.recall != pytest.approx(72.75)


class TestMacroAverage:
    """Macro helper."""

    def test_averages_classes(self) -> None:
        """Macro averages are the unweighted mean over classes."""
        truths = [[_box(0, 0, 0, 10, 10), _box(1, 20, 20, 30, 30)]]
        preds = [[_box(0, 0, 0, 10, 10)]]
        metrics = evaluate_detections(truths, preds)
        precision, recall, f1 = macro_average(metrics)
        vehicle = next(m for m in metrics if m.class_id == 0)
        person = next(m for m in metrics if m.class_id == 1)
        assert precision == pytest.approx((vehicle.precision + person.precision) / 2)
        assert recall == pytest.approx((vehicle.recall + person.recall) / 2)
        assert f1 == pytest.approx((vehicle.f1_score + person.f1_score) / 2)
