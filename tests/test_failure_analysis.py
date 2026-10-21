"""Unit tests for the failure categorisation heuristics (Checkpoint CP11).

The heuristics decide which explanation is attached to every error case, so
they are pinned down with synthetic images and boxes.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyze_failures import (  # noqa: E402
    categorize_false_negative,
    categorize_false_positive,
    mean_luminance_in_box,
)


class TestMeanLuminance:
    """Luminance proxy used by the contrast heuristic."""

    def test_reads_patch_mean(self) -> None:
        """The helper must return the mean of the inspected channel."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 40
        assert mean_luminance_in_box(image, (0, 0, 100, 100)) == pytest.approx(40.0)

    def test_box_is_clipped_to_image(self) -> None:
        """A box outside the image must not raise."""
        image = np.full((50, 50, 3), 10, dtype=np.uint8)
        value = mean_luminance_in_box(image, (-20, -20, 200, 200))
        assert value == pytest.approx(10.0)


class TestCategorizeFalseNegative:
    """Heuristics for missed pedestrians."""

    def test_small_object_category(self) -> None:
        """A pedestrian shorter than 10% of the image is classified as small."""
        category = categorize_false_negative(
            (10, 480, 30, 500), image_height=1000, image=None, vehicle_boxes=()
        )
        assert category == "small_object"

    def test_occluded_category(self) -> None:
        """A pedestrian overlapping a vehicle is classified as occluded."""
        category = categorize_false_negative(
            (100, 100, 200, 400),
            image_height=1000,
            image=None,
            vehicle_boxes=((90, 120, 210, 380),),
        )
        assert category == "occluded"

    def test_low_contrast_category(self) -> None:
        """A dark pedestrian on a bright background is classified as low contrast."""
        image = np.full((1000, 1000, 3), 200, dtype=np.uint8)
        image[100:400, 100:200] = 5
        category = categorize_false_negative(
            (100, 100, 200, 400),
            image_height=1000,
            image=image,
            vehicle_boxes=(),
            dataset_luminance=43.0,
        )
        assert category == "low_contrast"

    def test_fallback_category(self) -> None:
        """A large, bright, unoccluded pedestrian falls back to low confidence."""
        image = np.full((1000, 1000, 3), 200, dtype=np.uint8)
        category = categorize_false_negative(
            (100, 100, 200, 400),
            image_height=1000,
            image=image,
            vehicle_boxes=(),
            dataset_luminance=43.0,
        )
        assert category == "low_confidence"


class TestCategorizeFalsePositive:
    """Heuristics for spurious pedestrian detections."""

    def test_overlap_with_vehicle_is_ambiguity(self) -> None:
        """A detection on a vehicle is an annotation ambiguity."""
        category = categorize_false_positive(
            (100, 100, 200, 400), confidence=0.9, vehicle_boxes=((90, 120, 210, 380),)
        )
        assert category == "unlabelled_or_ambiguous"

    def test_low_confidence_detection(self) -> None:
        """A weak detection is flagged as a confidence problem."""
        category = categorize_false_positive((10, 10, 40, 60), confidence=0.31, vehicle_boxes=())
        assert category == "low_confidence"

    def test_plain_false_positive(self) -> None:
        """A confident detection without overlap is a plain false positive."""
        category = categorize_false_positive((10, 10, 40, 60), confidence=0.88, vehicle_boxes=())
        assert category == "false_positive"
