"""Unit tests for adaptive post-processing and dynamic illumination profiling."""

import numpy as np
import pytest

from src.adaptive_postprocessing import (
    DARK_LUMINANCE_THRESHOLD,
    MAX_SAFE_CONFIDENCE,
    MIN_PERSON_CONFIDENCE,
    MIN_SAFE_CONFIDENCE,
    calculate_frame_luminance,
    compute_adaptive_illumination_profile,
    compute_box_iou,
    detect_glare_ratio,
    fuse_dual_prior_detections,
    is_glare_artifact_box,
)


def test_calculate_frame_luminance_dark():
    """Verify that a pure black image returns 0.0 luminance."""
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    luminance = calculate_frame_luminance(black_img, is_bgr=True)
    assert luminance == pytest.approx(0.0, abs=0.5)


def test_calculate_frame_luminance_bright():
    """Verify that a pure white image returns ~100.0 luminance on CIE L* scale."""
    white_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    luminance = calculate_frame_luminance(white_img, is_bgr=True)
    assert luminance == pytest.approx(100.0, abs=1.0)


def test_calculate_frame_luminance_invalid():
    """Verify that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        calculate_frame_luminance(None)

    with pytest.raises(ValueError):
        calculate_frame_luminance(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        calculate_frame_luminance(np.zeros((100, 100), dtype=np.uint8))


def test_detect_glare_ratio_clean():
    """Verify that an image with no saturated pixels has a glare ratio of 0.0."""
    clean_img = np.full((100, 100, 3), 100, dtype=np.uint8)
    glare_ratio = detect_glare_ratio(clean_img)
    assert glare_ratio == 0.0


def test_detect_glare_ratio_with_saturation():
    """Verify that an image with 10% saturated pixels returns a 0.10 glare ratio."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:10, :, :] = 255  # 10 rows out of 100 = 10%
    glare_ratio = detect_glare_ratio(img)
    assert glare_ratio == pytest.approx(0.10, abs=0.01)


def test_compute_adaptive_profile_low_light():
    """Verify that a nocturnal scene activates low-light boost and lowers threshold."""
    dark_img = np.full((100, 100, 3), 20, dtype=np.uint8)
    base_conf = 0.25
    profile = compute_adaptive_illumination_profile(dark_img, base_conf=base_conf)

    assert profile.is_low_light_boost_active is True
    assert profile.is_glare_detected is False
    assert profile.effective_conf_threshold < base_conf
    assert profile.effective_conf_threshold >= MIN_SAFE_CONFIDENCE
    assert "Low-Light Sensitivity Boost" in profile.mode_label


def test_compute_adaptive_profile_glare_suppression():
    """Verify that direct headlamp glare elevates the threshold to suppress false alarms."""
    img = np.full((100, 100, 3), 30, dtype=np.uint8)
    # Add 8% saturated glare region
    img[:8, :, :] = 255
    base_conf = 0.25
    profile = compute_adaptive_illumination_profile(img, base_conf=base_conf)

    assert profile.is_glare_detected is True
    assert profile.is_low_light_boost_active is False
    assert profile.effective_conf_threshold > base_conf
    assert profile.effective_conf_threshold <= MAX_SAFE_CONFIDENCE
    assert "Glare Suppression" in profile.mode_label


def test_compute_adaptive_profile_balanced():
    """Verify that a scene with moderate lighting preserves the base threshold."""
    moderate_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    base_conf = 0.25
    profile = compute_adaptive_illumination_profile(moderate_img, base_conf=base_conf)

    assert profile.is_glare_detected is False
    assert profile.is_low_light_boost_active is False
    assert profile.effective_conf_threshold == pytest.approx(base_conf, abs=0.01)
    assert profile.effective_person_conf == pytest.approx(base_conf, abs=0.01)
    assert "Standard Balanced Mode" in profile.mode_label


def test_compute_adaptive_profile_class_specific_person_boost():
    """Verify dark scenes lower person threshold more aggressively than vehicle threshold."""
    dark_img = np.full((100, 100, 3), 15, dtype=np.uint8)
    profile = compute_adaptive_illumination_profile(dark_img, base_conf=0.25)

    assert profile.effective_person_conf < profile.effective_conf_threshold
    assert profile.effective_person_conf >= MIN_PERSON_CONFIDENCE


def test_compute_box_iou_overlap_and_disjoint():
    """Verify IoU calculation for identical, partial, and disjoint boxes."""
    assert compute_box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert compute_box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert compute_box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_is_glare_artifact_box_detects_saturated_region():
    """Verify glare artifact detection inside a bounding box."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:30, 10:30, :] = 255
    assert is_glare_artifact_box(img, [10, 10, 30, 30]) is True
    assert is_glare_artifact_box(img, [50, 50, 80, 80]) is False


def test_fuse_dual_prior_detections_consensus_and_recovery():
    """Verify consensus boost, glare rejection, and out-of-distribution recovery."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[150:180, 150:180, :] = 255  # Glare artifact region

    specialist = [
        {"class_id": 0, "confidence": 0.60, "box": [10, 10, 50, 50], "track_id": None},
        {"class_id": 1, "confidence": 0.15, "box": [60, 60, 100, 140], "track_id": None},
        # Low-conf person inside glare region -> should be rejected
        {"class_id": 1, "confidence": 0.18, "box": [150, 150, 180, 180], "track_id": None},
    ]
    prior = [
        # Overlaps with specialist [60,60,100,140] -> consensus boost to 0.56
        {"confidence": 0.56, "box": [62, 62, 100, 138]},
        # Missed close-up pedestrian -> recovered
        {"confidence": 0.69, "box": [110, 20, 145, 130]},
        # Conflicts with Vehicle [10,10,50,50] -> should NOT override vehicle
        {"confidence": 0.45, "box": [12, 12, 48, 48]},
    ]

    fused = fuse_dual_prior_detections(
        specialist_detections=specialist,
        prior_person_detections=prior,
        person_conf_threshold=0.16,
        vehicle_conf_threshold=0.25,
        image=img,
    )

    vehicles = [d for d in fused if d["class_id"] == 0]
    persons = [d for d in fused if d["class_id"] == 1]

    assert len(vehicles) == 1
    assert len(persons) == 2
    assert any(p["source"] == "consensus" and p["confidence"] == pytest.approx(0.56) for p in persons)
    assert any(p["source"] == "prior_recovery" and p["confidence"] == pytest.approx(0.69) for p in persons)
