"""Unit tests for adaptive post-processing and dynamic illumination profiling."""

import numpy as np
import pytest

from src.adaptive_postprocessing import (
    DARK_LUMINANCE_THRESHOLD,
    MAX_SAFE_CONFIDENCE,
    MIN_SAFE_CONFIDENCE,
    calculate_frame_luminance,
    compute_adaptive_illumination_profile,
    detect_glare_ratio,
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
    assert "Standard Balanced Mode" in profile.mode_label
