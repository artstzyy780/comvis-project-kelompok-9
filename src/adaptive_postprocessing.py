"""Adaptive Post-Processing Engine for Low-Light Vision and ADAS.

This module provides luminance-aware dynamic confidence thresholding and glare
detection to optimize object detection under non-uniform nocturnal illumination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Named Constants
# ----------------------------------------------------------------------------
MIN_SAFE_CONFIDENCE: float = 0.15
MAX_SAFE_CONFIDENCE: float = 0.45
DARK_LUMINANCE_THRESHOLD: float = 35.0  # L* scale (0 - 100)
BRIGHT_LUMINANCE_THRESHOLD: float = 70.0
GLARE_PIXEL_INTENSITY: int = 245        # Pixel intensity (0 - 255)
HIGH_GLARE_RATIO: float = 0.035         # 3.5% saturated pixels indicates headlamp glare
CONFIDENCE_BOOST_DELTA: float = 0.07    # Reduction in confidence for dark scenes
GLARE_PENALTY_DELTA: float = 0.10       # Increase in confidence under severe glare


@dataclass(frozen=True)
class AdaptiveIlluminationProfile:
    """Diagnostic profile describing scene illumination and adaptive parameters.

    Attributes:
        luminance: Average scene luminance on the CIE L* scale (0.0 to 100.0).
        glare_ratio: Fraction of pixels saturated by direct glare (0.0 to 1.0).
        effective_conf_threshold: Dynamically calculated confidence threshold.
        mode_label: Human-readable operational mode for dashboard display.
        is_glare_detected: Flag indicating severe headlamp/streetlight glare.
        is_low_light_boost_active: Flag indicating enhanced dark sensitivity.
    """

    luminance: float
    glare_ratio: float
    effective_conf_threshold: float
    mode_label: str
    is_glare_detected: bool
    is_low_light_boost_active: bool


def calculate_frame_luminance(image: np.ndarray, is_bgr: bool = True) -> float:
    """Calculate average perceptual luminance of an image using CIE L*a*b* space.

    Args:
        image: Input image array (H, W, 3) in BGR or RGB format.
        is_bgr: True if image is in BGR format (OpenCV default), False if RGB.

    Returns:
        Average luminance value normalized to the CIE L* range [0.0, 100.0].

    Raises:
        ValueError: If input image is empty or has invalid shape.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {image.shape}")

    conversion_code = cv2.COLOR_BGR2LAB if is_bgr else cv2.COLOR_RGB2LAB
    lab_image = cv2.cvtColor(image, conversion_code)
    
    # In OpenCV, L* is mapped to [0, 255]. Normalize back to [0.0, 100.0].
    raw_l_mean: float = float(np.mean(lab_image[:, :, 0]))
    return (raw_l_mean / 255.0) * 100.0


def detect_glare_ratio(image: np.ndarray, glare_threshold: int = GLARE_PIXEL_INTENSITY) -> float:
    """Detect the ratio of overexposed/saturated pixels causing glare.

    Args:
        image: Input image array (H, W, 3).
        glare_threshold: Minimum intensity threshold considered saturated (default 245).

    Returns:
        Ratio of glare pixels to total pixels [0.0, 1.0].

    Raises:
        ValueError: If input image is empty or invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    total_pixels: int = gray.size
    glare_pixels: int = int(np.count_nonzero(gray >= glare_threshold))
    return float(glare_pixels / total_pixels) if total_pixels > 0 else 0.0


def compute_adaptive_illumination_profile(
    image: np.ndarray,
    base_conf: float = 0.25,
    is_bgr: bool = True,
) -> AdaptiveIlluminationProfile:
    """Compute the adaptive illumination profile and dynamic threshold for an image.

    This function analyzes ambient luminance and optical glare, adjusting the
    confidence threshold dynamically:
    - In low-light environments, the threshold is safely lowered to capture dim
      pedestrians that would otherwise become false negatives.
    - Under severe headlamp glare, the threshold is elevated to eliminate spurious
      false positives triggered by lens flare or reflection.

    Args:
        image: Input image array.
        base_conf: User-selected base confidence threshold (default 0.25).
        is_bgr: True if image is in BGR format, False if RGB.

    Returns:
        AdaptiveIlluminationProfile containing computed metrics and dynamic threshold.
    """
    luminance: float = calculate_frame_luminance(image, is_bgr=is_bgr)
    glare_ratio: float = detect_glare_ratio(image)

    # Condition 1: Severe Glare Detection (Prioritize False Positive Suppression)
    if glare_ratio >= HIGH_GLARE_RATIO:
        penalty: float = GLARE_PENALTY_DELTA * min(2.0, glare_ratio / HIGH_GLARE_RATIO)
        effective_conf: float = min(MAX_SAFE_CONFIDENCE, base_conf + penalty)
        mode_label: str = f"Strict Glare Suppression (Conf: {effective_conf:.2f})"
        is_glare: bool = True
        is_boost: bool = False

    # Condition 2: Nocturnal Low-Light Boost (Prioritize Pedestrian Recall)
    elif luminance < DARK_LUMINANCE_THRESHOLD:
        deficit_ratio: float = (DARK_LUMINANCE_THRESHOLD - luminance) / DARK_LUMINANCE_THRESHOLD
        boost: float = CONFIDENCE_BOOST_DELTA * deficit_ratio
        effective_conf = max(MIN_SAFE_CONFIDENCE, base_conf - boost)
        mode_label = f"Low-Light Sensitivity Boost (Conf: {effective_conf:.2f})"
        is_glare = False
        is_boost = True

    # Condition 3: Standard Balanced Illumination
    else:
        effective_conf = base_conf
        mode_label = f"Standard Balanced Mode (Conf: {effective_conf:.2f})"
        is_glare = False
        is_boost = False

    return AdaptiveIlluminationProfile(
        luminance=round(luminance, 2),
        glare_ratio=round(glare_ratio, 4),
        effective_conf_threshold=round(effective_conf, 2),
        mode_label=mode_label,
        is_glare_detected=is_glare,
        is_low_light_boost_active=is_boost,
    )
