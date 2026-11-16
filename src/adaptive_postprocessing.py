"""Adaptive Post-Processing Engine for Low-Light Vision and ADAS.

This module provides luminance-aware dynamic confidence thresholding and glare
detection to optimize object detection under non-uniform nocturnal illumination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Named Constants
# ----------------------------------------------------------------------------
MIN_SAFE_CONFIDENCE: float = 0.15
MIN_PERSON_CONFIDENCE: float = 0.12
MAX_SAFE_CONFIDENCE: float = 0.45
DARK_LUMINANCE_THRESHOLD: float = 35.0  # L* scale (0 - 100)
BRIGHT_LUMINANCE_THRESHOLD: float = 70.0
GLARE_PIXEL_INTENSITY: int = 245        # Pixel intensity (0 - 255)
HIGH_GLARE_RATIO: float = 0.035         # 3.5% saturated pixels indicates headlamp glare
CONFIDENCE_BOOST_DELTA: float = 0.07    # Reduction in confidence for dark scenes (general/vehicle)
PERSON_BOOST_DELTA: float = 0.13        # Stronger reduction for under-confident person class
GLARE_PENALTY_DELTA: float = 0.10       # Increase in confidence under severe glare
BOX_GLARE_REJECTION_RATIO: float = 0.40 # Reject low-conf person boxes dominated by saturated glare
PRIOR_AGREEMENT_IOU: float = 0.40       # IoU threshold for specialist + generalist consensus
PRIOR_RECOVERY_MIN_CONF: float = 0.35   # Min B0 confidence to recover out-of-distribution person
PRIOR_RECOVERY_VEHICLE_CONFLICT_IOU: float = 0.50


@dataclass(frozen=True)
class AdaptiveIlluminationProfile:
    """Diagnostic profile describing scene illumination and adaptive parameters.

    Attributes:
        luminance: Average scene luminance on the CIE L* scale (0.0 to 100.0).
        glare_ratio: Fraction of pixels saturated by direct glare (0.0 to 1.0).
        effective_conf_threshold: Dynamically calculated general/vehicle confidence threshold.
        effective_person_conf: Class-specific threshold calibrated for nighttime pedestrians.
        mode_label: Human-readable operational mode for dashboard display.
        is_glare_detected: Flag indicating severe headlamp/streetlight glare.
        is_low_light_boost_active: Flag indicating enhanced dark sensitivity.
    """

    luminance: float
    glare_ratio: float
    effective_conf_threshold: float
    effective_person_conf: float
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
        effective_person_conf: float = min(MAX_SAFE_CONFIDENCE, base_conf + (penalty * 0.5))
        mode_label: str = (
            f"Strict Glare Suppression (Person: {effective_person_conf:.2f}, "
            f"Vehicle: {effective_conf:.2f})"
        )
        is_glare: bool = True
        is_boost: bool = False

    # Condition 2: Nocturnal Low-Light Boost (Prioritize Pedestrian Recall)
    elif luminance < DARK_LUMINANCE_THRESHOLD:
        deficit_ratio: float = (DARK_LUMINANCE_THRESHOLD - luminance) / DARK_LUMINANCE_THRESHOLD
        boost: float = CONFIDENCE_BOOST_DELTA * deficit_ratio
        person_boost: float = PERSON_BOOST_DELTA * deficit_ratio
        effective_conf = max(MIN_SAFE_CONFIDENCE, base_conf - boost)
        effective_person_conf = max(MIN_PERSON_CONFIDENCE, base_conf - person_boost)
        mode_label = (
            f"Low-Light Sensitivity Boost (Person: {effective_person_conf:.2f}, "
            f"Vehicle: {effective_conf:.2f})"
        )
        is_glare = False
        is_boost = True

    # Condition 3: Standard Balanced Illumination
    else:
        effective_conf = base_conf
        effective_person_conf = base_conf
        mode_label = f"Standard Balanced Mode (Conf: {effective_conf:.2f})"
        is_glare = False
        is_boost = False

    return AdaptiveIlluminationProfile(
        luminance=round(luminance, 2),
        glare_ratio=round(glare_ratio, 4),
        effective_conf_threshold=round(effective_conf, 2),
        effective_person_conf=round(effective_person_conf, 2),
        mode_label=mode_label,
        is_glare_detected=is_glare,
        is_low_light_boost_active=is_boost,
    )


def compute_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes.

    Args:
        box_a: First bounding box coordinates [x1, y1, x2, y2].
        box_b: Second bounding box coordinates [x1, y1, x2, y2].

    Returns:
        IoU score in the range [0.0, 1.0].
    """
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def is_glare_artifact_box(
    image: np.ndarray,
    box: Sequence[float],
    glare_threshold: int = GLARE_PIXEL_INTENSITY,
    max_glare_ratio: float = BOX_GLARE_REJECTION_RATIO,
) -> bool:
    """Check whether a bounding box is dominated by saturated glare pixels.

    Args:
        image: Source image array (H, W, 3) or grayscale (H, W).
        box: Bounding box coordinates [x1, y1, x2, y2].
        glare_threshold: Pixel intensity considered saturated.
        max_glare_ratio: Maximum allowed fraction of saturated pixels inside the box.

    Returns:
        True if the box region is dominated by glare artifacts, False otherwise.
    """
    if image is None or image.size == 0:
        return False

    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(box[0])))
    y1 = max(0, min(h - 1, int(box[1])))
    x2 = max(0, min(w, int(box[2])))
    y2 = max(0, min(h, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        return False

    crop = image[y1:y2, x1:x2]
    return detect_glare_ratio(crop, glare_threshold=glare_threshold) >= max_glare_ratio


def fuse_dual_prior_detections(
    specialist_detections: Sequence[Dict[str, Any]],
    prior_person_detections: Sequence[Dict[str, Any]],
    person_conf_threshold: float,
    vehicle_conf_threshold: float,
    image: Optional[np.ndarray] = None,
    agreement_iou: float = PRIOR_AGREEMENT_IOU,
    recovery_min_conf: float = PRIOR_RECOVERY_MIN_CONF,
) -> List[Dict[str, Any]]:
    """Fuse nighttime specialist detections with generalist COCO person priors.

    Combines specialist detections (e.g., Exp_D_Robust) with generalist person
    detections (e.g., B0_ZeroShot) to resolve out-of-distribution viewpoint
    failures (such as close-up / sidewalk camera angles) and class-imbalance
    confidence suppression without retraining.

    Args:
        specialist_detections: Detections from the specialist model, each with
            ``class_id`` (0=Vehicle, 1=person), ``confidence``, ``box``, and optional ``track_id``.
        prior_person_detections: Person detections from the generalist prior model,
            each with ``confidence`` and ``box``.
        person_conf_threshold: Effective threshold for accepting person detections.
        vehicle_conf_threshold: Effective threshold for accepting vehicle detections.
        image: Optional image array for local glare-artifact rejection on low-conf boxes.
        agreement_iou: Minimum IoU for specialist-prior consensus boosting.
        recovery_min_conf: Minimum prior confidence to recover missed pedestrians.

    Returns:
        Filtered and fused list of detection dictionaries.
    """
    fused: List[Dict[str, Any]] = []
    accepted_vehicles: List[Dict[str, Any]] = []
    accepted_persons: List[Dict[str, Any]] = []
    matched_prior_indices: set[int] = set()

    # 1. Process specialist Vehicle detections
    for det in specialist_detections:
        if int(det["class_id"]) != 0:
            continue
        if float(det["confidence"]) >= vehicle_conf_threshold:
            vehicle_item = {
                "class_id": 0,
                "confidence": float(det["confidence"]),
                "box": list(det["box"]),
                "track_id": det.get("track_id"),
                "source": "specialist",
            }
            accepted_vehicles.append(vehicle_item)
            fused.append(vehicle_item)

    # 2. Process specialist person detections with optional consensus boost
    for det in specialist_detections:
        if int(det["class_id"]) != 1:
            continue
        spec_conf = float(det["confidence"])
        best_prior_idx: Optional[int] = None
        best_iou: float = 0.0

        for idx, prior in enumerate(prior_person_detections):
            iou = compute_box_iou(det["box"], prior["box"])
            if iou >= agreement_iou and iou > best_iou:
                best_iou = iou
                best_prior_idx = idx

        effective_score = spec_conf
        source_tag = "specialist"
        if best_prior_idx is not None:
            matched_prior_indices.add(best_prior_idx)
            prior_conf = float(prior_person_detections[best_prior_idx]["confidence"])
            if prior_conf > effective_score:
                effective_score = prior_conf
                source_tag = "consensus"

        if effective_score < person_conf_threshold:
            continue

        # Reject unconfirmed very-low-confidence boxes dominated by glare
        if (
            source_tag == "specialist"
            and effective_score < 0.25
            and image is not None
            and is_glare_artifact_box(image, det["box"])
        ):
            continue

        person_item = {
            "class_id": 1,
            "confidence": effective_score,
            "box": list(det["box"]),
            "track_id": det.get("track_id"),
            "source": source_tag,
        }
        accepted_persons.append(person_item)
        fused.append(person_item)

    # 3. Recover missed out-of-distribution pedestrians from high-confidence prior
    for idx, prior in enumerate(prior_person_detections):
        if idx in matched_prior_indices:
            continue
        prior_conf = float(prior["confidence"])
        if prior_conf < recovery_min_conf:
            continue

        # Avoid duplicating an already-accepted person detection
        if any(compute_box_iou(prior["box"], p["box"]) >= agreement_iou for p in accepted_persons):
            continue

        # Avoid overriding a high-confidence specialist Vehicle detection
        if any(
            compute_box_iou(prior["box"], v["box"]) >= PRIOR_RECOVERY_VEHICLE_CONFLICT_IOU
            for v in accepted_vehicles
        ):
            continue

        recovered_item = {
            "class_id": 1,
            "confidence": prior_conf,
            "box": list(prior["box"]),
            "track_id": None,
            "source": "prior_recovery",
        }
        accepted_persons.append(recovered_item)
        fused.append(recovered_item)

    return fused
