"""Adaptive illumination preprocessing module for Low-light Pedestrian Detection.

This module provides deterministic, high-throughput image enhancement techniques:
1. L-channel CLAHE (Contrast Limited Adaptive Histogram Equalization) in CIE L*a*b* space.
2. Non-linear Gamma Correction using an optimized 256-element Lookup Table (LUT).

These functions enhance low-light pedestrian silhouettes and edge features
without corrupting chromaticity channels or incurring heavy deep-learning latency.
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


# Configuration Constants
DEFAULT_CLAHE_CLIP_LIMIT: float = 2.0
DEFAULT_CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
DEFAULT_GAMMA_VALUE: float = 0.6  # Values < 1.0 lift dark shadow regions


def apply_clahe_lab(
    image: np.ndarray,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = DEFAULT_CLAHE_TILE_GRID,
) -> np.ndarray:
    """Enhance luminance contrast using CLAHE applied solely to the L* channel.

    Converts an input BGR image into the CIE L*a*b* color space, isolates the
    lightness (L*) channel, applies Contrast Limited Adaptive Histogram Equalization,
    and recombines with original chrominance (a*, b*) channels. This prevents
    unwanted color cast shifts and noise over-amplification.

    Args:
        image: Input 3-channel image in BGR format (uint8, shape [H, W, 3]).
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Number of rectangular grid tiles for local histogram equalisation.

    Returns:
        Enhanced 3-channel image in BGR format (uint8, shape [H, W, 3]).

    Raises:
        ValueError: If input image is empty or not 3-channel uint8.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape: {image.shape}")

    # Convert BGR to CIE L*a*b* color space
    lab_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the lightness channel only
    clahe: cv2.CLAHE = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size
    )
    enhanced_l: np.ndarray = clahe.apply(l_channel)

    # Merge channels and convert back to BGR
    merged_lab: np.ndarray = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float = DEFAULT_GAMMA_VALUE,
) -> np.ndarray:
    """Apply non-linear tonal adjustment using an optimized Lookup Table (LUT).

    Transforms pixel intensities via the power-law equation I_out = 255 * (I_in / 255)^(gamma).
    For gamma < 1.0, shadow details are expanded while bright highlights are preserved.

    Args:
        image: Input image in BGR or grayscale format (uint8).
        gamma: Exponent parameter. Values < 1.0 brighten dark regions.

    Returns:
        Gamma-corrected image with identical dtype and spatial dimensions.

    Raises:
        ValueError: If input image is empty or gamma <= 0.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if gamma <= 0:
        raise ValueError(f"Gamma parameter must be strictly positive, got: {gamma}")

    # Construct 256-element LUT mapping input [0..255] to output [0..255]
    inv_gamma: float = 1.0 / gamma
    lut_values: np.ndarray = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )

    # Execute hardware-accelerated lookup
    return cv2.LUT(image, lut_values)


def create_comparison_grid(
    image: np.ndarray,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Generate a horizontal comparison canvas: Original, CLAHE, and Gamma.

    Args:
        image: Original input BGR image.
        output_path: Optional file path to save the generated comparison image.

    Returns:
        Horizontal concatenated BGR image with descriptive labels.
    """
    clahe_img: np.ndarray = apply_clahe_lab(image)
    gamma_img: np.ndarray = apply_gamma_correction(image)

    # Annotate labels onto images
    def annotate(img: np.ndarray, text: str) -> np.ndarray:
        annotated: np.ndarray = img.copy()
        cv2.rectangle(annotated, (0, 0), (260, 36), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    panel_orig: np.ndarray = annotate(image, "Original (Low-Light)")
    panel_clahe: np.ndarray = annotate(clahe_img, "CLAHE (L-Channel)")
    panel_gamma: np.ndarray = annotate(gamma_img, f"Gamma (gamma={DEFAULT_GAMMA_VALUE})")

    grid: np.ndarray = np.hstack([panel_orig, panel_clahe, panel_gamma])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), grid)

    return grid


if __name__ == "__main__":
    # Test on a sample validation image if available
    sample_dir: Path = Path("dataset/valid/images")
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.jpg"))
        if sample_files:
            test_img: np.ndarray = cv2.imread(str(sample_files[0]))
            out_file: Path = Path("results/preprocessing/sample_enhancement_comparison.png")
            create_comparison_grid(test_img, output_path=out_file)
            print(f"[INFO] Successfully created enhancement comparison at: {out_file}")
