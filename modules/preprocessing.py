"""
Module 1 — Image Preprocessing
───────────────────────────────────────────────────
Classifies image type (printed / handwritten / mixed)
and applies type-specific enhancement pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from skimage.transform import rotate as sk_rotate
from skimage.color import rgb2gray
import imutils
import logging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    MIN_IMAGE_WIDTH, MAX_SKEW_ANGLE, BORDER_PADDING,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# IMAGE TYPE CLASSIFICATION
# ────────────────────────────────────────────────

def _stroke_width_variance(gray: np.ndarray) -> float:
    """Estimate stroke width variance via distance transform on edges."""
    edges = cv2.Canny(gray, 50, 150)
    dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
    mask = edges > 0
    if mask.sum() == 0:
        return 0.0
    stroke_widths = dist[mask]
    return float(np.var(stroke_widths))


def _line_straightness(gray: np.ndarray) -> float:
    """Measure how straight detected lines are using HoughLinesP."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=40, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angles.append(angle)
    angles = np.array(angles)
    # Straightness = fraction of near-horizontal lines (0 ± 5°)
    horizontal = np.sum((angles < 5) | (angles > 175))
    return float(horizontal / len(angles))


def classify_image_type(image: np.ndarray) -> str:
    """
    Classify as 'printed', 'handwritten', or 'mixed'.

    Heuristic:
      - High stroke-width variance + low straightness → handwritten
      - Low variance + high straightness → printed
      - Otherwise → mixed
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    swv = _stroke_width_variance(gray)
    straightness = _line_straightness(gray)

    logger.info(f"Stroke width variance: {swv:.2f}, Straightness: {straightness:.2f}")

    if swv > 50 and straightness < 0.3:
        return "handwritten"
    elif swv < 25 and straightness > 0.4:
        return "printed"
    else:
        return "mixed"


# ────────────────────────────────────────────────
# PREPROCESSING STRATEGIES
# ────────────────────────────────────────────────

def _apply_clahe(gray: np.ndarray, clip: float = CLAHE_CLIP_LIMIT) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def _deskew(image: np.ndarray) -> np.ndarray:
    """Auto-correct skew up to MAX_SKEW_ANGLE degrees."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    coords = np.column_stack(np.where(gray < 128))
    if coords.shape[0] < 100:
        return image
    try:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > MAX_SKEW_ANGLE:
            logger.warning(f"Skew angle {angle:.1f}° exceeds max {MAX_SKEW_ANGLE}°, skipping")
            return image
        if abs(angle) > 0.5:
            logger.info(f"Correcting skew: {angle:.1f}°")
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        logger.warning(f"Deskew failed: {e}")
    return image


def _upscale_if_needed(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w < MIN_IMAGE_WIDTH:
        scale = MIN_IMAGE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Upscaled from {w}x{h} to {new_w}x{new_h}")
    return image


def _add_border(image: np.ndarray, pad: int = BORDER_PADDING) -> np.ndarray:
    return cv2.copyMakeBorder(image, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])


def preprocess_printed(image: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral filter + Otsu threshold + deskew + denoising."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = _apply_clahe(gray, clip=CLAHE_CLIP_LIMIT)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.medianBlur(binary, 3) # Remove isolated noise pixels
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary = _deskew(binary)
    return binary


def preprocess_handwritten(image: np.ndarray) -> np.ndarray:
    """Strong CLAHE + Gaussian blur + adaptive threshold + denoising + dilation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = _apply_clahe(gray, clip=CLAHE_CLIP_LIMIT * 2)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    binary = cv2.medianBlur(binary, 3) # Prevent salt and pepper noise crashes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary


def preprocess_mixed(image: np.ndarray) -> np.ndarray:
    """Adaptive threshold + moderate CLAHE + auto deskew + denoising."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = _apply_clahe(gray, clip=CLAHE_CLIP_LIMIT)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 15, 4)
    binary = cv2.medianBlur(binary, 3) # Prevent salt and pepper noise crashes
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary = _deskew(binary)
    return binary


# ────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────

STRATEGY = {
    "printed":     preprocess_printed,
    "handwritten": preprocess_handwritten,
    "mixed":       preprocess_mixed,
}


def preprocess_image(image_path: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns
    -------
    dict with keys:
        original        : np.ndarray — original image
        processed       : np.ndarray — processed image ready for OCR
        image_type      : str — 'printed' | 'handwritten' | 'mixed'
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original = image.copy()

    # Classify
    img_type = classify_image_type(image)
    logger.info(f"Image classified as: {img_type}")

    # Blur Detection & Adaptive Sharpening
    gray_for_blur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lap_var = cv2.Laplacian(gray_for_blur, cv2.CV_64F).var()
    logger.info(f"Blur variance (Laplacian): {lap_var:.2f}")
    if lap_var < 150.0:  # Threshold for blurry image
        logger.info("Image appears blurry, applying sharpening filter...")
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)

    # Upscale
    image = _upscale_if_needed(image)

    # Type-specific preprocessing
    processed = STRATEGY[img_type](image)

    # Add border for edge text
    processed = _add_border(processed)

    return {
        "original":   original,
        "processed":  processed,
        "image_type": img_type,
    }


def get_comparison_figure(original: np.ndarray, processed: np.ndarray):
    """Return a matplotlib figure showing original vs processed side by side."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Processed")
    axes[1].axis("off")
    plt.tight_layout()
    return fig
