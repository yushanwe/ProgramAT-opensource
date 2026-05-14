"""
Center Color Detection Tool

Detects the dominant color of the object/region in the middle of the camera frame
and returns an audio-friendly color name.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np


DEFAULT_CENTER_RATIO = 0.2


def extract_center_region(image: np.ndarray, center_ratio: float = DEFAULT_CENTER_RATIO) -> np.ndarray:
    """Extract a centered region from an image."""
    height, width = image.shape[:2]
    ratio = float(np.clip(center_ratio, 0.05, 1.0))

    half_w = max(1, int(width * ratio / 2.0))
    half_h = max(1, int(height * ratio / 2.0))

    cx, cy = width // 2, height // 2
    x1, x2 = max(0, cx - half_w), min(width, cx + half_w)
    y1, y2 = max(0, cy - half_h), min(height, cy + half_h)

    return image[y1:y2, x1:x2]


def classify_hsv_color(hsv_region: np.ndarray) -> str:
    """Classify a color name from an HSV image region."""
    if hsv_region.size == 0:
        return "unknown"

    h = hsv_region[:, :, 0].astype(np.float32)
    s = hsv_region[:, :, 1].astype(np.float32)
    v = hsv_region[:, :, 2].astype(np.float32)

    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    if mean_v < 40:
        return "black"
    if mean_s < 25 and mean_v > 210:
        return "white"
    if mean_s < 35:
        return "gray"

    colorful_mask = (s > 40) & (v > 40)
    if not np.any(colorful_mask):
        return "gray"

    selected_h = h[colorful_mask]
    mean_h = float(np.mean(selected_h))

    if 8 <= mean_h < 25 and mean_v < 170 and mean_s > 60:
        return "brown"
    if mean_h < 10 or mean_h >= 170:
        return "red"
    if mean_h < 22:
        return "orange"
    if mean_h < 35:
        return "yellow"
    if mean_h < 85:
        return "green"
    if mean_h < 100:
        return "cyan"
    if mean_h < 135:
        return "blue"
    if mean_h < 155:
        return "purple"
    return "pink"


def detect_center_color(image: np.ndarray, center_ratio: float = DEFAULT_CENTER_RATIO) -> str:
    """Detect the dominant color of the image center region."""
    center_region = extract_center_region(image, center_ratio=center_ratio)

    if center_region.size == 0:
        return "unknown"

    hsv_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
    return classify_hsv_color(hsv_region)


def parse_center_ratio(input_data: Any) -> float:
    """Read center_ratio from input_data if present."""
    if isinstance(input_data, dict):
        ratio = input_data.get("center_ratio", DEFAULT_CENTER_RATIO)
        try:
            return float(ratio)
        except (TypeError, ValueError):
            return DEFAULT_CENTER_RATIO
    return DEFAULT_CENTER_RATIO


def main(image: np.ndarray, input_data: Any = None) -> str:
    """Main entry point for ProgramAT tool runtime."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"

    center_ratio = parse_center_ratio(input_data)
    color = detect_center_color(image, center_ratio=center_ratio)

    return color if color != "unknown" else "Color not clear"


__all__ = [
    "main",
    "extract_center_region",
    "classify_hsv_color",
    "detect_center_color",
    "parse_center_ratio",
]
