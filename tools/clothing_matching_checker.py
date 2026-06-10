"""
Clothing Matching Checker Tool

Checks whether top and bottom clothing colors look coordinated.
Designed for audio-first feedback for blind and low-vision users.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


NEUTRAL_COLORS = {"black", "white", "gray", "navy", "beige", "brown"}
COLOR_GROUPS = {
    "warm": {"red", "orange", "yellow", "brown", "pink"},
    "cool": {"green", "cyan", "blue", "purple", "navy"},
    "neutral": {"black", "white", "gray", "beige"},
}
COMPLEMENTARY_PAIRS = {
    frozenset(("blue", "orange")),
    frozenset(("purple", "yellow")),
    frozenset(("red", "green")),
}


def _limit_words(text: str, max_words: int = 15) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    shortened = " ".join(words[:max_words]).rstrip(".,;:")
    return f"{shortened}."


def _extract_region(image: np.ndarray, y_start_ratio: float, y_end_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    y1 = max(0, int(h * y_start_ratio))
    y2 = min(h, int(h * y_end_ratio))
    x1 = max(0, int(w * 0.25))
    x2 = min(w, int(w * 0.75))
    if y2 <= y1 or x2 <= x1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def _classify_color_from_hsv(h: float, s: float, v: float) -> str:
    if v < 40:
        return "black"
    if s < 25 and v > 210:
        return "white"
    if s < 35:
        return "gray"

    if (h < 10) or (h >= 170):
        return "red"
    if h < 20:
        return "brown" if v < 130 else "orange"
    if h < 33:
        return "yellow"
    if h < 78:
        return "green"
    if h < 95:
        return "cyan"
    if h < 130:
        return "blue" if s > 80 else "navy"
    if h < 160:
        return "purple"
    return "pink"


def _get_dominant_color(region: np.ndarray) -> str:
    if region is None or region.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    if pixels.size == 0:
        return "unknown"

    vivid_mask = (pixels[:, 1] > 25) & (pixels[:, 2] > 25)
    selected = pixels[vivid_mask] if np.any(vivid_mask) else pixels

    if selected.shape[0] > 6000:
        step = max(1, selected.shape[0] // 6000)
        selected = selected[::step]

    h = float(np.median(selected[:, 0]))
    s = float(np.median(selected[:, 1]))
    v = float(np.median(selected[:, 2]))
    return _classify_color_from_hsv(h, s, v)


def _get_color_group(color: str) -> str:
    for group_name, group_colors in COLOR_GROUPS.items():
        if color in group_colors:
            return group_name
    return "other"


def evaluate_match(top_color: str, bottom_color: str) -> Tuple[str, str]:
    if top_color == "unknown" or bottom_color == "unknown":
        return "I could not clearly detect both clothing colors. Please try better lighting.", "warning"

    if top_color == bottom_color:
        return f"Great match. Both pieces are {top_color}.", "success"

    if top_color in NEUTRAL_COLORS or bottom_color in NEUTRAL_COLORS:
        return f"These match well. {top_color} with {bottom_color} is a safe combination.", "success"

    if frozenset((top_color, bottom_color)) in COMPLEMENTARY_PAIRS:
        return f"These colors coordinate nicely: {top_color} with {bottom_color}.", "success"

    if _get_color_group(top_color) == _get_color_group(bottom_color):
        return f"These colors match well. {top_color} and {bottom_color} feel coordinated.", "success"

    return f"These may clash a bit: {top_color} with {bottom_color}. Consider switching one piece.", "warning"


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Entry point for ProgramAT tools.

    Args:
        image: OpenCV BGR frame
        input_data: Optional dict. Supports:
            - is_streaming: bool, enforce short response for streaming use

    Returns:
        Dict with audio + text output
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {"type": "error", "text": "No camera image available.", "rate": 1.0, "interrupt": False},
            "text": "No camera image available.",
        }

    if not isinstance(input_data, dict):
        input_data = {}

    is_streaming = bool(input_data.get("is_streaming", False))

    try:
        top_region = _extract_region(image, 0.18, 0.52)
        bottom_region = _extract_region(image, 0.56, 0.92)

        top_color = _get_dominant_color(top_region)
        bottom_color = _get_dominant_color(bottom_region)
        feedback, audio_type = evaluate_match(top_color, bottom_color)

        text = feedback
        if is_streaming:
            text = _limit_words(text, max_words=15)

        return {
            "audio": {
                "type": audio_type if audio_type in {"success", "warning", "error", "speech"} else "speech",
                "text": text,
                "rate": 1.0,
                "interrupt": audio_type == "warning",
            },
            "text": text,
        }
    except Exception:
        return {
            "audio": {
                "type": "error",
                "text": "I could not check outfit matching right now.",
                "rate": 1.0,
                "interrupt": False,
            },
            "text": "I could not check outfit matching right now.",
        }


__all__ = [
    "main",
    "evaluate_match",
]
