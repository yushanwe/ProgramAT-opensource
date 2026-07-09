"""
Plug Point Finder Tool

Finds visible electrical plug points, counts them, and gives directional cues.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from model_router_client import copilot_llm_call

TOOL_NAME = "plug_point_finder"
LOGGER = logging.getLogger(__name__)
STRAIGHT_AHEAD_THRESHOLD = 0.11
SLIGHT_SIDE_THRESHOLD = 0.20
STRONG_SIDE_THRESHOLD = 0.34

_THREAD_STATE = threading.local()


def reset_state() -> None:
    setattr(_THREAD_STATE, "blocked_frame_streak", 0)


def _get_blocked_frame_streak() -> int:
    return int(getattr(_THREAD_STATE, "blocked_frame_streak", 0))


def _set_blocked_frame_streak(value: int) -> None:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("blocked_frame_streak cannot be negative")
    setattr(_THREAD_STATE, "blocked_frame_streak", parsed)


def _is_camera_blocked(image: Any) -> bool:
    if image is None or image.size == 0:
        return True
    mean_value = float(image.mean())
    std_value = float(image.std())
    return mean_value < 22.0 and std_value < 10.0


def _extract_detections(artifact: Any) -> List[Dict[str, Any]]:
    if not isinstance(artifact, dict):
        return []
    detections = artifact.get("detections")
    return detections if isinstance(detections, list) else []


def _bbox_area(detection: Dict[str, Any]) -> float:
    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return 0.0
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    try:
        width = max(0.0, float(right) - float(left))
        height = max(0.0, float(bottom) - float(top))
    except (TypeError, ValueError):
        return 0.0
    return width * height


def _closest_detection(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [item for item in detections if _bbox_area(item) > 0]
    if not valid:
        return None
    return max(valid, key=_bbox_area)


def _detection_center_x(detection: Dict[str, Any]) -> Optional[float]:
    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        return (float(bbox[0]) + float(bbox[2])) / 2.0
    except (TypeError, ValueError):
        return None


def _sort_detections_left_to_right(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(item) -> Tuple[int, float]:
        index, detection = item
        center_x = _detection_center_x(detection)
        if center_x is None:
            return (1, float(index))
        return (0, center_x)

    indexed = list(enumerate(detections))
    return [detection for _, detection in sorted(indexed, key=key_fn)]


def _clock_direction_from_bbox(detection: Dict[str, Any], frame_width: int) -> str:
    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4 or frame_width <= 0:
        return "straight ahead"
    try:
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    except (TypeError, ValueError):
        return "straight ahead"
    offset = (center_x - (frame_width / 2.0)) / frame_width
    if abs(offset) <= STRAIGHT_AHEAD_THRESHOLD:
        return "straight ahead"
    if offset > STRONG_SIDE_THRESHOLD:
        return "3 o'clock"
    if offset > SLIGHT_SIDE_THRESHOLD:
        return "2 o'clock"
    if offset > 0:
        return "1 o'clock"
    if offset < -STRONG_SIDE_THRESHOLD:
        return "9 o'clock"
    if offset < -SLIGHT_SIDE_THRESHOLD:
        return "10 o'clock"
    return "11 o'clock"


def _limit_words(text: str, max_words: int = 15) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    shortened = " ".join(words[:max_words]).rstrip(".,;:")
    return f"{shortened}."


def _join_with_and(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _format_direction_summary(directions: List[str]) -> str:
    if not directions:
        return "straight ahead"

    direction_counts = Counter(directions)
    ordered_directions = [
        "9 o'clock",
        "10 o'clock",
        "11 o'clock",
        "straight ahead",
        "1 o'clock",
        "2 o'clock",
        "3 o'clock",
    ]

    parts: List[str] = []
    for direction in ordered_directions:
        count = direction_counts.get(direction, 0)
        if count <= 0:
            continue
        if count == 1:
            parts.append(direction)
        elif direction == "straight ahead":
            parts.append(f"{count} straight ahead")
        else:
            parts.append(f"{count} at {direction}")
    return _join_with_and(parts)


def _build_output(count: int, directions: List[str]) -> str:
    direction_summary = _format_direction_summary(directions)
    if count == 1:
        if direction_summary == "straight ahead":
            return "One plug point visible, straight ahead."
        return f"One plug point visible at {direction_summary}."
    return f"{count} plug points visible: {direction_summary}."


def main(image, input_data=None):
    params = input_data if isinstance(input_data, dict) else {}
    is_streaming = bool(params.get("is_streaming") or params.get("live_mode"))
    blocked_frame_threshold = int(params.get("blocked_frame_threshold", 3))

    if image is None:
        return {"audio": {"type": "error", "text": "No camera frame available."}, "text": "No camera frame available."}

    if _is_camera_blocked(image):
        _set_blocked_frame_streak(_get_blocked_frame_streak() + 1)
        if _get_blocked_frame_streak() >= blocked_frame_threshold:
            warning_text = "Camera blocked, please clear the lens."
            return {
                "audio": {"type": "warning", "text": warning_text, "interrupt": True},
                "text": warning_text,
            }
        return ""
    _set_blocked_frame_streak(0)

    try:
        stage_one = copilot_llm_call(
            capability="object_detection_localization",
            goal="Locate visible electrical wall plug points and provide bounding boxes.",
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "detect electrical outlets and plug points for audio navigation guidance",
                "target_labels": ["electrical outlet", "wall outlet", "plug socket", "power socket"],
            },
        )
    except (RuntimeError, ValueError, TypeError):
        LOGGER.debug("Plug point finder detection call failed", exc_info=True)
        error_text = "I couldn't check for plug points right now."
        return {"audio": {"type": "error", "text": error_text}, "text": error_text}

    detections = _extract_detections(stage_one.get("artifact"))
    if not detections:
        return "No plug points found."

    frame_width = int(image.shape[1]) if hasattr(image, "shape") and len(image.shape) >= 2 else 0
    ordered_detections = _sort_detections_left_to_right(detections)
    directions = [_clock_direction_from_bbox(detection, frame_width) for detection in ordered_detections]
    message = _build_output(len(detections), directions)

    if is_streaming and len(message.split()) > 15:
        return _limit_words(message, max_words=15)
    return message
