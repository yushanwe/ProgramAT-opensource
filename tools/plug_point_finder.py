"""
Plug Point Finder Tool

Finds visible electrical plug points, counts them, and gives directional cues.
"""

from __future__ import annotations

import threading
import logging
import re
from typing import Any, Dict, List, Optional

from model_router_client import copilot_llm_call

TOOL_NAME = "plug_point_finder"
LOGGER = logging.getLogger(__name__)
STRAIGHT_AHEAD_THRESHOLD = 0.11
SLIGHT_SIDE_THRESHOLD = 0.20
STRONG_SIDE_THRESHOLD = 0.34
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
NUMBER_TOKEN_PATTERN = rf"(\d+|{'|'.join(re.escape(word) for word in NUMBER_WORDS)})"

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


def _build_output(count: int, direction: str) -> str:
    if count == 1:
        if direction == "straight ahead":
            return "One outlet visible, straight ahead."
        return f"One outlet visible, closest at {direction}."
    if direction == "straight ahead":
        return f"{count} outlets visible, closest one straight ahead."
    return f"{count} outlets visible, closest one at {direction}."


def _parse_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, str):
        normalized = value.strip().lower().rstrip(".!?")
        if normalized in NUMBER_WORDS:
            return NUMBER_WORDS[normalized]
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _socket_count_from_response_text(text: str) -> Optional[int]:
    normalized = text.strip().lower()
    if not normalized:
        return None

    direct = re.fullmatch(rf"{NUMBER_TOKEN_PATTERN}[.!?]?", normalized)
    if direct:
        return _parse_positive_int(direct.group(1))

    patterns = [
        rf"\b{NUMBER_TOKEN_PATTERN}\s+(?:individual\s+)?(?:plug\s+points?|plugs?|sockets?|outlets?)\b",
        rf"\b(?:plug\s+points?|plugs?|sockets?|outlets?)[\s:,\-]{{0,8}}{NUMBER_TOKEN_PATTERN}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            parsed = _parse_positive_int(match.group(1))
            if parsed is not None:
                return parsed
    return None


def _count_individual_sockets(image: Any, detections: List[Dict[str, Any]]) -> int:
    detection_count = len(detections)
    if detection_count <= 0:
        return 0

    try:
        stage_two = copilot_llm_call(
            capability="general_reasoning",
            goal="Count the number of individual electrical sockets visible in the image. Return only the total count.",
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "count visible individual sockets instead of outlet strips",
                "previous_stage_artifact": {"detections": detections},
            },
        )
    except (RuntimeError, ValueError, TypeError):
        LOGGER.debug("Socket counting reasoning stage failed, falling back to detection count", exc_info=True)
        return detection_count

    artifact = stage_two.get("artifact") if isinstance(stage_two, dict) else None
    if isinstance(artifact, dict):
        for key in ("socket_count", "count", "total"):
            parsed = _parse_positive_int(artifact.get(key))
            if parsed is not None:
                return parsed

    response_text = stage_two.get("response") if isinstance(stage_two, dict) else ""
    if isinstance(response_text, str):
        parsed = _socket_count_from_response_text(response_text)
        if parsed is not None:
            return parsed

    return detection_count


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
    closest = _closest_detection(detections)
    direction = _clock_direction_from_bbox(closest or {}, frame_width)
    socket_count = _count_individual_sockets(image, detections)
    message = _build_output(socket_count, direction)

    if is_streaming and len(message.split()) > 15:
        return _limit_words(message, max_words=15)
    return message
