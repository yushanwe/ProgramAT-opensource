"""
Plug Point Detector

Helps blind and low-vision users locate electrical outlets using object detection
and clock-face spatial navigation.

Scenarios:
- No outlets visible: audio confirmation that none are in frame
- Single outlet: count and clock-face position (e.g., "I see one at 11 o'clock")
- Multiple outlets: count and direction to the nearest one

Navigation uses clock-face positions restricted to 1-3 (right) and 9-12 (left/ahead),
as positions 4-8 would be behind the camera.

Live Mode: re-runs every ~5 seconds; streaming output capped at 15 words.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

import cv2
import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

from model_router_client import copilot_llm_call

TOOL_NAME = "plug_point_detector"
STREAMING_WORD_LIMIT = 15

# Only clock positions visible from the camera (front-facing hemisphere)
VALID_CLOCK_POSITIONS = [9, 10, 11, 12, 1, 2, 3]

OUTLET_LABELS = [
    "electrical outlet", "power outlet", "wall outlet", "power socket",
    "electrical socket", "plug socket", "power point", "outlet", "socket",
    "GFCI outlet", "switched outlet",
]


def _image_to_data_uri(image: np.ndarray) -> str:
    """Convert a BGR numpy array to a base64 data URI for LLM calls."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if PILImage is not None:
        pil_img = PILImage.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _truncate_to_words(text: str, limit: int = STREAMING_WORD_LIMIT) -> str:
    """Truncate to at most `limit` words while keeping the result readable."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def main(image: np.ndarray, input_data: Optional[Any] = None) -> Any:
    """
    Detect electrical outlets and report their count and position.

    Args:
        image: Camera frame as BGR numpy array (may be None).
        input_data: Optional dict; may contain 'streaming' (bool).

    Returns:
        Audio-friendly string or dict with 'audio'/'text' keys.
    """
    if image is None or (hasattr(image, "size") and image.size == 0):
        return "No camera image available."

    if input_data is None:
        input_data = {}
    is_streaming = bool(input_data.get("streaming", False))

    try:
        image_uri = _image_to_data_uri(image)

        # ── Stage 1: detect and localise outlets ──────────────────────────────
        detection_result = copilot_llm_call(
            capability="object_detection_localization",
            goal="Detect and count all visible electrical outlets or power sockets in the image.",
            images=[image_uri],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "detect electrical outlets and power sockets with bounding boxes",
                "target_labels": OUTLET_LABELS,
            },
        )

        detection_artifact = detection_result.get("artifact") or {}
        detection_response = detection_result.get("response", "")

        # ── Stage 2: spatial reasoning → clock-face positions ─────────────────
        spatial_result = copilot_llm_call(
            capability="spatial_reasoning",
            goal=(
                "Based on the detected electrical outlets, determine how many are visible "
                "and provide the clock-face position of each outlet. "
                "Use only clock positions 9, 10, 11, 12, 1, 2, or 3 "
                "(positions 4-8 are behind the camera and must never be used). "
                "If there are multiple outlets, identify which one is nearest (largest or most central). "
                "If no outlets are detected, confirm that none are visible. "
                "If obstructions (cords, furniture) cover an outlet, note that briefly. "
                "If the outlet type is identifiable (e.g. GFCI, switched), include it. "
                "Reply in one or two short sentences suitable for audio output."
            ),
            images=[image_uri],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "clock-face spatial position of detected electrical outlets",
                "previous_stage_artifact": detection_artifact,
                "detection_response": detection_response,
            },
        )

        response = spatial_result.get("response", "").strip()

        if not response:
            response = "No electrical outlets detected in the current view."

        if is_streaming:
            response = _truncate_to_words(response, STREAMING_WORD_LIMIT)

        return response

    except Exception as exc:  # noqa: BLE001 — per tools/CLAUDE.md tools must never raise; all errors must be caught and returned as audio
        error_msg = "Could not check for outlets right now."
        if is_streaming:
            return error_msg
        return {
            "audio": {"type": "error", "text": error_msg},
            "text": f"Plug point detector error: {exc}",
        }
