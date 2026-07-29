"""Recognize hand gestures and their meaning from camera frames."""

from __future__ import annotations

import asyncio
import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_recognition"
TOOL_PROMPT = (
    "Observe hand movements and identify the gesture or its meaning. "
    "When multiple chronological frames are available, inspect them from earliest "
    "to latest to track hand motion, direction, and changes in hand shape or position. "
    "When only one image is available, describe only the visible static hand pose "
    "and state that motion-based gestures cannot be determined from a single frame. "
    "Identify gestures such as waving (hello or goodbye), finger pointing, thumbs up, "
    "thumbs down, stop sign (palm facing camera), peace sign, OK sign, or finger "
    "movement side to side (no or incorrect). "
    "Return one concise sentence describing the gesture and its likely meaning, "
    "suitable for a blind or low-vision user. "
    "If no clear hand gesture is visible or evidence is insufficient, say exactly "
    "'No clear hand gesture detected.'"
)


async def on_take_photo(runtime, image, input_data):
    """Analyze a single frame for static hand pose."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."

    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 30, "num_retries": 0},
    )
    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("last_result_time", 0)
    runtime.set_state("analyzing", False)


async def on_frame(runtime, frame):
    """Process recent frames to identify gestures that may require motion."""
    if runtime.is_cancelled():
        return

    # Avoid overlapping analysis
    if runtime.get_state("analyzing", False):
        return

    # Throttle to analyze every 2 seconds
    current_time = frame.timestamp
    last_time = runtime.get_state("last_result_time", 0)
    if current_time - last_time < 2.0:
        return

    runtime.set_state("analyzing", True)
    runtime.set_state("last_result_time", current_time)

    try:
        # Get recent frames (last 3 seconds) for motion detection
        recent_frames = runtime.get_recent_frames(seconds=3)

        if not recent_frames:
            return

        # Use chronological order for motion analysis
        images = [f.image for f in recent_frames]

        # Call model with multiple frames in chronological order
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            images,
            {"timeout": 30, "num_retries": 0},
        )

        if runtime.is_cancelled():
            return

        result = extract_text(response)

        # Only emit if we have a meaningful result
        if result and "no clear hand gesture" not in result.lower():
            await runtime.emit(result, final=True)

    except Exception as e:
        # Silently handle errors to avoid disrupting the stream
        pass
    finally:
        runtime.set_state("analyzing", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
