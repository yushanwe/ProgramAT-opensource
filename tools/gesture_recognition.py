"""Gesture recognition tool that identifies hand gestures from motion over time."""

from __future__ import annotations

import asyncio
from typing import Any

from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_recognition"

TOOL_PROMPT = (
    "Observe the hand movements in the provided image or sequence of images. "
    "When multiple chronological frames are available, track the hand position, "
    "shape, orientation, and motion from earliest to latest to identify the gesture. "
    "When only one image is available, identify any clear static hand posture visible, "
    "but state that motion-based gestures cannot be determined from a single frame. "
    "Common gestures include waving (hello/goodbye), finger side-to-side (no/incorrect), "
    "thumbs up (approval), pointing, stop hand, etc. Return one concise spoken sentence "
    "identifying the gesture or short meaning, for example 'Person is waving hello' or "
    "'Finger moving side to side, means no'. If no clear gesture is visible or evidence "
    "is insufficient, say so plainly."
)

RECENT_FRAMES_COUNT = 15
RECENT_FRAMES_SECONDS = 3.0


async def on_take_photo(runtime, image, input_data):
    """Analyze a single image for static hand postures."""
    del input_data
    if image is None:
        return "No camera image is available."

    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("analyzing_window", None)
    runtime.set_state("last_analysis_timestamp", None)


async def on_frame(runtime, frame):
    """Analyze recent frames for motion-based gesture recognition."""
    if runtime.is_cancelled():
        return

    current_window = runtime.get_state("analyzing_window")
    if current_window is not None and not current_window.done():
        return

    last_timestamp = runtime.get_state("last_analysis_timestamp")
    if last_timestamp is not None:
        time_since_last = frame.timestamp - last_timestamp
        if time_since_last < 2.0:
            return

    recent_frames = runtime.get_recent_frames(seconds=RECENT_FRAMES_SECONDS)
    if not recent_frames or len(recent_frames) < 3:
        return

    frames_to_analyze = recent_frames[-RECENT_FRAMES_COUNT:]
    images = [f.image for f in frames_to_analyze]
    window_id = frame.frame_id

    async def analyze():
        try:
            response = await asyncio.to_thread(
                call_model,
                "gemini/gemini-3.1-flash-lite-preview",
                [{"role": "user", "content": TOOL_PROMPT}],
                images,
                {"timeout": 60, "num_retries": 0},
            )

            if runtime.is_cancelled():
                return

            text = extract_text(response)
            if not text or text.lower().startswith("no clear"):
                return

            await runtime.emit(text, final=True, replace=True, metadata={"window_id": window_id})
            runtime.set_state("last_analysis_timestamp", frame.timestamp)
        finally:
            if runtime.get_state("analyzing_window") is task:
                runtime.set_state("analyzing_window", None)

    task = asyncio.create_task(analyze())
    runtime.set_state("analyzing_window", task)


async def on_stream_stop(runtime):
    """Clean up when streaming stops."""
    current_window = runtime.get_state("analyzing_window")
    if current_window is not None and not current_window.done():
        current_window.cancel()
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
