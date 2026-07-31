"""Hand gesture recognition tool for identifying gestures from hand movements over time."""

from __future__ import annotations

import asyncio
import logging
import time

from litellm_utils import call_model, extract_text

logger = logging.getLogger(__name__)

TOOL_NAME = "hand_gesture_recognition"

TOOL_PROMPT = (
    "Identify the hand gesture or short meaning the person just expressed. "
    "When multiple chronological frames are available, analyze the hand motion "
    "sequence from earliest to latest to identify gestures defined by movement, "
    "such as waving (hello or goodbye), moving one finger side to side (no or incorrect), "
    "thumbs up (approval), or other common gestures. "
    "When only one image is available, identify only static hand poses that are "
    "recognizable without motion, such as a peace sign, thumbs up, or pointing. "
    "Do not invent motion from a single image. "
    "Return one concise spoken sentence for a blind user, such as 'The person is waving hello' "
    "or 'Thumbs up gesture.' "
    "If no clear gesture is visible or the hands are not in view, state 'No gesture detected.'"
)

# Continuous temporal analysis configuration
WINDOW_SECONDS = 4.0  # Analyze recent 4 seconds of motion
INTERVAL_SECONDS = 2.0  # Analyze every 2 seconds
MAX_SAMPLE_FRAMES = 8  # Send up to 8 representative frames per analysis

FALLBACK_TEXT = "No gesture detected."


def sample_evenly(frames: list, max_frames: int) -> list:
    """Sample frames evenly across the chronological sequence."""
    if not frames or max_frames <= 0:
        return []
    if len(frames) <= max_frames:
        return frames

    # Evenly sample across the full range
    indices = [int(i * len(frames) / max_frames) for i in range(max_frames)]
    return [frames[i] for i in indices]


async def on_take_photo(runtime, image, input_data):
    """Analyze a single frame for static hand poses only."""
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

    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_stream_start(runtime, input_data):
    """Initialize streaming state for continuous gesture analysis."""
    del input_data

    now = time.monotonic()
    runtime.set_state("analysis_in_flight", False)
    runtime.set_state("next_analysis_at", now + WINDOW_SECONDS)


async def on_frame(runtime, frame):
    """Continuously analyze hand gestures from recent frame history."""
    if runtime.is_cancelled():
        return

    now = time.monotonic()
    next_analysis_at = runtime.get_state("next_analysis_at")

    # Wait until the next scheduled analysis time
    if now < next_analysis_at:
        return

    # Prevent duplicate analysis if one is already running
    if runtime.get_state("analysis_in_flight", False):
        return

    # Get recent frames and sort chronologically
    recent_frames = runtime.get_recent_frames(seconds=WINDOW_SECONDS) or []
    ordered_frames = sorted(recent_frames, key=lambda candidate: candidate.timestamp)

    # Sample evenly across the time window
    selected_frames = sample_evenly(ordered_frames, max_frames=MAX_SAMPLE_FRAMES)
    images = [candidate.image for candidate in selected_frames if candidate.image is not None]

    # Advance next analysis time by the interval
    while next_analysis_at <= now:
        next_analysis_at += INTERVAL_SECONDS
    runtime.set_state("next_analysis_at", next_analysis_at)

    if not images:
        return

    runtime.set_state("analysis_in_flight", True)
    try:
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            images,
            {"timeout": 60, "num_retries": 0},
        )

        text = extract_text(response).strip() or FALLBACK_TEXT

        if runtime.is_cancelled():
            return

        await runtime.emit(
            text,
            final=True,
            metadata={
                "frame_id": frame.frame_id,
                "frame_count": len(images),
                "window_seconds": WINDOW_SECONDS,
            },
        )
    except Exception:
        logger.exception("Hand gesture analysis failed")
    finally:
        runtime.set_state("analysis_in_flight", False)


async def on_stream_stop(runtime):
    """Clean up state when streaming stops."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
