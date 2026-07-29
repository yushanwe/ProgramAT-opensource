"""Recognize hand gestures from camera frames with motion-aware analysis."""

from __future__ import annotations

import asyncio
from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_recognition"
TOOL_PROMPT = (
    "Identify the hand gesture or short meaning expressed by the person. "
    "When multiple chronological frames are available, observe the hand motion sequence "
    "from earliest to latest to recognize dynamic gestures such as waving, pointing, or "
    "sign language. When only one frame is available, identify static hand poses such as "
    "thumbs up, peace sign, or raised hand, and note that motion-based gestures cannot be "
    "determined from a single image. Return one concise sentence describing the gesture "
    "for a blind or low-vision user, such as 'The person is waving hello' or 'Thumbs up gesture.' "
    "If no clear gesture is visible or hands are not in frame, say so plainly."
)


async def on_take_photo(runtime, image, input_data):
    """Analyze a single frame for recognizable static hand poses."""
    del runtime, input_data
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
    """Initialize state for streaming gesture recognition."""
    del input_data
    runtime.set_state("last_gesture", None)
    runtime.set_state("frames_analyzed", 0)


async def on_frame(runtime, frame):
    """Analyze recent frames chronologically to recognize gestures with motion."""
    if runtime.is_cancelled():
        return

    # Get recent frames for motion analysis (last 3 seconds of history)
    recent_frames = runtime.get_recent_frames(seconds=3.0)

    # Only analyze if we have sufficient frame history
    if not recent_frames or len(recent_frames) < 3:
        return

    # Increment frame counter to track analysis frequency
    frames_analyzed = runtime.get_state("frames_analyzed", 0)

    # Analyze every 6th frame to avoid redundant processing (roughly 1-2 times per second)
    if frames_analyzed % 6 != 0:
        runtime.set_state("frames_analyzed", frames_analyzed + 1)
        return

    runtime.set_state("frames_analyzed", frames_analyzed + 1)

    # Select frames chronologically - earliest to latest for motion tracking
    # Use up to 5 frames from the recent history for temporal analysis
    selected_frames = recent_frames[-5:] if len(recent_frames) >= 5 else recent_frames
    images = [f.image for f in selected_frames]

    try:
        # Call model with chronological frame sequence
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            images,
            {"timeout": 60, "num_retries": 0},
        )

        result = extract_text(response)

        # Only emit if gesture changed or it's a meaningful detection
        last_gesture = runtime.get_state("last_gesture")
        if result and result != last_gesture:
            # Avoid emitting negative results repeatedly
            no_gesture_phrases = ["no clear gesture", "no gesture", "hands are not", "not in frame"]
            is_negative = any(phrase in result.lower() for phrase in no_gesture_phrases)

            if not is_negative or last_gesture is not None:
                runtime.set_state("last_gesture", result)
                await runtime.emit(
                    result,
                    final=False,
                    metadata={
                        "frame_id": frame.frame_id,
                        "num_frames_analyzed": len(selected_frames),
                    },
                )
    except Exception as exc:
        # Silently handle errors to avoid disrupting the stream
        # Gesture recognition is best-effort
        pass


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
