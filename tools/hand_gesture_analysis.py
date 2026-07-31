"""Identify hand gestures from static poses or motion sequences."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_analysis"

TOOL_PROMPT = (
    "Assist a blind user by identifying hand gestures or their meanings. "
    "When multiple chronological frames are provided, observe the full motion sequence "
    "from earliest to latest to identify the gesture. Track hand shapes, positions, "
    "orientations, and movement patterns. When only one frame is provided, identify "
    "any clearly recognizable static pose, but state uncertainty about motion-based "
    "gestures. For example, a waving motion may mean 'hello' or 'goodbye,' while "
    "moving one finger side to side may mean 'no' or 'incorrect.' A thumbs-up or "
    "peace sign may be recognizable from a static pose. Report the identified gesture "
    "and its likely meaning in one concise spoken sentence. If no clear gesture is "
    "visible or evidence is insufficient, say so plainly. Do not invent motion from "
    "a single image."
)

WINDOW_SECONDS = 3.0
FALLBACK_TEXT = "No clear gesture detected."


async def on_take_photo(runtime, image, input_data):
    """Analyze one image for static hand poses; report uncertainty about motion."""
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
    text = extract_text(response)
    return (text or FALLBACK_TEXT).strip() or FALLBACK_TEXT


async def on_stream_start(runtime, input_data):
    """Initialize state for this streaming session."""
    del input_data
    runtime.set_state("is_analyzing", False)
    runtime.set_state("last_processed_frame_id", None)
    runtime.set_state("last_result_text", None)


async def on_frame(runtime, frame):
    """Analyze recent chronological frames to identify gestures from motion."""
    if runtime.is_cancelled():
        return

    if runtime.get_state("is_analyzing", False):
        return

    if runtime.get_state("last_processed_frame_id") == frame.frame_id:
        return

    recent_frames = runtime.get_recent_frames(seconds=WINDOW_SECONDS) or [frame]
    ordered_frames = sorted(recent_frames, key=lambda f: f.timestamp)
    images = [f.image for f in ordered_frames if f.image is not None]

    if not images:
        return

    runtime.set_state("is_analyzing", True)
    try:
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            images,
            {"timeout": 30, "num_retries": 0},
        )
        text = extract_text(response)
        text = (text or FALLBACK_TEXT).strip() or FALLBACK_TEXT

        if runtime.is_cancelled():
            return

        if text != runtime.get_state("last_result_text"):
            await runtime.emit(
                text,
                final=True,
                metadata={
                    "frame_id": frame.frame_id,
                    "frame_count": len(images),
                },
            )
            runtime.set_state("last_result_text", text)

        runtime.set_state("last_processed_frame_id", frame.frame_id)
    finally:
        runtime.set_state("is_analyzing", False)


async def on_stream_stop(runtime):
    """Clean up state when streaming stops."""
    runtime.clear_state()
