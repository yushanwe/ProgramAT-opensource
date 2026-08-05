"""Identify hand gestures from motion sequences over recent frames."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_identification"
TOOL_PROMPT = (
    "Observe the hand movements across these chronological frames from earliest to "
    "latest. Track the position, orientation, and motion of all visible hands. "
    "Identify the gesture or short meaning each person just expressed, such as "
    "waving for hello or goodbye, moving one finger side to side for no or incorrect, "
    "thumbs up for approval, pointing for direction, or other recognizable gestures. "
    "When multiple people or hands are visible, describe the gesture from each person "
    "separately. If only one stable hand pose is visible with no motion, identify it "
    "only if it is a clearly recognizable static gesture. If the gesture is unclear or "
    "no hand movement is visible, say so plainly. Return one concise user-facing "
    "response suitable for spoken output. Prefer one or two short sentences in a "
    "single paragraph with no unnecessary line breaks. State only the information "
    "needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot determine a clear gesture from this view."
RECENT_FRAMES_SECONDS = 3


async def analyze_gesture(images):
    """Analyze gesture from chronological frame sequence."""
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": TOOL_PROMPT}],
        images,
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Analyze a single image for static gesture recognition."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_gesture([image])


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Analyze recent frames to identify hand gesture from motion sequence."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)
    try:
        recent = runtime.get_recent_frames(seconds=RECENT_FRAMES_SECONDS)
        if not recent:
            recent = [frame]
        images = [f.image for f in recent]
        text = await analyze_gesture(images)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
