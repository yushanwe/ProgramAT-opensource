"""Locate the nearest exit and provide navigation directions for blind users."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "nearest_exit_finder"
TOOL_PROMPT = (
    "Locate the nearest reachable exit door in this indoor environment for a blind or "
    "low-vision user. Look for exit signs, emergency exit markers, or doors with exit "
    "signage. Distinguish genuine exits from restricted or locked doors by checking for "
    "restriction signs or notices. Identify the most accessible exit, determine its "
    "observable clock-face direction (9-12 o'clock on left, 12 straight ahead, 1-3 "
    "o'clock on right), and give one concise navigation instruction toward it using "
    "body-relative and clock-face cues. Do not use color or person-based landmarks as "
    "the sole navigation cue. If no clear exit is visible or the exit is uncertain, say so. "
    "Return one concise user-facing response suitable for spoken output. Prefer one or two "
    "short sentences in a single paragraph with no unnecessary line breaks. State only the "
    "information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "No clear exit is visible from this view."


async def analyze_image(image):
    """Call the vision model to locate exit and provide navigation."""
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Handle one-shot take photo request."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Analyze each frame for nearest exit."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)
    try:
        text = await analyze_image(frame.image)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
