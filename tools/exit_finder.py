"""Exit Finder Tool

Locate the nearest exit door and provide step-by-step directions for blind and
low-vision users to reach it safely in unfamiliar indoor environments.

The tool identifies exit doors, distinguishes them from restricted or emergency-only
exits by checking for signs or notices, and provides concise spoken navigation
instructions using clock-face directions and body-relative cues.
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "For a blind or low-vision user, identify the nearest visible exit door in this "
    "indoor environment. Look for exit signs, door markings, or doorway features that "
    "indicate an exit. Check for any signs or notices that indicate the exit is "
    "restricted, emergency-only, or should not be used under normal conditions. "
    "Identify reliable exit evidence, choose the most likely reachable exit, determine "
    "its observable clock-face direction (use only forward-facing positions 9, 10, 11, "
    "12, 1, 2, 3 o'clock), estimate approximate distance if supportable, and give one "
    "concise movement instruction. Use body-relative cues like left, right, straight "
    "ahead, or clock positions. Do not rely on color or other visual-only landmarks as "
    "the sole navigation cue. If no exit is visible or the view is unclear, say so. "
    "Return one concise user-facing response suitable for spoken output. Prefer one or "
    "two short sentences in a single paragraph with no unnecessary line breaks. State "
    "only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify an exit from this view."


async def analyze_image(image):
    """Analyze image to find nearest exit and provide navigation guidance."""
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
    """Handle Take Photo mode: analyze single frame for exit location."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Handle streaming mode: analyze current frame for exit location."""
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
