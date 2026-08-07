"""
Indoor Exit Navigator

Helps blind and low-vision users locate the nearest exit door indoors by providing
distance estimates and clock-face directions.

Features:
- Identifies visible exit doors and doorways in indoor environments
- Provides approximate distance estimates (very close, close, medium, far)
- Uses clock-face directions (1-3, 9-12) for navigation guidance
- Speech-friendly output suitable for text-to-speech
- Handles uncertainty gracefully when exits cannot be identified

Example Output:
"The exit is about 2 meters from you and is in your 12 o'clock direction."
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "indoor_exit_navigator"
TOOL_PROMPT = (
    "For a blind or low-vision user seeking an indoor exit, examine this image and: "
    "1) identify any visible exit doors, exit signs, or doorways that appear to lead outside, "
    "2) select the nearest or most accessible one, "
    "3) estimate its approximate distance in meters based on visible cues, "
    "4) determine its observable clock-face direction (1-3 o'clock on right, 9-12 o'clock on left, 12 straight ahead), "
    "5) provide one concise spoken instruction with distance and direction. "
    "If no exit is visible or identifiable, state that clearly. "
    "Do not use color or other purely visual landmarks as the sole navigation cue. "
    "Return one concise user-facing response suitable for spoken output. "
    "Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. "
    "State only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify any exits from this view."


async def analyze_image(image):
    """Analyze the image to identify exits and provide navigation guidance."""
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
    """Handle one-shot take photo requests."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Process each frame in streaming mode."""
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
