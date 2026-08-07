"""Exit Finder Tool for Navigation Assistance

Locates the nearest exit door and provides step-by-step audio directions
for blind and low-vision users in unfamiliar indoor environments.

This tool identifies exit doors, distinguishes standard exits from
restricted/emergency-only doors based on signage, and provides concise
spoken navigation guidance using body-relative and clock-face directions.
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "You are assisting a blind or low-vision user who needs to find an exit. "
    "Examine this image and identify any visible exit doors, doorways, or exit signs. "
    "Distinguish between standard accessible exits and restricted-access or emergency-only exits "
    "by checking for warning signs, 'Emergency Only', 'Alarm Will Sound', or similar notices. "
    "For the nearest accessible exit, determine its observable direction using clock-face positions "
    "(1-3 o'clock for right, 9-12 o'clock for left and forward, 12 o'clock for straight ahead) "
    "and estimate approximate distance if supportable from visual cues. "
    "Provide one concise step-by-step navigation instruction toward that exit. "
    "Do not use color, people, or positions behind the camera as the sole navigation cue. "
    "If no exit is visible or all exits appear restricted, state that clearly. "
    "Return one concise user-facing response suitable for spoken output. "
    "Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. "
    "State only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot locate a clear exit from this view."


async def analyze_image(image):
    """Analyze a single frame to locate exit and provide navigation guidance."""
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
    """Handle one-shot Take Photo mode."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Process each frame in streaming mode with in-flight protection."""
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
