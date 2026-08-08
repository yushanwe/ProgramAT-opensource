"""Nearest Exit Finder - Locates the nearest exit and provides navigation directions.

For blind and low-vision users navigating indoor environments, this tool identifies
the nearest reachable exit door or exit sign and provides step-by-step directions
using clock-face navigation (9-12 and 1-3 o'clock) and body-relative cues.

The tool looks for:
- Exit signs (illuminated or visible "EXIT" signage)
- Doors marked or labeled as exits
- Doors that appear to lead outside or to stairwells
- Emergency exit indicators

Navigation uses observable forward directions only, avoiding color-only landmarks.
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "nearest_exit_finder"
TOOL_PROMPT = (
    "For a blind or low-vision user in an indoor space, identify any visible exit "
    "signs, exit doors, or doors that clearly lead outside or to emergency stairwells. "
    "From what is visible, determine which exit appears nearest and most directly "
    "reachable. Give one concise spoken instruction with: the clock-face direction "
    "(use only 9-12 for left side, 1-3 for right side, or straight ahead for 12), "
    "approximate distance cue (very close, close, medium, or far), and a short movement "
    "instruction (reach forward, move forward, walk forward, or walk ahead). "
    "Do not use color, brightness, or person-based landmarks as the sole navigation cue. "
    "If no exit is visible or the scene is unclear, say so plainly."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "No exit is visible from this view."


async def analyze_image(image):
    """Analyze image to find nearest exit and generate navigation instruction."""
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
    """One-shot exit finding from a single captured photo."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Process each frame to locate nearest exit and provide navigation."""
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
