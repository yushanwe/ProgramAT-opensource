"""
Exit Finder Tool

Locates the nearest standard exit door and provides step-by-step navigation
directions in unfamiliar indoor environments for blind and low-vision users.

Features:
- Detects standard exit doors and doorways
- Filters out emergency-only exits with alarm warnings
- Provides clock-face navigation (9-12-3) and body-relative directions
- Audio-optimized concise output for text-to-speech
- Single-frame analysis for current exit visibility

Navigation System:
- Uses clock face positions (1-3 o'clock on right, 9-12 o'clock straight/left)
- Body-relative cues: left, right, straight ahead
- Approximate distance when supportable
- No color-only or visual-exclusive landmarks

Example Output:
"The nearest exit is 15 feet ahead; proceed forward through the double doors."
"Exit door at 10 o'clock on your left, approximately 20 feet away."
"No visible exit from this view."
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "You are assisting a blind or low-vision user who needs to find the nearest "
    "standard exit in an indoor environment. Analyze this image and:\n"
    "1. Identify all visible exits, doorways, or exit signs. Look for exit signage, "
    "doors leading outside, or clear pathways to exits.\n"
    "2. Distinguish standard exits from emergency-only exits. Emergency exits often "
    "have warning signs about alarms. Focus only on standard exits that can be used "
    "without triggering alarms.\n"
    "3. Choose the nearest, most accessible standard exit that appears reachable.\n"
    "4. Determine its direction using clock-face positions (9-12 for left/straight, "
    "1-3 for right) or body-relative terms (left, right, straight ahead).\n"
    "5. Estimate distance if supportable (e.g., close, 10-20 feet, across the room).\n"
    "6. Provide one concise spoken navigation instruction. Do not use color or other "
    "purely visual landmarks as the sole navigation cue.\n"
    "\n"
    "If no exit is visible or the view is unclear, say so plainly. If you can only "
    "see emergency exits with alarm warnings, explain that standard exits are not "
    "visible from this view.\n"
    "\n"
    "Return one concise user-facing response suitable for spoken output. Prefer one "
    "or two short sentences in a single paragraph with no unnecessary line breaks. "
    "State only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify a clear exit from this view."


async def analyze_image(image):
    """Call the vision model to identify and navigate to the nearest standard exit."""
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
    """Handle Take Photo mode: analyze one captured image for exit navigation."""
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Handle Streaming mode: analyze each frame for exit navigation."""
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
