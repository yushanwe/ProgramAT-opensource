"""
Exit Finder Tool for Blind and Low-Vision Users

Locates the nearest standard exit door in unfamiliar indoor environments and provides
step-by-step directions to reach it safely.

Features:
- Identifies standard exit doors (excludes restricted/maintenance doors)
- Provides clock-face navigation (9-12, 1-3 o'clock)
- Estimates approximate distance when supportable
- Speech-friendly output for blind and low-vision users
- Works in streaming and take photo modes

Example Usage:
User activates the tool in an unfamiliar building. The tool detects and reports:
"The nearest exit is 15 feet ahead, through the double doors on your left."
or "Exit at 2 o'clock, walk forward."
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "You are helping a blind or low-vision user locate the nearest exit in an unfamiliar "
    "indoor environment. Analyze the image and:\n"
    "1. Identify all visible standard exit doors. Exit doors typically have exit signs, "
    "push bars, or are clearly marked as emergency exits. Exclude restricted-access doors "
    "or maintenance doors that have notices, warnings, or restricted access signs on them.\n"
    "2. Select the nearest, most reachable standard exit door.\n"
    "3. Determine the exit's direction using clock-face positions (9-12 o'clock for left side, "
    "12 o'clock for straight ahead, 1-3 o'clock for right side). Do not use positions 4-8 as "
    "they would be behind the camera.\n"
    "4. Provide one concise navigation instruction with the clock direction, approximate "
    "distance if reliably estimable from visual cues, and any helpful landmarks like "
    "'through the double doors' or 'past the desk'.\n\n"
    "If no clear exit is visible, say that you cannot identify an exit from this view.\n"
    "If the scene is too dark, blurry, or obstructed to make a reliable judgment, say that "
    "the view is unclear.\n\n"
    "Return one concise user-facing response suitable for spoken output. Prefer one or two "
    "short sentences in a single paragraph with no unnecessary line breaks. State only the "
    "information needed for navigation."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify an exit from this view."


async def analyze_image(image):
    """
    Analyze image to find the nearest exit and provide navigation instructions.

    Args:
        image: Decoded frame image

    Returns:
        Speech-friendly navigation instruction string
    """
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
    """
    Handle Take Photo mode: analyze single captured image.

    Args:
        runtime: Runtime interface with state and emit methods
        image: Captured image
        input_data: Optional input data (unused for this tool)

    Returns:
        Navigation instruction string
    """
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """
    Handle streaming mode initialization.

    Args:
        runtime: Runtime interface
        input_data: Optional input data (unused for this tool)
    """
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """
    Handle incoming frame in streaming mode.

    Args:
        runtime: Runtime interface with state and emit methods
        frame: Frame object with image attribute
    """
    if runtime.is_cancelled():
        return

    # Prevent overlapping analysis
    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)  # Claim before the first await

    try:
        text = await analyze_image(frame.image)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """
    Handle streaming mode cleanup.

    Args:
        runtime: Runtime interface
    """
    runtime.clear_state()
