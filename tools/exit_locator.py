"""
Exit Locator Tool for Blind Users

Locates the nearest exit door and provides step-by-step navigation directions
to help blind and low-vision users reach it safely.

Features:
- Uses VLM to identify exit evidence (exit signs, doors, open passages)
- Provides clock-face navigation directions (1-3, 9-12 o'clock)
- Gives concise movement instructions
- Prioritizes exits with visible exit signs or clear egress indicators
- Audio-optimized output for text-to-speech
- Streaming mode provides continuous guidance as user walks around

Navigation System:
- Uses clock face positions (1-3 o'clock on right, 9-12 o'clock on left)
- Positions 4-8 not used as they would be behind the camera
- Provides body-relative spatial cues that work without vision
- Avoids color-only or purely visual landmarks

Example Usage:
User activates tool while in an unfamiliar indoor space. The tool detects
exit evidence and reports: "Exit straight ahead, walk forward"

This tool runs on the backend server and receives camera frames from the mobile app.
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_locator"
TOOL_PROMPT = (
    "For a blind or low-vision user in an indoor space, identify reliable exit evidence "
    "such as exit signs, open doorways with visible egress, or clear passages that appear "
    "to lead outside. Choose the most likely reachable exit, determine its observable "
    "clock-face direction using only forward-facing positions (9 through 12 to 3 o'clock), "
    "and give one concise movement instruction such as 'walk forward,' 'turn toward 2 o'clock,' "
    "or 'move to your left.' If no exit evidence is visible, say 'No clear exit visible from "
    "this view.'"
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite-preview"
FALLBACK_TEXT = "No clear exit visible from this view."


async def analyze_image(image):
    """
    Analyze a single image for exit evidence and provide navigation guidance.

    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)

    Returns:
        Concise navigation instruction string
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
    Take Photo mode: analyze single frame for exit.

    Args:
        runtime: Tool runtime with state management and frame access
        image: Current camera frame
        input_data: Optional configuration dictionary

    Returns:
        Navigation instruction string
    """
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """
    Streaming mode initialization: set up state tracking.

    Args:
        runtime: Tool runtime with state management
        input_data: Optional configuration dictionary
    """
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """
    Streaming mode: analyze each frame for exit evidence.

    Args:
        runtime: Tool runtime with state management and cancellation
        frame: Current camera frame object with .image attribute
    """
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)  # claim before the first await
    try:
        text = await analyze_image(frame.image)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """
    Streaming mode cleanup: clear state when streaming ends.

    Args:
        runtime: Tool runtime with state management
    """
    runtime.clear_state()
