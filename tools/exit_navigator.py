"""
Exit Navigator Tool for Blind Users

Locates the nearest exit door or exit sign and provides step-by-step directions
to help blind and low-vision users reach it safely in indoor environments.

Features:
- Detects exit signs and exit doors using vision model
- Provides accessible navigation instructions using clock-face directions
- Audio-optimized output for text-to-speech
- Streaming and Take Photo modes

Navigation System:
- Uses clock-face positions (1-3 o'clock on right, 9-12 o'clock on left)
- Positions 4-8 not used as they would be behind the camera
- Avoids color-only or purely visual landmarks
- Provides approximate distance and movement instructions

Example Output:
"The nearest exit is straight ahead at 12 o'clock, approximately 20 feet away; proceed forward."
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_navigator"
TOOL_PROMPT = (
    "You are assisting a blind or low-vision user who needs to locate and navigate to "
    "the nearest exit. Analyze the image to identify exit signs, exit doors, or doorways "
    "that appear to be exits. "
    "For the nearest or most prominent exit you can identify: "
    "1. Determine its position using clock-face direction (12 o'clock is straight ahead, "
    "1-3 o'clock is to the right, 9-11 o'clock is to the left; do not use positions "
    "4-8 as they would be behind the user). "
    "2. Estimate the approximate distance if possible from visual cues (for example, "
    "very close if it fills much of the view, far if it is small or distant). "
    "3. Provide one concise navigation instruction such as 'proceed forward', 'turn right', "
    "or 'move slightly left' based on the clock-face position. "
    "If multiple exits are visible, focus on the nearest or most accessible one. "
    "If no exit is visible or identifiable, say that no exit can be detected from this view. "
    "Do not rely on color or person-based landmarks. "
    "Return one concise user-facing response suitable for spoken output. Prefer one or two "
    "short sentences in a single paragraph with no unnecessary line breaks. State only the "
    "information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "No exit is visible from this view."


async def analyze_image(image):
    """
    Analyze a single image to locate exits and provide navigation guidance.

    Args:
        image: Camera frame (PIL Image or similar)

    Returns:
        String with navigation instruction or fallback text
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
    Handle Take Photo mode: single image analysis.

    Args:
        runtime: Tool runtime context
        image: Camera frame
        input_data: Optional runtime input (not used for this tool)

    Returns:
        Navigation instruction string
    """
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """
    Initialize streaming mode state.

    Args:
        runtime: Tool runtime context
        input_data: Optional runtime input (not used for this tool)
    """
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """
    Handle each frame in streaming mode.

    Args:
        runtime: Tool runtime context
        frame: Frame object with image attribute
    """
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
    """
    Clean up streaming mode state.

    Args:
        runtime: Tool runtime context
    """
    runtime.clear_state()
