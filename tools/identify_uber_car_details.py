"""
Identify Uber Car Details Tool

Extracts vehicle make, model, color, and license plate number from a scene to help
blind and low-vision users identify their Uber car.

Features:
- Uses Gemini vision model for intelligent vehicle identification
- Extracts make, model, color, and plate number
- Handles partial information when plate is unreadable
- Audio-optimized output for text-to-speech
- Concise, speech-friendly format

Example Output:
"White Toyota Camry with license plate ABC123"

When plate is unreadable:
"White Toyota Camry, license plate not readable"
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "identify_uber_car_details"
TOOL_PROMPT = (
    "Identify the vehicle at the center of the scene for a blind or low-vision user "
    "waiting for their Uber. Extract the make, model, color, and license plate number. "
    "If the license plate is reflective, obscured, or unreadable, report the other "
    "details you can identify and explicitly state which information cannot be read. "
    "Return one concise user-facing response suitable for spoken output. Prefer one or "
    "two short sentences in a single paragraph with no unnecessary line breaks. State "
    "only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify a vehicle clearly from this view."


async def analyze_image(image):
    """
    Analyze the image to extract vehicle details.

    Args:
        image: Camera frame (PIL Image or numpy array)

    Returns:
        String with vehicle details suitable for speech
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
    Handle Take Photo mode - analyze a single captured image.

    Args:
        runtime: Tool runtime instance
        image: Captured camera frame
        input_data: Optional runtime input data

    Returns:
        String with vehicle identification results
    """
    if image is None:
        return FALLBACK_TEXT

    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """
    Handle streaming mode initialization.

    Args:
        runtime: Tool runtime instance
        input_data: Optional runtime input data
    """
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """
    Handle streaming mode - process each incoming frame.

    Args:
        runtime: Tool runtime instance
        frame: Current camera frame
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
    Handle streaming mode cleanup.

    Args:
        runtime: Tool runtime instance
    """
    runtime.clear_state()
