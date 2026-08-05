"""
Identify Uber Car Details Tool

Extracts vehicle make, model, color, and license plate number from a scene to help
blind and low-vision users identify their Uber car. Optionally accepts expected car
characteristics to assess match likelihood.

Features:
- Uses Gemini vision model for intelligent vehicle identification
- Extracts make, model, color, and plate number
- Compares visible car against expected characteristics when provided
- Provides likelihood assessment for Uber car matching
- Handles partial information when plate is unreadable
- Audio-optimized output for text-to-speech
- Concise, speech-friendly format

Example Output (without expected characteristics):
"White Toyota Camry with license plate ABC123"

Example Output (with expected characteristics):
"This is likely your car - white Toyota matches your description"
"This is not your car - I see a black Honda but you're looking for a white Toyota"

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
TOOL_RUNTIME_INPUT = {
    "key": "expected_car",
    "label": "Expected car (optional)",
    "placeholder": "e.g., white Toyota Camry",
    "prompt_instruction": (
        "The user is looking for: {value}. First identify the visible vehicle's make, "
        "model, and color. Then compare it against the expected car and assess the "
        "match likelihood. If it's a clear match, say 'This is likely your car' and "
        "briefly confirm the matching details. If it's clearly not a match, say 'This "
        "is not your car' and explain what you see versus what they're looking for. "
        "If you're uncertain, provide the details you can see and note the uncertainty."
    ),
}

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify a vehicle clearly from this view."


async def analyze_image(image, expected_car=None):
    """
    Analyze the image to extract vehicle details and optionally match against expected car.

    Args:
        image: Camera frame (PIL Image or numpy array)
        expected_car: Optional string describing expected car characteristics

    Returns:
        String with vehicle details and match assessment suitable for speech
    """
    # Build the prompt based on whether we have expected car info
    if expected_car:
        # Use the runtime input instruction with substitution
        prompt = TOOL_PROMPT + " " + TOOL_RUNTIME_INPUT["prompt_instruction"].format(
            value=expected_car
        )
    else:
        prompt = TOOL_PROMPT

    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
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

    # Extract expected car characteristics if provided
    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])

    return await analyze_image(image, expected_car)


async def on_stream_start(runtime, input_data):
    """
    Handle streaming mode initialization.

    Args:
        runtime: Tool runtime instance
        input_data: Optional runtime input data
    """
    # Extract and store expected car characteristics if provided
    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])
    runtime.set_state("expected_car", expected_car)
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
        # Retrieve expected car characteristics from state
        expected_car = runtime.get_state("expected_car")
        text = await analyze_image(frame.image, expected_car)
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
