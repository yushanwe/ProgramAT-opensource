"""
Uber Car Identifier Tool

Identifies the make, model, color, and plate number of a car in view and compares
it to the user's search criteria to help blind or low-vision users find their Uber.
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_car_identifier"
TOOL_PROMPT = (
    "Identify the car at the center of this image. Report its make, model, color, "
    "and license plate number if visible. If the car is not clearly visible or "
    "identifiable, say so. Be concise and speak naturally for audio output."
)
TOOL_RUNTIME_INPUT = {
    "key": "search_criteria",
    "label": "Car search criteria",
    "placeholder": "Enter expected car details (e.g., white Toyota Camry, ABC123)",
    "prompt_instruction": (
        "The user is looking for a car with these characteristics: {value}. "
        "After identifying the car in the image, compare it to these expected "
        "characteristics and state whether this is likely their car. If it matches, "
        "confirm it. If it doesn't match, explain what differs and estimate how "
        "likely this car is to be their Uber."
    ),
}

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "The car is not clearly visible in this view."


async def analyze_car(image, search_criteria=None):
    """Analyze car in image and optionally compare to search criteria."""
    # Build the prompt based on whether search criteria is provided
    if search_criteria and search_criteria.strip():
        prompt = TOOL_PROMPT + " " + TOOL_RUNTIME_INPUT["prompt_instruction"].format(
            value=search_criteria
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
    """Handle Take Photo mode: single image analysis."""
    del runtime
    if image is None:
        return FALLBACK_TEXT

    search_criteria = None
    if input_data and isinstance(input_data, dict):
        search_criteria = input_data.get("search_criteria")

    return await analyze_car(image, search_criteria)


async def on_stream_start(runtime, input_data):
    """Initialize streaming session state."""
    # Extract and store search_criteria from input_data
    search_criteria = None
    if input_data and isinstance(input_data, dict):
        search_criteria = input_data.get("search_criteria")

    runtime.set_state("search_criteria", search_criteria)
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Handle Streaming mode: analyze latest frame when not busy."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)
    try:
        # Get search criteria from input_data if available
        search_criteria = runtime.get_state("search_criteria")

        text = await analyze_car(frame.image, search_criteria)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming session state."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
