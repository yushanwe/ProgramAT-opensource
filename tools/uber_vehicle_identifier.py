"""Uber Vehicle Identifier for blind and low-vision users."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_vehicle_identifier"
TOOL_PROMPT = (
    "Identify the car at the center of the scene. Report its make, model, color, "
    "and license plate number if visible. If the car is partially obstructed, the "
    "license plate is obscured by dirt or glare, or the lighting conditions are too "
    "poor for accurate identification, clearly state what is obstructed or unclear. "
    "If no car is visible at the center, say so. Return one concise user-facing "
    "response suitable for spoken output. Prefer one or two short sentences in a "
    "single paragraph with no unnecessary line breaks. State only the information "
    "needed for the user's task."
)
TOOL_RUNTIME_INPUT = {
    "key": "expected_car",
    "label": "Expected car description",
    "placeholder": "Enter your expected Uber (e.g., Black Honda)",
    "prompt_instruction": (
        "After identifying the car, compare it against the user's expected Uber: {value}. "
        "Determine if the identified car matches the expected description and clearly state "
        "whether this is likely their car or not."
    ),
}

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify the car from this view."


async def analyze_image(image, expected_car_description):
    """Analyze the image to identify the car and compare against expected description."""
    # Build the prompt with optional comparison instruction
    prompt = TOOL_PROMPT
    if expected_car_description:
        prompt = (
            TOOL_PROMPT + " " + TOOL_RUNTIME_INPUT["prompt_instruction"].format(
                value=expected_car_description
            )
        )

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
    """Handle take photo mode."""
    if image is None:
        return FALLBACK_TEXT

    # Extract expected car description from input_data
    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])

    return await analyze_image(image, expected_car)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    # Extract and store expected car description
    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])
    runtime.set_state("expected_car", expected_car)
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Handle streaming mode frame analysis."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)
    try:
        expected_car = runtime.get_state("expected_car")
        text = await analyze_image(frame.image, expected_car)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
