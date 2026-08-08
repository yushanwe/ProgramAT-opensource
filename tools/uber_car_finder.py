"""Uber Car Finder for blind and low-vision users."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_car_finder"
TOOL_PROMPT = (
    "Identify the car at the center of this image. Report its make, model, color, "
    "and license plate number in one concise spoken paragraph. If the car is "
    "partially obscured, occluded, or if lighting makes the plate unreadable, "
    "explicitly state what cannot be determined. If no car is clearly visible at "
    "the center, say so. Return one concise user-facing response suitable for "
    "spoken output. Prefer one or two short sentences in a single paragraph with "
    "no unnecessary line breaks. State only the information needed for the user's task."
)
TOOL_RUNTIME_INPUT = {
    "key": "expected_car",
    "label": "Expected car details",
    "placeholder": "e.g., black Honda Accord",
    "prompt_instruction": (
        "After identifying the car, compare it to the user's expected car: {value}. "
        "State whether this is likely their car or not, and why."
    ),
}

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify the car from this view."


async def analyze_car(image, expected_car=None):
    """Analyze car in image and optionally compare to expected characteristics."""
    effective_prompt = TOOL_PROMPT
    if expected_car:
        effective_prompt = (
            f"{TOOL_PROMPT}\n\n"
            f"{TOOL_RUNTIME_INPUT['prompt_instruction'].format(value=expected_car)}"
        )

    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": effective_prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Handle Take Photo mode: analyze the car once."""
    if image is None:
        return FALLBACK_TEXT

    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])

    return await analyze_car(image, expected_car)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])
    runtime.set_state("expected_car", expected_car)
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Handle streaming mode: analyze each new frame."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)
    try:
        expected_car = runtime.get_state("expected_car")
        text = await analyze_car(frame.image, expected_car)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
