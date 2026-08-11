"""Identify car characteristics and compare against expected Uber vehicle."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_vehicle_identifier"
TOOL_PROMPT = (
    "Identify the make, model, color, and license plate number of the car at the "
    "center of this image. If the user has provided expected car characteristics, "
    "compare the identified car against those expected characteristics and clearly "
    "state whether this appears to be the expected car. If the car is partially "
    "obstructed, identify what is visible and note the obstruction. If no car is "
    "visible at the center, say so. Return one concise user-facing response suitable "
    "for spoken output. Prefer one or two short sentences in a single paragraph with "
    "no unnecessary line breaks. State only the information needed for the user's task."
)
TOOL_RUNTIME_INPUT = {
    "key": "expected_car",
    "label": "Expected car characteristics",
    "placeholder": "e.g., Black Honda or white Toyota Camry",
    "prompt_instruction": (
        "The user is expecting a car with these characteristics: {value}. "
        "After identifying the car at the center of the scene, compare it against "
        "these expected characteristics and clearly state whether this is likely "
        "the expected car or not."
    ),
}

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify a car clearly from this view."


async def analyze_image(image, expected_car=None):
    """Identify car and compare against expected characteristics if provided."""
    prompt = TOOL_PROMPT
    if expected_car:
        prompt = (
            f"{TOOL_PROMPT}\n\n"
            f"{TOOL_RUNTIME_INPUT['prompt_instruction'].format(value=expected_car)}"
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
    """Handle Take Photo mode with optional expected car characteristics."""
    if image is None:
        return FALLBACK_TEXT

    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])

    return await analyze_image(image, expected_car)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state with expected car characteristics."""
    expected_car = None
    if isinstance(input_data, dict):
        expected_car = input_data.get(TOOL_RUNTIME_INPUT["key"])
    runtime.set_state("expected_car", expected_car)
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Process each frame to identify car and compare with expected characteristics."""
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
