"""Uber Car Finder - identify car details and match to user expectations."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_car_finder"
TOOL_PROMPT = (
    "You are helping a blind or low-vision user identify their Uber car. "
    "Look at the scene and identify any visible cars. For each car, determine: "
    "make, model, color, and license plate number (if visible). "
    "Provide the information concisely in a spoken format. "
    "Focus on the most prominent car in the scene. "
    "If no car is visible or details cannot be determined, say so clearly."
)
TOOL_RUNTIME_INPUT = {
    "key": "expected_car",
    "label": "Expected car details",
    "placeholder": "e.g., white Toyota Camry ABC123 or red or black Honda",
    "prompt_instruction": (
        "The user is looking for a car matching: {value}. "
        "After identifying the visible car details, compare them to this description. "
        "If multiple options are mentioned with 'or' (like 'red or black'), treat any match as correct. "
        "Report whether the visible car matches the user's expected car, is likely their car, "
        "is possibly their car, or is not their car. Be specific about what matches and what doesn't."
    ),
}

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot determine the car details from this view."


async def analyze_car(image, user_input=None):
    """Analyze image to identify car details and optionally compare to user expectation."""
    prompt = TOOL_PROMPT

    if user_input and user_input.strip():
        prompt = TOOL_RUNTIME_INPUT["prompt_instruction"].format(value=user_input)
        prompt = TOOL_PROMPT + " " + prompt

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
    """Handle one-shot Take Photo request."""
    del runtime
    if image is None:
        return FALLBACK_TEXT

    user_input = None
    if isinstance(input_data, dict):
        user_input = input_data.get("expected_car")

    return await analyze_car(image, user_input)


async def on_stream_start(runtime, input_data):
    """Initialize streaming session state."""
    runtime.set_state("in_flight", False)
    runtime.set_state("input_data", input_data)


async def on_frame(runtime, frame):
    """Process streaming frames and emit car identification results."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)

    try:
        user_input = None
        input_data = runtime.get_state("input_data")
        if isinstance(input_data, dict):
            user_input = input_data.get("expected_car")

        text = await analyze_car(frame.image, user_input)

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
