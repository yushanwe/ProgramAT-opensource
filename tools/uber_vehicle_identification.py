"""Uber Vehicle Identification and Verification Tool."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_vehicle_identification"
TOOL_PROMPT = (
    "For a blind or low-vision user waiting for a rideshare pickup, identify the "
    "vehicle at the center of the scene: state its color, make, model, and license "
    "plate number when they are clearly visible. If any of those details, such as "
    "the license plate, cannot be clearly read due to distance, angle, glare, dirt, "
    "or obstruction, say plainly that you cannot identify that detail instead of "
    "guessing. Return one concise user-facing response suitable for spoken output in "
    "one or two short sentences with no unnecessary line breaks."
)
TOOL_RUNTIME_INPUT = {
    "key": "expected_vehicle",
    "label": "Vehicle you're expecting",
    "placeholder": "e.g. white Toyota Camry, plate ABC123",
    "prompt_instruction": (
        "The user is expecting a vehicle matching this description: {value}. "
        "Compare the vehicle you identify against this description and clearly "
        "state whether it matches or is a mismatch."
    ),
}

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I can't clearly identify the vehicle from this view."


def build_prompt(expected_vehicle):
    if not expected_vehicle:
        return TOOL_PROMPT
    instruction = TOOL_RUNTIME_INPUT["prompt_instruction"].format(value=expected_vehicle)
    return f"{TOOL_PROMPT} {instruction}"


async def analyze_image(image, expected_vehicle=None):
    prompt = build_prompt(expected_vehicle)
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
    del runtime
    if image is None:
        return FALLBACK_TEXT
    expected_vehicle = None
    if isinstance(input_data, dict):
        expected_vehicle = input_data.get(TOOL_RUNTIME_INPUT["key"])
    return await analyze_image(image, expected_vehicle)


async def on_stream_start(runtime, input_data):
    expected_vehicle = None
    if isinstance(input_data, dict):
        expected_vehicle = input_data.get(TOOL_RUNTIME_INPUT["key"])
    runtime.set_state("expected_vehicle", expected_vehicle)
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)
    try:
        expected_vehicle = runtime.get_state("expected_vehicle")
        text = await analyze_image(frame.image, expected_vehicle)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    runtime.clear_state()


__all__ = [
    "build_prompt",
    "analyze_image",
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
