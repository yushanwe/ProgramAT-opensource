"""Locate the nearest exit door and give step-by-step navigation toward it."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "For a blind or low-vision user, locate the nearest visible exit door or exit sign in this "
    "image. First check whether the way to it is obstructed, such as by furniture or clutter in "
    "the path, a closed or locked door, smoke, or another hazard, and if so say plainly that the "
    "exit is obstructed and briefly why. Otherwise give brief step-by-step directions to reach it "
    "safely, using body-relative or clock-face directions such as straight ahead, left, right, or "
    "a clock position between 9 and 12 or between 1 and 3, plus a short movement instruction and "
    "an approximate distance when you can tell. Do not use color as the only cue. If no exit is "
    "visible, say so clearly instead of guessing. Return one concise user-facing response "
    "suitable for spoken output, in one or two short sentences with no lists or headings."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I can't find an exit from this view. Try turning slowly to scan the room."


async def analyze_image(image):
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
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
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
    runtime.clear_state()
