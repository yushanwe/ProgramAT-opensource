"""Locate the nearest exit and give spoken directions toward it."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "For a blind or low-vision user looking for a way out of this indoor space, "
    "first look for exit evidence such as exit signs, doors, or hallways leading "
    "outward. Choose the nearest standard exit that is reachable from this view; "
    "if a fire emergency exit or a restricted or hazardous exit is also visible, "
    "you may mention it too, but state plainly what kind of exit it is and its "
    "function. Give one concise, step-by-step instruction using body-relative or "
    "clock-face direction (9 to 12 or 1 to 3) and approximate distance, such as "
    "turning toward a hallway and walking a short distance to the marked exit. Do "
    "not rely on color alone as the identifying cue. Return one concise "
    "user-facing response suitable for spoken output, in one or two short "
    "sentences. If no exit is visible or the evidence is insufficient, say so "
    "plainly instead of guessing."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I can't find a clear exit from this view. Try slowly scanning the room."


async def analyze_image(image) -> str:
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


__all__ = [
    "analyze_image",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
