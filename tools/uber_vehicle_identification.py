"""Uber Vehicle Identification

Identifies the make, model, color, and license plate number of an Uber car
in the scene to help blind and low-vision users locate their ride.
"""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_vehicle_identification"
TOOL_PROMPT = (
    "Identify the make, model, color, and license plate number of the vehicle in "
    "the center of this image. Focus on the car that appears most prominent or "
    "central. Format the response as a concise spoken description suitable for a "
    "blind user, such as: 'A white Toyota Camry with license plate ABC123.' If any "
    "detail is unclear or not visible, state which information is unavailable. If "
    "no vehicle is visible, say so plainly."
)


async def on_take_photo(runtime, image, input_data):
    """Analyze one current image to identify the Uber vehicle."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."

    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("last_result", None)


async def on_frame(runtime, frame):
    """Process changed latest frame and emit vehicle identification."""
    if runtime.is_cancelled():
        return

    # Get the latest frame for analysis
    latest = runtime.get_latest_frame()
    if latest is None:
        return

    # Use a faster model for streaming to provide quicker updates
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [latest.image],
        {"timeout": 30, "num_retries": 0},
    )
    result = extract_text(response)

    # Only emit if the result changed meaningfully
    last_result = runtime.get_state("last_result")
    if result != last_result and result:
        runtime.set_state("last_result", result)
        await runtime.emit(result, partial=False, final=False)


async def on_stream_stop(runtime):
    """Clear streaming state when stopped."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
