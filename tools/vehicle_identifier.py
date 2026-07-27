"""Identify the vehicle in the center of the scene for blind and low-vision users."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "vehicle_identifier"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Identify the vehicle located in the center of this scene for a blind or low-vision user. "
    "State its make, model, color, and license plate number if visible. "
    "Give the result as one concise spoken sentence, for example: "
    "'The vehicle in the center is a white Toyota Camry with license plate ABC 123.' "
    "If no vehicle is visible in the center, or the make, model, or color cannot be determined, "
    "say so plainly."
)


async def on_take_photo(runtime, image, input_data):
    """Identify the vehicle in the center of the scene using a single model call."""
    del runtime, input_data
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60},
    )
    return extract_text(response)


__all__ = ["on_take_photo"]
