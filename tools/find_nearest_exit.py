"""Find the nearest exit in an unfamiliar indoor environment."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "find_nearest_exit"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "You are assisting a blind or low-vision user who needs to find the nearest exit "
    "in an unfamiliar indoor environment. Examine the image carefully for any exit "
    "signs, emergency exit signs, exit doors, stairwell doors, or hallways that lead "
    "to an exit. Provide concise, spoken step-by-step directions using clock-face "
    "positions (for example, '12 o'clock is straight ahead, 3 o'clock is to your "
    "right, 9 o'clock is to your left') and approximate distances or step counts. "
    "Do not use color or other visual-only landmarks as the sole navigation cue. "
    "Describe the path as: first direction and distance, then any turns needed, and "
    "confirm the exit. If no exit or exit sign is visible, say clearly that no exit "
    "is visible in this view and suggest the user slowly turn to scan the room."
)


async def on_take_photo(runtime, image, input_data):
    """Locate the nearest exit and return spoken step-by-step directions."""
    del runtime, input_data
    if image is None:
        return "No camera image is available. Please point your camera at the room."
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


__all__ = ["on_take_photo"]
