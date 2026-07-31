"""Find Uber Car Attributes Tool for Blind Users

Identifies the make, model, color, and license plate number of a car in the center
of the scene to help users find their Uber ride.

Features:
- Analyzes car attributes using vision AI
- Focuses on the car at the center of the frame
- Returns speech-friendly descriptions
- Works in both Take Photo and Streaming modes
"""

import asyncio
import cv2
import numpy as np
from typing import Optional, Any

from litellm_utils import call_model, extract_text

TOOL_NAME = "find_uber_car_attributes"
TOOL_PROMPT = (
    "Identify the car at the center of this image. State its make, model, color, "
    "and license plate number if visible. If you cannot determine any attribute "
    "with confidence, say so. Format your response as: '[color] [make] [model] "
    "with license plate [plate number]' or state what is unclear."
)


def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    """Resize image efficiently while maintaining aspect ratio."""
    height, width = image.shape[:2]
    max_width, max_height = max_size

    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


async def on_take_photo(runtime, image, input_data):
    """Analyze car attributes from a single photo."""
    del runtime, input_data

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"

    processed_image = resize_image_if_needed(image)

    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [processed_image],
        {"timeout": 60, "num_retries": 0},
    )

    text = extract_text(response).strip()
    return text or "Unable to identify car attributes"


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("last_result", None)
    runtime.set_state("analyzing", False)


async def on_frame(runtime, frame):
    """Analyze the latest frame for car attributes."""
    if runtime.is_cancelled():
        return

    # Guard: prevent duplicate overlapping calls
    if runtime.get_state("analyzing"):
        return

    # Set state BEFORE first await to prevent race condition
    runtime.set_state("analyzing", True)

    try:
        if frame.image is None or not isinstance(frame.image, np.ndarray) or frame.image.size == 0:
            return

        processed_image = resize_image_if_needed(frame.image)

        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            [processed_image],
            {"timeout": 60, "num_retries": 0},
        )

        if runtime.is_cancelled():
            return

        text = extract_text(response).strip()
        result = text or "Unable to identify car attributes"

        last_result = runtime.get_state("last_result")
        if result != last_result:
            runtime.set_state("last_result", result)
            await runtime.emit(result, final=True)

    finally:
        runtime.set_state("analyzing", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
