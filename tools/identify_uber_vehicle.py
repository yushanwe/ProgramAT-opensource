"""
Identify Uber Vehicle in Scene

Identifies the make, model, color, and plate number of an Uber car in the scene
for blind or low-vision users looking for their ride.

Features:
- Uses gpt-5 for high-accuracy vehicle and plate identification
- Handles partial visibility gracefully
- Audio-optimized output for text-to-speech
- Returns all identifiable information when plate is not fully visible

Example Usage:
User is waiting for their Uber and wants to identify the vehicle approaching.
The tool will report: "White Toyota Camry, plate ABC123" or
"White sedan, plate partially visible: AB..."

Audio Output:
- Concise format suitable for speech
- Clear indication when information is incomplete
- Example: "White Toyota Camry, plate ABC123"
- Partial: "Silver SUV, make unclear, plate partially visible"
"""

import asyncio
from typing import Any, Optional

from litellm_utils import call_model, extract_text

TOOL_NAME = "identify_uber_vehicle"
TOOL_PROMPT = (
    "Identify the make, model, color, and license plate number of the vehicle "
    "in the center of this image for a blind user waiting for their Uber. "
    "If the vehicle or any details are not fully visible or cannot be determined, "
    "clearly state what you can identify and what is unclear. "
    "Format: 'Color Make Model, Plate: NUMBER' or 'Color vehicle type, make unclear, "
    "plate partially visible: ABC'. Be concise and direct."
)

DEFAULT_MODEL = "gpt-5"
FALLBACK_TEXT = "No vehicle clearly visible in the scene."


async def analyze_vehicle(image):
    """Analyze the image to identify vehicle details using gpt-5."""
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
    """Handle Take Photo mode - identify vehicle details from a single photo."""
    if image is None:
        return FALLBACK_TEXT
    return await analyze_vehicle(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming session state."""
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Handle streaming mode - identify vehicle from latest frame."""
    if runtime.is_cancelled():
        return

    # Prevent overlapping requests
    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)

    try:
        text = await analyze_vehicle(frame.image)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming session state."""
    runtime.clear_state()
