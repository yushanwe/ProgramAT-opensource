"""
Identify Uber Car Tool

Identifies the car at the middle of the scene and provides its make, model,
color, and license plate number to help blind users find their Uber.

This tool analyzes the center of the camera view to identify a specific vehicle.
It reports the car's make, model, color, and license plate when visible.
"""

from typing import Any
from litellm_utils import call_model, extract_text, _image_to_data_uri

TOOL_NAME = "identify_uber_car"
TOOL_PROMPT = (
    "Identify the car at the center of the image. Report its make, model, color, "
    "and license plate number if visible. Focus on the most prominent vehicle in "
    "the middle of the scene. If details are unclear or no car is visible at the "
    "center, state what can and cannot be determined."
)


async def on_take_photo(runtime, image, input_data):
    """Handle Take Photo mode - analyze one frame."""
    if image is None:
        return "No camera image available"

    try:
        image_uri = _image_to_data_uri(image)

        response = call_model(
            model_name="gemini/gemini-2.0-flash-exp",
            messages=[
                {
                    "role": "user",
                    "content": TOOL_PROMPT
                }
            ],
            images=[image_uri],
            metadata={
                "temperature": 0.3,
                "max_tokens": 200
            }
        )

        text = extract_text(response).strip()
        text = text or "Unable to identify car details at this time"

        return text

    except Exception as e:
        return f"Error identifying car: unable to process image"


async def on_stream_start(runtime, input_data):
    """Initialize streaming session."""
    runtime.clear_state()


async def on_frame(runtime, frame):
    """Handle streaming mode - analyze latest frame when scene changes."""
    if runtime.is_cancelled():
        return

    if frame is None or frame.image is None:
        return

    # Check if already processing
    if runtime.get_state("processing", False):
        return

    # Claim processing slot before any await to prevent duplicate calls
    runtime.set_state("processing", True)

    try:
        last_result = runtime.get_state("last_result", "")
        image_uri = _image_to_data_uri(frame.image)

        response = call_model(
            model_name="gemini/gemini-2.0-flash-exp",
            messages=[
                {
                    "role": "user",
                    "content": TOOL_PROMPT
                }
            ],
            images=[image_uri],
            metadata={
                "temperature": 0.3,
                "max_tokens": 200
            }
        )

        text = extract_text(response).strip()
        text = text or "Unable to identify car details"

        if text != last_result:
            runtime.set_state("last_result", text)
            if not runtime.is_cancelled():
                await runtime.emit(text, final=True)

    except Exception as e:
        error_msg = "Error identifying car"
        last_result = runtime.get_state("last_result", "")
        if error_msg != last_result:
            runtime.set_state("last_result", error_msg)
            if not runtime.is_cancelled():
                await runtime.emit(error_msg, final=True)
    finally:
        runtime.set_state("processing", False)


async def on_stream_stop(runtime):
    """Clean up streaming session."""
    runtime.clear_state()
