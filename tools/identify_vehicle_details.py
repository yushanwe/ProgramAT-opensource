"""Identify vehicle make, model, color, and license plate for blind users finding their ride."""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "identify_vehicle_details"
TOOL_PROMPT = (
    "For a blind or low-vision user looking for their ride, identify any visible vehicle "
    "in the scene and report its make, model, color, and license plate number. Focus on the "
    "vehicle at or near the center of the scene. If multiple vehicles are visible, describe "
    "the most prominent one. For each detail, state what you can see clearly and indicate "
    "which details you are uncertain about or cannot read. If no vehicle is clearly visible "
    "or identifiable, say so. Return the information in a concise, speech-friendly format "
    "like: 'White Toyota Camry with license plate ABC123.' If some details are unreadable, "
    "say which ones, such as: 'White sedan, make and model unclear, license plate not visible.' "
    "If you are uncertain about any detail, state the uncertainty, such as: 'appears to be "
    "a white Toyota, possibly a Camry, license plate ABC123, though the plate is partially "
    "obscured.'"
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify a vehicle clearly from this view."


async def analyze_image(image):
    """Analyze image for vehicle details using vision model."""
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
    """Single photo analysis for vehicle identification."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Process each frame for vehicle identification in streaming mode."""
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
    """Clean up streaming state."""
    runtime.clear_state()
