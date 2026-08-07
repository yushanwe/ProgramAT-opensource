"""
Exit Finder Tool

A tool that locates the nearest exit door and provides step-by-step navigation
directions for blind and low-vision users in indoor environments.

Features:
- Identifies exit signs, exit doors, and doorways
- Prioritizes the closest or most approachable exit
- Provides spoken navigation using clock-face directions (9-12, 1-3)
- Includes approximate distance when supportable
- Audio-optimized output for text-to-speech

Example output:
"The nearest exit is through the door 10 feet to your left; proceed straight and turn left."
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "You are helping a blind or low-vision user locate the nearest exit in an indoor environment. "
    "Identify visible exit signs, exit doors, or doorways that could serve as exits. "
    "Prioritize the closest or most approachable exit based on visibility, distance, and clear path. "
    "Determine the exit's location using clock-face directions (only 9, 10, 11, 12, 1, 2, or 3 o'clock - "
    "never use 4-8 as those are behind the camera). "
    "Estimate approximate distance when clear structural cues support it (such as 5 feet, 10 feet, 15 feet). "
    "Do not rely on color alone or visual-only landmarks. "
    "Provide one concise spoken instruction with: exit type, clock position or body-relative direction, "
    "approximate distance if supportable, and a brief movement instruction. "
    "If no exit is visible or identifiable, say that clearly. "
    "Return one concise user-facing response suitable for spoken output. Prefer one or two short sentences "
    "in a single paragraph with no unnecessary line breaks. State only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify an exit from this view."


async def analyze_image(image):
    """Analyze image to locate exits and provide navigation directions."""
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
    """Handle Take Photo mode execution."""
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming mode state."""
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Handle streaming mode frame processing."""
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
    """Clean up streaming mode state."""
    runtime.clear_state()
