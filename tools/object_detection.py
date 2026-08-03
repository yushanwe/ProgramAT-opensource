"""
Object Detection Tool

Detects objects in the camera view and provides location guidance for blind and
low-vision users. Supports two modes:
1. General detection: Lists all visible objects when no query is provided
2. Specific search: Locates a specific object by name and provides clock-face direction

Features:
- Semantic object understanding with Gemini VLM
- Clock-face navigation guidance (9-12 and 1-3 positions only)
- Handles multiple similar objects by offering choices
- Speech-friendly output for accessibility
- Streaming support with state tracking
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "object_detection"
TOOL_PROMPT = (
    "For a blind or low-vision user, identify useful visible objects and give a concise spoken "
    "list with counts and accessible relative positions such as left, center, right, or clock "
    "direction when helpful. Do not locate objects only by color or nearby people. If no objects "
    "can be identified clearly, say so."
)
TOOL_PROMPT_SPECIFIC = (
    "For a blind or low-vision user looking for {query}, examine the scene carefully. If the "
    "requested object is visible, report its location using clock-face directions (1-3 o'clock "
    "on the right, 9-12 o'clock on the left, or straight ahead at 12). If multiple matching "
    "objects are visible, mention how many and describe the nearest one's location. If the "
    "object is not visible, say so clearly. Do not use color-only landmarks or positions "
    "behind the camera."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite-preview"
FALLBACK_TEXT = "I cannot determine what objects are visible from this view."
FALLBACK_TEXT_SPECIFIC = "I cannot find {query} in this view."


async def detect_general_objects(image):
    """Detect and describe all visible objects in general mode."""
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def detect_specific_object(image, query):
    """Locate a specific object and provide direction guidance."""
    prompt = TOOL_PROMPT_SPECIFIC.format(query=query)
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT_SPECIFIC.format(query=query)


async def on_take_photo(runtime, image, input_data):
    """Handle Take Photo mode - single frame analysis."""
    if image is None:
        return FALLBACK_TEXT

    # Check if user provided a specific object query
    query = None
    if input_data:
        if isinstance(input_data, dict):
            query = input_data.get("query") or input_data.get("target_object")
        elif isinstance(input_data, str) and input_data.strip():
            query = input_data.strip()

    if query:
        # Specific object search mode
        return await detect_specific_object(image, query)
    else:
        # General detection mode
        return await detect_general_objects(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    runtime.set_state("in_flight", False)
    runtime.set_state("last_result", None)

    # Store query from input_data if provided
    query = None
    if input_data:
        if isinstance(input_data, dict):
            query = input_data.get("query") or input_data.get("target_object")
        elif isinstance(input_data, str) and input_data.strip():
            query = input_data.strip()
    runtime.set_state("query", query)


async def on_frame(runtime, frame):
    """Handle streaming frames with state tracking to avoid repetition."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)
    try:
        query = runtime.get_state("query")

        if query:
            # Specific object search mode
            text = await detect_specific_object(frame.image, query)
        else:
            # General detection mode
            text = await detect_general_objects(frame.image)

        # Only emit if result is different from last time (avoid repetition)
        last_result = runtime.get_state("last_result")
        if not runtime.is_cancelled() and text != last_result:
            await runtime.emit(text, final=True)
            runtime.set_state("last_result", text)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
