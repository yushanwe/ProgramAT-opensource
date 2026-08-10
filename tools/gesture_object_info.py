"""
Gesture-Based Object Information Identifier

Identifies objects indicated by hand gestures and provides specific information
based on gesture commands:
- Double tap: general description
- Swipe up: nutrition information
- Swipe down: expiry date

Provides accessibility feedback about hand focus status.
"""

import asyncio
from typing import Optional

from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_object_info"
TOOL_PROMPT = (
    "You are assisting a blind or low-vision user who is pointing at an object "
    "with their hand. Analyze the image to: (1) Check if a hand is visible and "
    "whether it is in focus or out of focus; (2) Identify what object the hand "
    "is pointing at or holding; (3) Based on the user's gesture command, extract "
    "the requested information about that object. For 'description' give a brief "
    "product name or item description. For 'nutrition' report key nutrition facts "
    "like sugar, caffeine, fiber if visible on packaging. For 'expiry' report the "
    "expiration or best-before date if visible. If the requested information is "
    "not visible or readable, say 'Rotate or flip the object'. If no hand is "
    "visible, say 'Hand not detected'. Start your response with the hand focus "
    "status: 'Hand in focus' or 'Hand out of focus', then provide the requested "
    "information. Return one concise user-facing response suitable for spoken "
    "output. Prefer one or two short sentences in a single paragraph with no "
    "unnecessary line breaks. State only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I am not sure from this view."
RECENT_FRAMES_COUNT = 6  # Analyze last 6 frames for gesture detection


async def detect_gesture(runtime) -> Optional[str]:
    """
    Detect dynamic gestures (double tap, swipe up, swipe down) by analyzing
    recent frames for hand motion patterns.

    Returns:
        "double_tap", "swipe_up", "swipe_down", or None if no gesture detected
    """
    frames = runtime.get_recent_frames(count=RECENT_FRAMES_COUNT)
    if len(frames) < 3:
        # Not enough frames to detect dynamic gesture
        return None

    # Use VLM to detect gesture from temporal sequence
    gesture_prompt = (
        "Analyze this sequence of chronological frames showing a person's hand. "
        "Detect if the hand performs one of these gestures: (1) double tap - hand "
        "moves down twice quickly in succession, (2) swipe up - hand moves upward "
        "in a clear upward motion, (3) swipe down - hand moves downward in a clear "
        "downward motion. If you detect one of these gestures, respond with ONLY "
        "the gesture name: 'double_tap', 'swipe_up', or 'swipe_down'. If no clear "
        "gesture is detected, respond with 'none'. Do not include any other text."
    )

    images = [frame.image for frame in frames]

    try:
        response = await asyncio.to_thread(
            call_model,
            DEFAULT_MODEL,
            [{"role": "user", "content": gesture_prompt}],
            images,
            {"timeout": 60, "num_retries": 0},
        )
        gesture_text = extract_text(response).strip().lower()

        # Parse gesture response
        if "double_tap" in gesture_text or "double tap" in gesture_text:
            return "double_tap"
        elif "swipe_up" in gesture_text or "swipe up" in gesture_text:
            return "swipe_up"
        elif "swipe_down" in gesture_text or "swipe down" in gesture_text:
            return "swipe_down"
        else:
            return None
    except Exception:
        return None


async def get_object_info(image, info_type: str) -> str:
    """
    Get specific information about the pointed object.

    Args:
        image: Current frame image
        info_type: "description", "nutrition", or "expiry"
    """
    # Build specific prompt based on info type
    info_instructions = {
        "description": (
            "provide a brief description or product name of the object"
        ),
        "nutrition": (
            "extract and report key nutrition information such as sugar, "
            "caffeine, fiber, or other relevant nutritional facts if visible "
            "on the packaging"
        ),
        "expiry": (
            "find and report the expiry date or best-before date if visible "
            "on the packaging"
        ),
    }

    instruction = info_instructions.get(info_type, info_instructions["description"])

    prompt = (
        f"{TOOL_PROMPT.replace('Based on the user's gesture command, extract the requested information about that object.', instruction)}"
    )

    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Handle take photo mode - provides description by default."""
    if image is None:
        return FALLBACK_TEXT

    # In take photo mode, default to description
    return await get_object_info(image, "description")


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    runtime.set_state("in_flight", False)
    runtime.set_state("last_gesture", None)
    runtime.set_state("gesture_cooldown", 0)


async def on_frame(runtime, frame):
    """Process each frame for gesture detection and object information."""
    if runtime.is_cancelled():
        return

    # Check if already processing
    if runtime.get_state("in_flight"):
        return

    # Claim in-flight before any await
    runtime.set_state("in_flight", True)

    try:
        # Check gesture cooldown (prevent rapid repeated detections)
        cooldown = runtime.get_state("gesture_cooldown", 0)
        if cooldown > 0:
            runtime.set_state("gesture_cooldown", cooldown - 1)
            return

        # Detect gesture from recent frames
        gesture = await detect_gesture(runtime)

        if gesture is None:
            # No gesture detected, provide basic feedback about hand presence
            # Only emit occasionally to avoid spam
            frame_count = runtime.get_state("frame_count", 0)
            runtime.set_state("frame_count", frame_count + 1)

            # Emit hand focus feedback every 30 frames (~1-2 seconds)
            if frame_count % 30 == 0:
                focus_prompt = (
                    "Check if a hand is visible in this image and whether it is "
                    "in focus or out of focus. Respond with ONLY: 'Hand in focus', "
                    "'Hand out of focus', or 'Hand not detected'. No other text."
                )
                response = await asyncio.to_thread(
                    call_model,
                    DEFAULT_MODEL,
                    [{"role": "user", "content": focus_prompt}],
                    [frame.image],
                    {"timeout": 30, "num_retries": 0},
                )
                feedback = extract_text(response).strip()
                if feedback and not runtime.is_cancelled():
                    await runtime.emit(feedback, partial=True)
            return

        # Gesture detected - determine info type
        info_type_map = {
            "double_tap": "description",
            "swipe_up": "nutrition",
            "swipe_down": "expiry",
        }
        info_type = info_type_map.get(gesture, "description")

        # Get object information
        result = await get_object_info(frame.image, info_type)

        if not runtime.is_cancelled():
            await runtime.emit(result, final=True)

        # Set cooldown to prevent immediate re-detection
        runtime.set_state("gesture_cooldown", 20)  # ~20 frames cooldown
        runtime.set_state("last_gesture", gesture)

    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
