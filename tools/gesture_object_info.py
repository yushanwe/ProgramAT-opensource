"""Gesture-based Object Information Identifier

Identifies objects pointed at or held by the user and provides specific information
based on hand gestures: double tap for general description, swipe up for nutrition,
and swipe down for expiry date.
"""

import asyncio
from typing import List, Optional

from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_object_info"
TOOL_PROMPT = """You are assisting a blind or low-vision user who is holding or pointing at an object.

Analyze the provided frames and:
1. Assess if a hand is visible and in focus. Report "hand out of focus" if blurry, or "hand in focus" if clear.
2. Identify the object being held or pointed at by the user.
3. Based on the requested information type, provide the appropriate details:
   - DESCRIPTION: Give a concise description of the object (e.g., "Lipton Lemon and Earl Grey tea bag")
   - NUTRITION: Provide key nutrition facts (e.g., "0g sugar, 100mg caffeine, 0g fiber")
   - EXPIRY: Report the expiry or best-by date (e.g., "Expiry date November 16, 2027")

If the requested information is not visible, say "Rotate or flip the object".
If no object is clearly held or pointed at, say "No object detected in hand".

Return one concise user-facing response suitable for spoken output. Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. State only the information needed for the user's task."""

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "Unable to detect gesture or object."
USE_RECENT_FRAMES = True

# Gesture detection window (seconds)
GESTURE_DETECTION_WINDOW = 2.0
# Minimum frames to detect gesture
MIN_FRAMES_FOR_GESTURE = 4


def build_prompt_for_info_type(info_type: str) -> str:
    """Build a VLM prompt for the specific information type requested."""
    base = TOOL_PROMPT.replace(
        "Based on the requested information type, provide the appropriate details:",
        f"The user requested: {info_type}. Provide:",
    )

    if info_type == "DESCRIPTION":
        return base.replace(
            "- DESCRIPTION: Give a concise description of the object",
            "A concise description of the object being held or pointed at",
        )
    elif info_type == "NUTRITION":
        return base.replace(
            "- NUTRITION: Provide key nutrition facts",
            "Key nutrition facts like sugar, caffeine, fiber, calories if visible on the packaging",
        )
    elif info_type == "EXPIRY":
        return base.replace(
            "- EXPIRY: Report the expiry or best-by date",
            "The expiry date, best-by date, or use-by date if visible on the packaging",
        )

    return base


async def detect_gesture_from_frames(frames: List) -> Optional[str]:
    """Detect gesture type from recent frames.

    Returns:
        "double_tap" for double tap (general description)
        "swipe_up" for swipe up (nutrition info)
        "swipe_down" for swipe down (expiry date)
        None if no clear gesture detected
    """
    if not frames or len(frames) < MIN_FRAMES_FOR_GESTURE:
        return None

    # Build a gesture detection prompt
    gesture_prompt = """Analyze these chronological frames showing a user's hand gesture.

Detect ONE of these specific gestures:
1. DOUBLE TAP: Hand makes two quick tapping motions (fingers close twice in sequence)
2. SWIPE UP: Hand moves upward across the frames
3. SWIPE DOWN: Hand moves downward across the frames

If you detect a clear gesture, respond with ONLY ONE of: "double_tap", "swipe_up", or "swipe_down".
If no clear gesture is detected, respond with: "none".

Return ONLY the gesture name, nothing else."""

    # Extract images from frames
    images = [frame.image for frame in frames if frame and hasattr(frame, 'image')]

    if len(images) < MIN_FRAMES_FOR_GESTURE:
        return None

    try:
        response = await asyncio.to_thread(
            call_model,
            DEFAULT_MODEL,
            [{"role": "user", "content": gesture_prompt}],
            images,
            {"timeout": 30, "num_retries": 0},
        )

        gesture_text = extract_text(response).strip().lower()

        if "double_tap" in gesture_text or "double tap" in gesture_text:
            return "double_tap"
        elif "swipe_up" in gesture_text or "swipe up" in gesture_text:
            return "swipe_up"
        elif "swipe_down" in gesture_text or "swipe down" in gesture_text:
            return "swipe_down"

        return None

    except Exception:
        return None


async def analyze_object_with_info_type(image, info_type: str) -> str:
    """Analyze the object in the image and return the requested information."""
    prompt = build_prompt_for_info_type(info_type)

    try:
        response = await asyncio.to_thread(
            call_model,
            DEFAULT_MODEL,
            [{"role": "user", "content": prompt}],
            [image],
            {"timeout": 60, "num_retries": 0},
        )
        text = extract_text(response).strip()
        return text or FALLBACK_TEXT
    except Exception:
        return FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Take Photo handler - defaults to description mode."""
    if image is None:
        return FALLBACK_TEXT

    # Take Photo mode always provides description
    return await analyze_object_with_info_type(image, "DESCRIPTION")


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    runtime.set_state("in_flight", False)
    runtime.set_state("last_gesture_time", 0.0)
    runtime.set_state("last_info_type", None)


async def on_frame(runtime, frame):
    """Process each frame in streaming mode."""
    if runtime.is_cancelled():
        return

    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)

    try:
        # Get recent frames for gesture detection
        recent_frames = runtime.get_recent_frames(seconds=GESTURE_DETECTION_WINDOW)

        if not recent_frames or len(recent_frames) < MIN_FRAMES_FOR_GESTURE:
            # Not enough frames yet, skip
            return

        # Try to detect gesture from recent frames
        detected_gesture = await detect_gesture_from_frames(recent_frames)

        if detected_gesture is None:
            # No gesture detected, don't emit anything
            return

        # Check if this is the same gesture we just processed
        last_info_type = runtime.get_state("last_info_type")

        # Map gesture to info type
        info_type_map = {
            "double_tap": "DESCRIPTION",
            "swipe_up": "NUTRITION",
            "swipe_down": "EXPIRY",
        }

        info_type = info_type_map.get(detected_gesture)

        if info_type is None:
            return

        # Throttle: don't process the same info type repeatedly
        if info_type == last_info_type:
            return

        # Update state before processing
        runtime.set_state("last_info_type", info_type)
        runtime.set_state("last_gesture_time", frame.timestamp if hasattr(frame, 'timestamp') else 0.0)

        # Use the latest frame to analyze the object
        current_image = frame.image

        if current_image is None:
            await runtime.emit(FALLBACK_TEXT, final=True)
            return

        # Analyze object with the requested information type
        result = await analyze_object_with_info_type(current_image, info_type)

        if not runtime.is_cancelled():
            await runtime.emit(result, final=True)

    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Cleanup streaming state."""
    runtime.clear_state()
