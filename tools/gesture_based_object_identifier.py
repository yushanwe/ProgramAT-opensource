"""Gesture-based object identifier with hand focus feedback."""

from __future__ import annotations

import asyncio
import time

from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_based_object_identifier"
TOOL_PROMPT = (
    "You are helping a blind or low-vision user identify objects through hand gestures. "
    "The user is holding or pointing at an object. Analyze the image and respond with "
    "only the requested information in a concise, spoken-friendly format."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify the object clearly."
GESTURE_COOLDOWN_SECONDS = 3.0
HAND_FOCUS_CHECK_INTERVAL = 1.0


def _build_gesture_prompt(gesture_type: str) -> str:
    """Build a specific prompt based on the detected gesture type."""
    if gesture_type == "double_tap":
        return (
            f"{TOOL_PROMPT}\n\n"
            "The user performed a double tap gesture. Provide a brief, accurate "
            "description of the object they are holding or pointing at. "
            "Keep it minimally verbose and suitable for speech output. "
            "Return one concise user-facing response suitable for spoken output. "
            "Prefer one or two short sentences in a single paragraph with no unnecessary "
            "line breaks. State only the information needed for the user's task."
        )
    elif gesture_type == "swipe_up":
        return (
            f"{TOOL_PROMPT}\n\n"
            "The user performed a swipe up gesture. Provide nutrition information "
            "for the object they are holding: sugar content, caffeine, fiber, calories, "
            "or other relevant nutritional facts. If this information is not visible, "
            "say 'Rotate or flip the object'. Keep the response concise and spoken-friendly. "
            "Return one concise user-facing response suitable for spoken output. "
            "Prefer one or two short sentences in a single paragraph with no unnecessary "
            "line breaks. State only the information needed for the user's task."
        )
    elif gesture_type == "swipe_down":
        return (
            f"{TOOL_PROMPT}\n\n"
            "The user performed a swipe down gesture. Look for and report the expiry date, "
            "best before date, or use by date on the object they are holding. "
            "If this information is not visible, say 'Rotate or flip the object'. "
            "Keep the response concise and spoken-friendly. "
            "Return one concise user-facing response suitable for spoken output. "
            "Prefer one or two short sentences in a single paragraph with no unnecessary "
            "line breaks. State only the information needed for the user's task."
        )
    return TOOL_PROMPT


async def _detect_hand_and_gesture(frames) -> tuple[str, str]:
    """
    Analyze recent frames to detect hand presence, focus, and gesture type.

    Returns:
        tuple: (hand_status, gesture_type)
        hand_status: "in_focus", "out_of_focus", or "not_detected"
        gesture_type: "double_tap", "swipe_up", "swipe_down", or "none"
    """
    if not frames or len(frames) < 2:
        return "not_detected", "none"

    # Use first, middle, and last frames to detect motion
    images = [frames[0].image, frames[len(frames)//2].image, frames[-1].image]

    gesture_detection_prompt = (
        "Analyze these chronological frames to detect hand gestures and focus. "
        "Look for: "
        "1. Hand presence (is a hand visible holding or pointing at an object?) "
        "2. Hand focus (is the hand sharp and clear, or blurry/out of focus?) "
        "3. Gesture motion across frames: "
        "   - Double tap: hand makes a quick tapping motion (appears, moves down, returns) "
        "   - Swipe up: hand moves upward across frames "
        "   - Swipe down: hand moves downward across frames "
        "   - None: hand is stationary or no clear gesture pattern "
        "\n"
        "Respond in exactly this format: "
        "HAND_STATUS: [in_focus / out_of_focus / not_detected] "
        "GESTURE: [double_tap / swipe_up / swipe_down / none] "
        "\n"
        "Be conservative: only report a gesture if the motion pattern is clear across frames."
    )

    try:
        response = await asyncio.to_thread(
            call_model,
            DEFAULT_MODEL,
            [{"role": "user", "content": gesture_detection_prompt}],
            images,
            {"timeout": 30, "num_retries": 0},
        )
        text = extract_text(response).strip()

        # Parse the response
        hand_status = "not_detected"
        gesture = "none"

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("HAND_STATUS:"):
                status = line.split(":", 1)[1].strip().lower()
                if status in ["in_focus", "out_of_focus", "not_detected"]:
                    hand_status = status
            elif line.startswith("GESTURE:"):
                gest = line.split(":", 1)[1].strip().lower()
                if gest in ["double_tap", "swipe_up", "swipe_down", "none"]:
                    gesture = gest

        return hand_status, gesture

    except Exception:
        return "not_detected", "none"


async def _identify_object(image, gesture_type: str) -> str:
    """Identify the object and return information based on gesture type."""
    prompt = _build_gesture_prompt(gesture_type)

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
    """Handle single photo capture - use latest frame for gesture detection."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT

    # For take photo mode, we can't detect temporal gestures reliably
    # Default to description (double tap equivalent)
    return await _identify_object(image, "double_tap")


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("last_gesture_time", 0.0)
    runtime.set_state("last_focus_check_time", 0.0)
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Analyze frames for hand focus and gestures."""
    if runtime.is_cancelled():
        return

    if runtime.get_state("in_flight"):
        return

    current_time = time.time()
    last_gesture_time = runtime.get_state("last_gesture_time", 0.0)
    last_focus_check_time = runtime.get_state("last_focus_check_time", 0.0)

    # Enforce cooldown after gesture detection
    if current_time - last_gesture_time < GESTURE_COOLDOWN_SECONDS:
        return

    # Check hand focus periodically (less frequently than gesture detection)
    should_check_focus = (current_time - last_focus_check_time) >= HAND_FOCUS_CHECK_INTERVAL

    runtime.set_state("in_flight", True)

    try:
        # Get recent frames for temporal gesture analysis
        recent_frames = runtime.get_recent_frames(seconds=2)

        if not recent_frames or len(recent_frames) < 2:
            # Not enough frames for gesture detection
            return

        # Detect hand presence, focus, and gesture
        hand_status, gesture_type = await _detect_hand_and_gesture(recent_frames)

        if runtime.is_cancelled():
            return

        # Provide hand focus feedback
        if should_check_focus and hand_status in ["in_focus", "out_of_focus"]:
            runtime.set_state("last_focus_check_time", current_time)
            focus_message = f"Hand {hand_status.replace('_', ' ')}"
            await runtime.emit(focus_message, partial=True, replace=True)

        # If a gesture was detected, identify the object
        if gesture_type != "none" and hand_status == "in_focus":
            runtime.set_state("last_gesture_time", current_time)

            # Use the latest frame for object identification
            result = await _identify_object(frame.image, gesture_type)

            if not runtime.is_cancelled():
                await runtime.emit(result, final=True)

    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
