"""Gesture-Based Object Information Identifier

Identifies objects indicated by hand gestures and provides specific information
based on gesture commands:
- Double tap: general object description
- Swipe up: nutrition information
- Swipe down: expiry date

Includes accessibility feedback for hand focus status.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_based_object_info"
TOOL_PROMPT = (
    "Identify the object being held in the hand visible in this image. Provide a "
    "concise description suitable for spoken output. Focus on the object in the hand, "
    "not other items in the background. Return one concise user-facing response suitable "
    "for spoken output. Prefer one or two short sentences in a single paragraph with no "
    "unnecessary line breaks. State only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I am not sure from this view."

# Gesture detection thresholds
HAND_DETECTION_CONFIDENCE = 0.3
SWIPE_DISTANCE_THRESHOLD = 100  # pixels
DOUBLE_TAP_TIME_WINDOW = 1.0  # seconds
DOUBLE_TAP_POSITION_TOLERANCE = 80  # pixels
GESTURE_COOLDOWN = 2.0  # seconds between gesture actions


def detect_hand(image: np.ndarray) -> Optional[tuple[int, int, float]]:
    """Detect hand in image using simple skin color detection.

    Returns (center_x, center_y, confidence) or None if no hand detected.
    """
    if image is None or image.size == 0:
        return None

    try:
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour (assumed to be hand)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Calculate confidence based on area
        image_area = image.shape[0] * image.shape[1]
        confidence = min(area / (image_area * 0.3), 1.0)

        if confidence < HAND_DETECTION_CONFIDENCE:
            return None

        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy, confidence)

    except Exception:
        return None


def detect_gesture(
    current_hand: Optional[tuple[int, int, float]],
    previous_positions: list[tuple[Optional[tuple[int, int, float]], float]],
) -> Optional[str]:
    """Detect gesture from hand position history.

    Returns one of: "double_tap", "swipe_up", "swipe_down", or None
    """
    if len(previous_positions) < 3:
        return None

    current_time = time.time()

    # Check for double tap: present -> absent -> present
    if current_hand is not None and len(previous_positions) >= 3:
        # Get recent history within time window
        recent = [
            (pos, t) for pos, t in previous_positions[-10:]
            if current_time - t <= DOUBLE_TAP_TIME_WINDOW
        ]

        if len(recent) >= 3:
            # Check pattern: hand present, then absent, then present again
            has_hand = [pos is not None for pos, _ in recent]

            # Find transitions
            if current_hand is not None:
                # Current frame has hand, check if there was an absence
                for i in range(len(has_hand) - 1):
                    if has_hand[i] and not has_hand[i + 1]:
                        # Found present -> absent transition
                        # Check if the hand positions before and after are similar
                        before_pos = recent[i][0]
                        after_pos = current_hand[:2]

                        if before_pos is not None:
                            distance = np.sqrt(
                                (before_pos[0] - after_pos[0]) ** 2 +
                                (before_pos[1] - after_pos[1]) ** 2
                            )
                            if distance < DOUBLE_TAP_POSITION_TOLERANCE:
                                return "double_tap"

    # Check for swipe gestures
    if current_hand is not None and len(previous_positions) >= 2:
        # Get hand positions from recent history
        hand_positions = [
            (pos[:2], t) for pos, t in previous_positions[-5:]
            if pos is not None
        ]

        if len(hand_positions) >= 2:
            oldest_pos, oldest_time = hand_positions[0]
            newest_pos = current_hand[:2]

            # Calculate movement
            dx = newest_pos[0] - oldest_pos[0]
            dy = newest_pos[1] - oldest_pos[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)

            # Check if movement is significant
            if distance >= SWIPE_DISTANCE_THRESHOLD:
                # Determine primary direction
                if abs(dy) > abs(dx):
                    if dy < 0:
                        return "swipe_up"
                    else:
                        return "swipe_down"

    return None


async def get_object_description(image: np.ndarray) -> str:
    """Get general object description."""
    prompt = (
        "Identify the object being held in the hand visible in this image. Provide "
        "a concise brand name and description suitable for spoken output. For example: "
        "'Lipton Lemon and Earl Grey tea bag'. Return one concise user-facing response "
        "suitable for spoken output. Prefer one or two short sentences in a single "
        "paragraph with no unnecessary line breaks."
    )
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or "Unable to identify the object clearly."


async def get_nutrition_info(image: np.ndarray) -> str:
    """Get nutrition information from the object."""
    prompt = (
        "Read the nutrition information from the object visible in this image. "
        "Focus on key nutritional facts such as sugar content, caffeine, fiber, "
        "calories, protein, or other notable nutrients. Provide a concise spoken "
        "summary suitable for audio output. For example: '0g sugar, 100mg caffeine, "
        "0g fiber'. If the nutrition information is not visible or readable, say "
        "'Rotate or flip the object'. Return one concise user-facing response suitable "
        "for spoken output."
    )
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    if not text or "not visible" in text.lower() or "cannot see" in text.lower():
        return "Rotate or flip the object."
    return text


async def get_expiry_date(image: np.ndarray) -> str:
    """Get expiry date from the object."""
    prompt = (
        "Read the expiry date, best before date, or use by date from the object "
        "visible in this image. Provide the date in a natural spoken format suitable "
        "for audio output. For example: 'Expiry date November 16, 2027' or 'Best "
        "before March 15, 2026'. If the expiry date is not visible or readable, say "
        "'Rotate or flip the object'. Return one concise user-facing response suitable "
        "for spoken output."
    )
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    if not text or "not visible" in text.lower() or "cannot see" in text.lower():
        return "Rotate or flip the object."
    return text


async def on_take_photo(runtime, image, input_data):
    """Take photo mode: provide general object description."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await get_object_description(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("hand_positions", [])  # List of (hand_pos, timestamp)
    runtime.set_state("last_hand_status", None)  # Track hand focus status
    runtime.set_state("last_gesture_time", 0.0)  # Cooldown for gestures
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Process each frame for gesture detection and object information."""
    if runtime.is_cancelled():
        return

    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)

    try:
        # Detect hand in current frame
        hand_result = await asyncio.to_thread(detect_hand, frame.image)

        # Update hand position history
        hand_positions = runtime.get_state("hand_positions", [])
        current_time = time.time()
        hand_positions.append((hand_result, current_time))

        # Keep only recent history (last 2 seconds)
        hand_positions = [
            (pos, t) for pos, t in hand_positions
            if current_time - t <= 2.0
        ]
        runtime.set_state("hand_positions", hand_positions)

        # Check hand focus status for accessibility feedback
        last_status = runtime.get_state("last_hand_status")
        current_status = "in_focus" if hand_result is not None else "out_of_focus"

        if last_status != current_status:
            runtime.set_state("last_hand_status", current_status)
            if current_status == "in_focus":
                await runtime.emit("Hand in focus.", partial=True)
            else:
                await runtime.emit("Hand out of focus.", partial=True)

        # Detect gesture
        gesture = detect_gesture(hand_result, hand_positions)

        # Check gesture cooldown
        last_gesture_time = runtime.get_state("last_gesture_time", 0.0)
        if gesture and (current_time - last_gesture_time) >= GESTURE_COOLDOWN:
            runtime.set_state("last_gesture_time", current_time)

            # Process gesture and get appropriate information
            if gesture == "double_tap":
                result = await get_object_description(frame.image)
                if not runtime.is_cancelled():
                    await runtime.emit(result, final=True)

            elif gesture == "swipe_up":
                result = await get_nutrition_info(frame.image)
                if not runtime.is_cancelled():
                    await runtime.emit(result, final=True)

            elif gesture == "swipe_down":
                result = await get_expiry_date(frame.image)
                if not runtime.is_cancelled():
                    await runtime.emit(result, final=True)

    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()


__all__ = [
    "detect_hand",
    "detect_gesture",
    "get_object_description",
    "get_nutrition_info",
    "get_expiry_date",
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
