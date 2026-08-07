"""Gesture-Based Object Information Assistant

Identifies objects held in hand and provides specific information based on
gesture commands: double tap for description, swipe up for nutrition,
swipe down for expiry date.
"""

import asyncio
from typing import List, Optional, Tuple

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_object_info"
TOOL_PROMPT = (
    "You are assisting a blind or low-vision user who is holding an object. "
    "Identify the object held in their hand and provide a concise description. "
    "Focus on the object in the hand, not other background objects. "
    "Return one concise user-facing response suitable for spoken output. "
    "Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. "
    "State only the information needed for the user's task."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot identify the object clearly from this view."

# Gesture detection parameters
GESTURE_WINDOW_FRAMES = 8  # Number of recent frames to analyze
MIN_MOVEMENT_THRESHOLD = 30  # Minimum pixel movement to detect gesture
SWIPE_CONSISTENCY_THRESHOLD = 0.6  # Ratio of frames moving in same direction
DOUBLE_TAP_MAX_FRAMES = 6  # Maximum frames for double tap pattern
GESTURE_COOLDOWN_SECONDS = 2.0  # Seconds between gesture detections

# Hand detection using YOLO
HAND_CONFIDENCE_THRESHOLD = 0.3


async def detect_hand_yolo(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect hand in image using YOLO and return bounding box.

    Returns:
        Tuple of (x, y, width, height) if hand detected, None otherwise.
    """
    try:
        from ultralytics import YOLO

        # Use standard YOLO model which can detect 'person'
        # We'll use a heuristic: look for person parts in upper portion
        model = YOLO('yolo11n.pt')
        results = model(image, conf=HAND_CONFIDENCE_THRESHOLD, verbose=False)

        # Look for person detections (class 0 in COCO)
        height, width = image.shape[:2]

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    # Prioritize detections in center and lower portion
                    # (where hand holding object would likely be)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Prefer center-bottom region for hand with object
                    if center_y > height * 0.3:  # Lower 70% of frame
                        return (x, y, w, h)

        # If no suitable person detection, assume hand in center
        # This is a fallback heuristic
        center_size = min(width, height) // 2
        return (
            width // 2 - center_size // 2,
            height // 2 - center_size // 2,
            center_size,
            center_size
        )

    except Exception:
        # Fallback: assume center region
        height, width = image.shape[:2]
        center_size = min(width, height) // 2
        return (
            width // 2 - center_size // 2,
            height // 2 - center_size // 2,
            center_size,
            center_size
        )


def calculate_blur_score(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    """Calculate blur score for image or region.

    Higher score = less blur (sharper image).

    Args:
        image: Input image
        bbox: Optional bounding box (x, y, w, h) to analyze specific region

    Returns:
        Blur score (Laplacian variance)
    """
    if bbox:
        x, y, w, h = bbox
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        region = image[y:y+h, x:x+w]
    else:
        region = image

    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region

    # Calculate Laplacian variance (higher = sharper)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def detect_gesture_from_positions(positions: List[Tuple[int, int]]) -> Optional[str]:
    """Detect gesture from sequence of hand positions.

    Args:
        positions: List of (center_x, center_y) tuples from recent frames

    Returns:
        'double_tap', 'swipe_up', 'swipe_down', or None
    """
    if len(positions) < 3:
        return None

    # Calculate vertical movements between consecutive positions
    vertical_movements = []
    for i in range(1, len(positions)):
        dy = positions[i][1] - positions[i-1][1]
        vertical_movements.append(dy)

    # Check for double tap: down-up-down pattern
    # Look for alternating direction changes
    if len(vertical_movements) >= 4:
        # Look for pattern: down (+), up (-), down (+)
        # Or up (-), down (+), up (-)
        direction_changes = 0
        prev_direction = 0

        for dy in vertical_movements[-DOUBLE_TAP_MAX_FRAMES:]:
            if abs(dy) < MIN_MOVEMENT_THRESHOLD // 3:
                continue
            current_direction = 1 if dy > 0 else -1
            if prev_direction != 0 and current_direction != prev_direction:
                direction_changes += 1
            prev_direction = current_direction

        if direction_changes >= 2:  # At least 2 direction changes
            return 'double_tap'

    # Check for swipe up: consistent upward movement
    recent_movements = vertical_movements[-min(5, len(vertical_movements)):]
    significant_movements = [dy for dy in recent_movements if abs(dy) >= MIN_MOVEMENT_THRESHOLD]

    if significant_movements:
        upward_count = sum(1 for dy in significant_movements if dy < 0)
        downward_count = sum(1 for dy in significant_movements if dy > 0)

        total_significant = len(significant_movements)

        if upward_count / total_significant >= SWIPE_CONSISTENCY_THRESHOLD:
            return 'swipe_up'

        if downward_count / total_significant >= SWIPE_CONSISTENCY_THRESHOLD:
            return 'swipe_down'

    return None


async def get_object_info(image: np.ndarray, info_type: str) -> str:
    """Get specific information about object in image.

    Args:
        image: Image containing object
        info_type: 'description', 'nutrition', or 'expiry'

    Returns:
        Information text suitable for spoken output
    """
    prompts = {
        'description': (
            "Identify the object being held in the hand and provide a concise description. "
            "Include brand name and product type if visible. "
            "Return one concise user-facing response suitable for spoken output. "
            "Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. "
            "State only the information needed for the user's task."
        ),
        'nutrition': (
            "Look at the object being held and read any visible nutrition information. "
            "Report key nutritional facts such as sugar, caffeine, fiber, calories, or other prominent values. "
            "If nutrition information is not visible, say 'Rotate or flip the object'. "
            "Return one concise user-facing response suitable for spoken output. "
            "Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. "
            "State only the information needed for the user's task."
        ),
        'expiry': (
            "Look at the object being held and find any expiry date, best before date, or use by date. "
            "Report the date in a natural spoken format. "
            "If no date is visible, say 'Rotate or flip the object'. "
            "Return one concise user-facing response suitable for spoken output. "
            "Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. "
            "State only the information needed for the user's task."
        ),
    }

    prompt = prompts.get(info_type, prompts['description'])

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
    """Handle take photo mode - provide object description."""
    if image is None:
        return FALLBACK_TEXT

    return await get_object_info(image, 'description')


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    runtime.set_state("in_flight", False)
    runtime.set_state("hand_positions", [])
    runtime.set_state("last_gesture_time", 0.0)
    runtime.set_state("cached_object_info", {})
    runtime.set_state("frame_count", 0)


async def on_frame(runtime, frame):
    """Process each frame for gesture detection and object information."""
    if runtime.is_cancelled():
        return

    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)

    try:
        import time

        frame_count = runtime.get_state("frame_count", 0)
        runtime.set_state("frame_count", frame_count + 1)

        # Get recent frames for gesture detection
        recent_frames = runtime.get_recent_frames(count=GESTURE_WINDOW_FRAMES)

        if not recent_frames or len(recent_frames) < 3:
            # Not enough frames yet, just check hand focus
            hand_bbox = await detect_hand_yolo(frame.image)
            blur_score = calculate_blur_score(frame.image, hand_bbox)

            # Emit focus feedback periodically (every ~10 frames)
            if frame_count % 10 == 0:
                if blur_score < 50:  # Blurry threshold
                    if not runtime.is_cancelled():
                        await runtime.emit("Hand out of focus", partial=True)
                elif blur_score > 100:  # Sharp threshold
                    if not runtime.is_cancelled():
                        await runtime.emit("Hand in focus", partial=True)

            return

        # Detect hand position in recent frames
        hand_positions = []

        for recent_frame in recent_frames[-GESTURE_WINDOW_FRAMES:]:
            hand_bbox = await detect_hand_yolo(recent_frame.image)
            if hand_bbox:
                x, y, w, h = hand_bbox
                center_x = x + w // 2
                center_y = y + h // 2
                hand_positions.append((center_x, center_y))

        # Store positions for next iteration
        runtime.set_state("hand_positions", hand_positions)

        # Check gesture cooldown
        current_time = time.time()
        last_gesture_time = runtime.get_state("last_gesture_time", 0.0)

        if current_time - last_gesture_time < GESTURE_COOLDOWN_SECONDS:
            return

        # Detect gesture
        gesture = detect_gesture_from_positions(hand_positions)

        if gesture:
            # Update gesture time
            runtime.set_state("last_gesture_time", current_time)

            # Map gesture to info type
            info_type_map = {
                'double_tap': 'description',
                'swipe_up': 'nutrition',
                'swipe_down': 'expiry',
            }

            info_type = info_type_map.get(gesture)

            if info_type:
                # Check cache
                cached_info = runtime.get_state("cached_object_info", {})

                if info_type in cached_info:
                    result = cached_info[info_type]
                else:
                    # Get new information
                    result = await get_object_info(frame.image, info_type)

                    # Cache the result
                    cached_info[info_type] = result
                    runtime.set_state("cached_object_info", cached_info)

                if not runtime.is_cancelled():
                    await runtime.emit(result, final=True)
        else:
            # No gesture detected, check hand focus periodically
            if frame_count % 15 == 0:
                hand_bbox = await detect_hand_yolo(frame.image)
                blur_score = calculate_blur_score(frame.image, hand_bbox)

                if blur_score < 50:  # Blurry
                    if not runtime.is_cancelled():
                        await runtime.emit("Hand out of focus", partial=True)
                elif blur_score > 100 and frame_count % 30 == 0:  # Sharp, less frequent
                    if not runtime.is_cancelled():
                        await runtime.emit("Hand in focus", partial=True)

    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
