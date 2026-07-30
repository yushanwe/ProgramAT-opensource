"""
Scene Descriptor Tool for Blind Users

Progressive scene description that provides brief object listing first,
then detailed descriptions when the user focuses on the same scene.

Features:
- Initial brief enumeration of objects in scene
- Detailed description on prolonged camera focus
- Scene change detection
- Audio-optimized output for blind and low-vision users

Example Usage:
First frame: "A desk, a cabinet, a printer, and a cat"
After prolonged focus: "A cat sitting on top of a white desk, looking out the window"

This tool implements both Take Photo and Streaming modes with shared prompt.
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from litellm_utils import call_model, extract_text

TOOL_NAME = "scene_descriptor"
TOOL_PROMPT = (
    "Describe the scene for a blind user. "
    "1. List key objects briefly when the scene is first viewed. "
    "2. Provide detailed description with spatial relationships and actions when the same scene "
    "is viewed for several frames. "
    "Keep output concise and speech-ready. If the scene is unclear, state what cannot be determined."
)

# State tracking for progressive description
STATE_SCENE_HASH = "scene_hash"
STATE_FRAME_COUNT = "frame_count"
STATE_LAST_DESCRIPTION = "last_description"

# Configuration
FRAMES_FOR_DETAILED = 6  # Number of consistent frames before detailed description (increased from 3)
SCENE_CHANGE_THRESHOLD = 0.4  # Scene similarity threshold (lower = more different)
YOLO_CONFIDENCE = 0.3  # Confidence threshold for YOLO object detection (lowered for YOLO-World)


def compute_scene_hash(image_base64: str) -> str:
    """
    Compute a simple hash of the image to detect scene changes.

    Args:
        image_base64: Base64 encoded image

    Returns:
        Hash string representing the scene
    """
    # Use first and last 100 characters as a simple hash proxy
    # This is a heuristic - real scene change would need more sophisticated comparison
    if len(image_base64) > 200:
        return image_base64[:100] + image_base64[-100:]
    return image_base64


def scene_similarity(hash1: str, hash2: str) -> float:
    """
    Estimate similarity between two scene hashes.

    Args:
        hash1: First scene hash
        hash2: Second scene hash

    Returns:
        Similarity score (0.0 to 1.0), where 1.0 means identical
    """
    if not hash1 or not hash2:
        return 0.0

    if hash1 == hash2:
        return 1.0

    # Simple character overlap similarity
    matches = sum(1 for c1, c2 in zip(hash1, hash2) if c1 == c2)
    max_len = max(len(hash1), len(hash2))

    return matches / max_len if max_len > 0 else 0.0


def detect_objects_yolo(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect objects using YOLO-World for open-vocabulary detection.

    YOLO-World can detect objects beyond COCO classes, making it much more
    powerful for general scene understanding.

    Args:
        image: Input image as numpy array

    Returns:
        List of detection dictionaries with class_name and confidence
    """
    detections = []

    try:
        from ultralytics import YOLO

        # Load YOLO-World model (yolov8s-world.pt for open-vocabulary detection)
        # This model can detect many more objects than standard COCO-trained models
        model = YOLO('yolov8s-world.pt')

        # Set custom classes for YOLO-World to look for
        # These are common indoor/outdoor objects a blind user might encounter
        model.set_classes([
            "person", "child", "man", "woman", "face", "hand",
            "chair", "table", "desk", "couch", "sofa", "bed", "cabinet", "shelf",
            "door", "window", "wall", "floor", "ceiling", "stairs", "elevator",
            "phone", "laptop", "computer", "keyboard", "mouse", "monitor", "screen",
            "cup", "mug", "bottle", "glass", "plate", "bowl", "fork", "knife", "spoon",
            "book", "paper", "pen", "pencil", "bag", "backpack", "purse",
            "car", "bus", "truck", "bicycle", "motorcycle", "traffic light", "sign",
            "tree", "plant", "flower", "grass", "bench", "sidewalk", "road",
            "cat", "dog", "bird", "animal",
            "light", "lamp", "clock", "picture", "mirror", "television", "tv"
        ])

        # Run inference
        results = model(image, conf=YOLO_CONFIDENCE, verbose=False)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])

                # Get class name from the model
                class_id = int(box.cls[0])
                class_name = model.names[class_id] if hasattr(model, 'names') else f"object_{class_id}"

                detections.append({
                    'class_name': class_name,
                    'confidence': confidence
                })

    except Exception:
        # YOLO-World not available or error - return empty list
        pass

    return detections


def create_brief_description_from_yolo(detections: List[Dict[str, Any]]) -> str:
    """
    Create a concise comma-separated list from YOLO detections.

    Args:
        detections: List of YOLO detection dictionaries

    Returns:
        Brief description string like "A desk, a chair, a laptop, and a coffee mug"
    """
    if not detections:
        return ""

    # Count objects by class
    from collections import defaultdict
    counts = defaultdict(int)
    for det in detections:
        counts[det['class_name']] += 1

    # Build object list with counts
    object_parts = []
    for class_name, count in sorted(counts.items()):
        if count == 1:
            object_parts.append(f"a {class_name}")
        else:
            # Pluralize (simple approach)
            plural = f"{class_name}s" if not class_name.endswith('s') else class_name
            object_parts.append(f"{count} {plural}")

    # Format as natural list
    if len(object_parts) == 0:
        return ""
    elif len(object_parts) == 1:
        return object_parts[0].capitalize()
    elif len(object_parts) == 2:
        return f"{object_parts[0].capitalize()} and {object_parts[1]}"
    else:
        # Multiple items: "A, B, C, and D"
        last_item = object_parts[-1]
        other_items = ", ".join(object_parts[:-1])
        return f"{other_items.capitalize()}, and {last_item}"


async def on_take_photo(runtime, image, input_data):
    """
    Take Photo mode: Provide brief object listing using YOLO-World.

    Always returns YOLO results immediately - no Gemini fallback.

    Args:
        runtime: Tool runtime with state and emission
        image: Current image from camera
        input_data: Optional input configuration

    Returns:
        Brief scene description
    """
    # Use YOLO-World for fast object detection
    detections = await asyncio.to_thread(detect_objects_yolo, image)
    description = create_brief_description_from_yolo(detections)

    # Return YOLO description or "nothing detected" - no Gemini fallback
    if description:
        return description
    else:
        return "No clear objects detected"


async def on_stream_start(runtime, input_data):
    """
    Initialize streaming session.

    Args:
        runtime: Tool runtime
        input_data: Optional input configuration
    """
    # Clear state at start of new stream
    runtime.clear_state()


async def on_frame(runtime, frame):
    """
    Process each streaming frame with progressive description logic.

    Args:
        runtime: Tool runtime
        frame: Current frame object with image, timestamp, etc.
    """
    if runtime.is_cancelled():
        return

    # Get current scene hash
    current_hash = compute_scene_hash(frame.image_base64)

    # Get previous state
    prev_hash = runtime.get_state(STATE_SCENE_HASH)
    frame_count = runtime.get_state(STATE_FRAME_COUNT, 0)

    # Check if scene changed
    if prev_hash:
        similarity = scene_similarity(current_hash, prev_hash)
        if similarity < SCENE_CHANGE_THRESHOLD:
            # Scene changed - reset and give brief description
            runtime.clear_state()
            frame_count = 0

    # Increment frame count for this scene
    frame_count += 1
    runtime.set_state(STATE_FRAME_COUNT, frame_count)
    runtime.set_state(STATE_SCENE_HASH, current_hash)

    # Decide what type of description to provide
    if frame_count == 1:
        # First frame of this scene - use YOLO-World for immediate brief object listing
        detections = await asyncio.to_thread(detect_objects_yolo, frame.image)
        description = create_brief_description_from_yolo(detections)

        # Return YOLO description immediately - no Gemini fallback
        if not description:
            description = "No clear objects detected"

        # Mark as brief description
        runtime.set_state(STATE_LAST_DESCRIPTION, f"BRIEF:{description}")
        await runtime.emit(description, final=True)
        return

    elif frame_count >= FRAMES_FOR_DETAILED:
        # Prolonged focus - provide detailed description using Gemini (only once)
        last_desc = runtime.get_state(STATE_LAST_DESCRIPTION)
        if last_desc and last_desc.startswith("DETAILED"):
            # Already gave detailed description, skip to avoid repetition
            return

        prompt = (
            "Provide a detailed scene description for a blind user. "
            "Include spatial relationships, object states, and visible actions. "
            "Keep it under 30 words and speech-ready. "
            "Example: 'A cat sitting on top of a white desk near the window, looking outside.'"
        )

        # Use Gemini for detailed description
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": prompt}],
            [frame.image],
        )

        description = extract_text(response).strip()
        if len(description) > 300:
            description = description[:300].rsplit(' ', 1)[0] + '...'

        # Mark as detailed description
        runtime.set_state(STATE_LAST_DESCRIPTION, f"DETAILED:{description}")
        await runtime.emit(description, final=True)
        return

    else:
        # In between - skip to avoid spam
        return


async def on_stream_stop(runtime):
    """
    Clean up streaming session.

    Args:
        runtime: Tool runtime
    """
    # Clear state on stream stop for fresh start next time
    runtime.clear_state()
