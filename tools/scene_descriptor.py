"""Progressive scene description: YOLO brief list then Gemini detailed description."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "scene_descriptor"

TOOL_PROMPT = (
    "Describe the scene for a blind user. "
    "List the main objects present in a brief, concise manner. "
    "Use body-relative directions when describing locations. "
    "Do not use colors or other visual-only landmarks as the sole navigation cue. "
    "Be concise and speech-ready."
)

SCENE_SIMILARITY_THRESHOLD = 0.985


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized appearance embedding for scene continuity."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(512, dtype=np.float32)
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]
    ).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm else histogram


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = SCENE_SIMILARITY_THRESHOLD,
) -> bool:
    return cosine_similarity(reference, current) >= threshold


def _get_yolo_description(image: np.ndarray) -> str:
    """Get brief YOLO-based object description."""
    try:
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        results = model(image, conf=0.5, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = _get_coco_class_name(class_id)
                detections.append(class_name)

        if not detections:
            return "No objects detected."

        # Count objects
        from collections import Counter
        counts = Counter(detections)

        # Build concise description
        parts = []
        for obj, count in counts.most_common():
            if count == 1:
                parts.append(f"1 {obj}")
            else:
                parts.append(f"{count} {obj}s")

        return "I see: " + ", ".join(parts)
    except Exception:
        return "Unable to detect objects."


def _get_coco_class_name(class_id: int) -> str:
    """Get COCO class name from class ID."""
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"object_{class_id}"


async def on_take_photo(runtime, image, input_data):
    """Return brief YOLO-based scene description for a single photo."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."

    # Use YOLO for brief description
    description = await asyncio.to_thread(_get_yolo_description, image)
    return description or "Unable to describe the scene."


async def on_stream_start(runtime, input_data):
    """Initialize streaming state for progressive scene description."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("progression_stage", None)
    runtime.set_state("analyzing", False)


async def on_frame(runtime, frame):
    """
    Provide progressive scene description:
    1. Brief object list on new scene
    2. Detailed description if scene remains stable
    """
    if runtime.is_cancelled():
        return

    if runtime.get_state("analyzing"):
        return

    runtime.set_state("analyzing", True)

    try:
        await _analyze_frame(runtime, frame)
    finally:
        runtime.set_state("analyzing", False)


async def _analyze_frame(runtime, frame):
    """Helper that handles all awaited work for scene analysis."""
    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))
    current_stage = runtime.get_state("progression_stage")

    # Check if scene has changed
    if anchor is not None and is_same_scene(anchor, embedding):
        # Same scene - check if we can progress to detailed description
        if current_stage == "yolo_brief":
            # Escalate to Gemini for detailed description
            if runtime.is_cancelled():
                return

            response = await asyncio.to_thread(
                call_model,
                "gemini/gemini-3.1-flash-lite-preview",
                [{"role": "user", "content": TOOL_PROMPT}],
                [frame.image],
                {"timeout": 60, "num_retries": 0},
            )

            if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
                return

            text = extract_text(response).strip()
            text = text or "Unable to provide detailed description."

            await runtime.emit(
                text,
                final=True,
                metadata={
                    "stage": "gemini_detailed",
                    "scene_generation": generation,
                    "frame_id": frame.frame_id,
                },
            )
            runtime.set_state("progression_stage", "gemini_detailed")
        # If already at detailed stage, do nothing
        return
    else:
        # New scene detected
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("progression_stage", None)

    # Emit brief YOLO description for new scene
    if runtime.is_cancelled():
        return

    yolo_description = await asyncio.to_thread(_get_yolo_description, frame.image)

    if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
        return

    text = yolo_description or "Unable to describe the scene."

    await runtime.emit(
        text,
        final=True,
        metadata={
            "stage": "yolo_brief",
            "scene_generation": generation,
            "frame_id": frame.frame_id,
        },
    )
    runtime.set_state("progression_stage", "yolo_brief")


async def on_stream_stop(runtime):
    """Invalidate scene state when streaming stops."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing", False)
    runtime.set_state("progression_stage", None)


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
    "TOOL_NAME",
    "TOOL_PROMPT",
]
