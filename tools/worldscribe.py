"""Progressive scene description that provides brief then detailed descriptions."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, call_openai_responses_model, extract_text

TOOL_NAME = "scene_descriptor"

TOOL_PROMPT = (
    "Describe the scene for a blind user. Identify objects, people, and spatial "
    "relationships. Be concise and speech-ready. If uncertain, state what cannot "
    "be determined."
)

YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

DETAILED_PROMPT = (
    "Describe this scene in more detail for a blind user. Include spatial "
    "relationships, positions, and notable characteristics. For example: "
    "'A cat sitting on top of a white desk, looking out the window.' "
    "Be concise but informative."
)

SCENE_SIMILARITY_THRESHOLD = 0.985


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized appearance embedding for scene continuity."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(512, dtype=np.float32)
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = (
        cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        .astype(np.float32)
        .reshape(-1)
    )
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


async def on_take_photo(runtime, image, input_data):
    """Provide a single standard description for one-shot requests."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize state for progressive description tracking."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("brief_done", False)
    runtime.set_state("gemini_detailed_done", False)
    runtime.set_state("gpt5_detailed_done", False)
    runtime.set_state("in_flight_brief", None)
    runtime.set_state("in_flight_gemini_detailed", None)
    runtime.set_state("in_flight_gpt5_detailed", None)


async def on_frame(runtime, frame):
    """Emit brief YOLO description, then parallel Gemini and GPT-5 detailed descriptions."""
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    # Check if scene changed
    scene_changed = anchor is None or not is_same_scene(anchor, embedding)

    if scene_changed:
        # New scene detected - reset state
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("brief_done", False)
        runtime.set_state("gemini_detailed_done", False)
        runtime.set_state("gpt5_detailed_done", False)

        # Cancel any in-flight detailed analyses from previous scene
        gemini_task = runtime.get_state("in_flight_gemini_detailed")
        if gemini_task and not gemini_task.done():
            gemini_task.cancel()
        gpt5_task = runtime.get_state("in_flight_gpt5_detailed")
        if gpt5_task and not gpt5_task.done():
            gpt5_task.cancel()
        runtime.set_state("in_flight_gemini_detailed", None)
        runtime.set_state("in_flight_gpt5_detailed", None)

    # Decide what to emit
    brief_done = runtime.get_state("brief_done", False)
    gemini_detailed_done = runtime.get_state("gemini_detailed_done", False)
    gpt5_detailed_done = runtime.get_state("gpt5_detailed_done", False)

    if not brief_done:
        # Emit brief YOLO-based description for new or unprocessed scene
        brief_task = runtime.get_state("in_flight_brief")
        if brief_task and not brief_task.done():
            return  # Already analyzing

        async def run_brief():
            try:
                from PIL import Image
                from ultralytics import YOLO
                import io

                # Use YOLO for object detection
                pil_image = Image.fromarray(frame.image[:, :, ::-1].copy()).convert("RGB")
                model = YOLO("yolo11n.pt")
                detections = []
                for result in model(pil_image, verbose=False):
                    for box in result.boxes:
                        label = str(result.names[int(box.cls[0])])
                        detections.append(label)

                # Format as brief sentence
                if detections:
                    unique_objects = []
                    seen = set()
                    for obj in detections:
                        if obj.lower() not in seen:
                            unique_objects.append(obj)
                            seen.add(obj.lower())
                    if len(unique_objects) == 1:
                        text = f"A {unique_objects[0]}."
                    elif len(unique_objects) == 2:
                        text = f"A {unique_objects[0]} and a {unique_objects[1]}."
                    else:
                        text = ", ".join(f"a {obj}" for obj in unique_objects[:-1])
                        text = f"{text}, and a {unique_objects[-1]}."
                else:
                    text = "No objects detected."

                if (
                    runtime.is_cancelled()
                    or runtime.get_state("scene_generation") != generation
                ):
                    return
                await runtime.emit(text, final=False, replace=True)
                runtime.set_state("brief_done", True)
            finally:
                if runtime.get_state("in_flight_brief") is task:
                    runtime.set_state("in_flight_brief", None)

        task = asyncio.create_task(run_brief())
        runtime.set_state("in_flight_brief", task)

    elif not gemini_detailed_done or not gpt5_detailed_done:
        # Scene is stable - launch parallel detailed descriptions

        # Launch Gemini detailed if not done
        if not gemini_detailed_done:
            gemini_task = runtime.get_state("in_flight_gemini_detailed")
            if not gemini_task or gemini_task.done():
                async def run_gemini_detailed():
                    try:
                        response = await asyncio.to_thread(
                            call_model,
                            "gemini/gemini-3.1-flash-lite-preview",
                            [{"role": "user", "content": DETAILED_PROMPT}],
                            [frame.image],
                            {"timeout": 60, "num_retries": 0},
                        )
                        text = extract_text(response)
                        if (
                            runtime.is_cancelled()
                            or runtime.get_state("scene_generation") != generation
                        ):
                            return
                        # Emit as non-final if GPT-5 is still pending
                        gpt5_done = runtime.get_state("gpt5_detailed_done", False)
                        await runtime.emit(text, final=gpt5_done, replace=True)
                        runtime.set_state("gemini_detailed_done", True)
                    finally:
                        if runtime.get_state("in_flight_gemini_detailed") is task:
                            runtime.set_state("in_flight_gemini_detailed", None)

                task = asyncio.create_task(run_gemini_detailed())
                runtime.set_state("in_flight_gemini_detailed", task)

        # Launch GPT-5 detailed if not done
        if not gpt5_detailed_done:
            gpt5_task = runtime.get_state("in_flight_gpt5_detailed")
            if not gpt5_task or gpt5_task.done():
                async def run_gpt5_detailed():
                    try:
                        text = await asyncio.to_thread(
                            call_openai_responses_model,
                            "gpt-5",
                            DETAILED_PROMPT,
                            frame.image,
                            reasoning_effort="medium",
                            timeout=90,
                        )
                        if (
                            runtime.is_cancelled()
                            or runtime.get_state("scene_generation") != generation
                        ):
                            return
                        # Emit as final - this is the most accurate model
                        await runtime.emit(text, final=True, replace=True)
                        runtime.set_state("gpt5_detailed_done", True)
                    finally:
                        if runtime.get_state("in_flight_gpt5_detailed") is task:
                            runtime.set_state("in_flight_gpt5_detailed", None)

                task = asyncio.create_task(run_gpt5_detailed())
                runtime.set_state("in_flight_gpt5_detailed", task)


async def on_stream_stop(runtime):
    """Clean up state when streaming stops."""
    generation = int(runtime.get_state("scene_generation", 0))
    runtime.set_state("scene_generation", generation + 1)

    # Cancel any in-flight tasks
    brief_task = runtime.get_state("in_flight_brief")
    if brief_task and not brief_task.done():
        brief_task.cancel()
    gemini_task = runtime.get_state("in_flight_gemini_detailed")
    if gemini_task and not gemini_task.done():
        gemini_task.cancel()
    gpt5_task = runtime.get_state("in_flight_gpt5_detailed")
    if gpt5_task and not gpt5_task.done():
        gpt5_task.cancel()

    runtime.set_state("in_flight_brief", None)
    runtime.set_state("in_flight_gemini_detailed", None)
    runtime.set_state("in_flight_gpt5_detailed", None)


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]