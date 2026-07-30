"""Recognize hand gestures from static poses or motion sequences."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_recognition"
TOOL_PROMPT = (
    "Observe hand movements to identify the gesture or short meaning. "
    "When multiple chronological frames are available, track hand shapes, "
    "orientation, location, and motion sequence from earliest to latest frames. "
    "Some gestures like waving (hello or goodbye) or moving one finger side to side "
    "(no or incorrect) require motion over time. Other gestures may be recognizable "
    "from a single stable hand pose. When only one image is available, identify any "
    "clearly recognizable static hand posture but do not infer motion that is not "
    "visible. Return one concise spoken sentence for a blind or low-vision user, for "
    "example 'The person is waving hello' or 'Hand gesture shows thumbs up.' If the "
    "available evidence is insufficient or no clear gesture is visible, return exactly "
    "'No clear gesture detected.' Do not return JSON, field names, or analysis."
)

SCENE_SIMILARITY_THRESHOLD = 0.92
GESTURE_WINDOW_SECONDS = 3.0


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


async def on_take_photo(runtime, image, input_data):
    """Analyze a single image for static hand gestures."""
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
    """Initialize state for streaming gesture analysis."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("in_flight", {})


async def on_frame(runtime, frame):
    """Analyze gesture from recent frame sequence when scene changes."""
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is not None and is_same_scene(anchor, embedding):
        return

    generation += 1
    runtime.set_state("scene_generation", generation)
    runtime.set_state("scene_anchor", embedding)

    key = ("gesture_analysis", generation)
    tasks = dict(runtime.get_state("in_flight", {}))
    existing = tasks.get(key)
    if existing is not None and not existing.done():
        return

    async def run():
        try:
            recent_frames = runtime.get_recent_frames(seconds=GESTURE_WINDOW_SECONDS)
            if not recent_frames:
                recent_frames = [runtime.get_latest_frame()]

            images = [f.image for f in recent_frames if f.image is not None]
            if not images:
                return

            response = await asyncio.to_thread(
                call_model,
                "gemini/gemini-3.1-flash-lite-preview",
                [{"role": "user", "content": TOOL_PROMPT}],
                images,
                {"timeout": 60, "num_retries": 0},
            )
            text = extract_text(response)

            if runtime.is_cancelled():
                return
            if runtime.get_state("scene_generation") != generation:
                return

            await runtime.emit(text, final=True, replace=True)
        finally:
            current = dict(runtime.get_state("in_flight", {}))
            if current.get(key) is task:
                current.pop(key, None)
                runtime.set_state("in_flight", current)

    task = asyncio.create_task(run())
    tasks[key] = task
    runtime.set_state("in_flight", tasks)


async def on_stream_stop(runtime):
    """Make outstanding gesture analysis stale when streaming stops."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.clear_state()


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
