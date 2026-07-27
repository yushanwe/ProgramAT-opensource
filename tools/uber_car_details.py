"""Identify Uber car details from a camera image or live stream."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "uber_car_details"
TOOL_PROMPT = (
    "You are helping a blind or low-vision user identify their Uber. "
    "Look at this image and report any visible car details: the car color, "
    "make and model if identifiable, license plate number or partial characters, "
    "and whether there are any Uber indicators such as a phone on the dashboard, "
    "an Uber decal, or a glowing Uber sign. "
    "Give direction using clock positions (only 9 to 12 and 1 to 3) and approximate "
    "distance if the car is visible. "
    "If no car matching an Uber pickup is visible, say so plainly. "
    "Keep the response under three short sentences and speak directly to the user."
)

_SCENE_SIMILARITY_THRESHOLD = 0.985


def _compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Return a compact normalized color histogram for scene comparison."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(512, dtype=np.float32)
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]
    ).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm else histogram


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return -1.0
    return float(np.dot(a, b))


def _is_same_scene(ref: np.ndarray, cur: np.ndarray) -> bool:
    return _cosine_similarity(ref, cur) >= _SCENE_SIMILARITY_THRESHOLD


async def on_take_photo(runtime, image, input_data):
    """Analyze a single image for Uber car details."""
    del runtime, input_data
    if image is None:
        return "No camera image available."
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize per-session state for the streaming lifecycle."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)


async def on_frame(runtime, frame):
    """Emit Uber car details whenever the scene changes significantly."""
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(_compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is None or not _is_same_scene(anchor, embedding):
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)

    if runtime.get_state("analyzing_generation") == generation:
        return
    runtime.set_state("analyzing_generation", generation)

    image_snapshot = frame.image
    frame_id = frame.frame_id

    try:
        text = await asyncio.to_thread(
            lambda: extract_text(
                call_model(
                    "gemini/gemini-3.1-flash-lite-preview",
                    [{"role": "user", "content": TOOL_PROMPT}],
                    [image_snapshot],
                    {"timeout": 60, "num_retries": 0},
                )
            )
        )
    except Exception as exc:
        text = f"Could not identify Uber car details: {exc}"
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)

    if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
        return

    await runtime.emit(
        text or "No Uber car visible.",
        final=True,
        metadata={"scene_generation": generation, "frame_id": frame_id},
    )


async def on_stream_stop(runtime):
    """Advance generation so any in-flight analysis is discarded."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
