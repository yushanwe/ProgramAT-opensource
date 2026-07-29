"""Identify the Uber car in the middle of the scene."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, call_openai_responses_model, extract_text

TOOL_NAME = "find_uber_car_info"
TOOL_PROMPT = (
    "Identify the car in the middle of the scene and report its make, model, "
    "color, and license plate in one concise spoken sentence. If any detail is "
    "unclear or not visible, say which detail is uncertain instead of guessing."
)

FAST_MODEL = "gemini/gemini-3.1-flash-lite-preview"
STRONG_MODEL = "gpt-5"
SCENE_SIMILARITY_THRESHOLD = 0.985


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized embedding for scene change detection."""
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


def _needs_escalation(text: str) -> bool:
    if not text:
        return True
    normalized = text.lower()
    uncertain_markers = (
        "unclear",
        "uncertain",
        "not visible",
        "not readable",
        "cannot",
        "can't",
        "unable",
        "unknown",
        "sorry",
    )
    return any(marker in normalized for marker in uncertain_markers)


def _call_fast_model(image: np.ndarray) -> str:
    response = call_model(
        FAST_MODEL,
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response).strip()


def _call_strong_model(image: np.ndarray) -> str:
    return call_openai_responses_model(
        STRONG_MODEL,
        TOOL_PROMPT,
        image,
        reasoning_effort="medium",
        timeout=60,
    ).strip()


def _analyze_image(image: np.ndarray, prefer_fast_only: bool) -> str:
    text = _call_fast_model(image)
    if prefer_fast_only or not _needs_escalation(text):
        return text or "I can't determine the car details from this view."

    strong_text = _call_strong_model(image)
    return strong_text or text or "I can't determine the car details from this view."


async def on_take_photo(runtime, image, input_data):
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    return await asyncio.to_thread(_analyze_image, image, False)


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("completed_generation", None)


async def on_frame(runtime, frame):
    if runtime.is_cancelled() or frame is None or frame.image is None:
        return

    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is not None and cosine_similarity(anchor, embedding) >= SCENE_SIMILARITY_THRESHOLD:
        if (
            runtime.get_state("analyzing_generation") == generation
            or runtime.get_state("completed_generation") == generation
        ):
            return
    else:
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("completed_generation", None)

    if runtime.get_state("analyzing_generation") == generation:
        return

    runtime.set_state("analyzing_generation", generation)
    try:
        text = await asyncio.to_thread(_analyze_image, frame.image, True)
        if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
            return
        runtime.set_state("completed_generation", generation)
        await runtime.emit(
            text or "I can't determine the car details from this view.",
            final=True,
            metadata={"frame_id": frame.frame_id, "scene_generation": generation},
        )
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "TOOL_NAME",
    "TOOL_PROMPT",
    "compute_scene_embedding",
    "cosine_similarity",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
