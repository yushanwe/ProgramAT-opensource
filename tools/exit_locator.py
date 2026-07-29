"""Locate the nearest visible exit and give concise spoken guidance."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_locator"
TOOL_PROMPT = (
    "Assist a blind or low-vision user in finding the nearest indoor exit. "
    "1. Look for visible exit signs, exit doors, building entrances, or doors that clearly lead out. "
    "2. Choose the most likely nearest visible exit based only on what is visible. "
    "3. Give concise step-by-step spoken directions using body-relative wording or clock direction only from 9 to 12 or 1 to 3, plus brief uncertainty if the evidence is weak. "
    "If no likely exit is visible, say that clearly."
)

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
SCENE_SIMILARITY_THRESHOLD = 0.985


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized embedding for scene-change detection."""
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
    """Measure similarity between two normalized scene embeddings."""
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = SCENE_SIMILARITY_THRESHOLD,
) -> bool:
    """Return True when two frames are visually similar enough to skip re-analysis."""
    return cosine_similarity(reference, current) >= threshold


def analyze_exit(image: np.ndarray) -> str:
    """Run the vision model on one image and normalize empty responses."""
    response = call_model(
        MODEL_NAME,
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or "I cannot confirm a visible exit from this view."


async def on_take_photo(runtime, image, input_data):
    """Analyze the current image and return concise exit guidance."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    return await asyncio.to_thread(analyze_exit, image)


async def on_stream_start(runtime, input_data):
    """Initialize per-session scene gating state."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("last_emitted_text", None)


async def on_frame(runtime, frame):
    """Analyze only changed scenes and emit updated guidance."""
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

    if runtime.get_state("analyzing_generation") == generation:
        return
    runtime.set_state("analyzing_generation", generation)

    try:
        text = await asyncio.to_thread(analyze_exit, frame.image)
        if runtime.is_cancelled():
            return
        if runtime.get_state("scene_generation") != generation:
            return

        last_text = runtime.get_state("last_emitted_text")
        if text != last_text:
            runtime.set_state("last_emitted_text", text)
            await runtime.emit(
                text,
                final=True,
                replace=True,
                metadata={
                    "scene_generation": generation,
                    "frame_id": frame.frame_id,
                },
            )
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    """Invalidate in-flight work for this session."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
