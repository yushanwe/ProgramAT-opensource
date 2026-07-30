"""Indoor Exit Locator

Assists blind and low-vision users in finding the nearest exit in unfamiliar
indoor environments by detecting exit signs, doors, and doorways, then providing
step-by-step navigation guidance.

Features:
- Detects exit signs, emergency exits, and doors
- Provides clock-face navigation (9-12, 1-3)
- Gives step-by-step directions to reach the nearest exit
- Concise, speech-ready output for audio accessibility
- Supports both Take Photo (single image) and Streaming (latest frame) modes
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "indoor_exit_locator"
TOOL_PROMPT = (
    "Assist a blind user in finding the nearest exit in an unfamiliar indoor environment. "
    "1. Identify all visible exit signs, emergency exit indicators, doors, and doorways. "
    "2. Determine which is the most likely exit based on exit signage or door characteristics. "
    "3. Report the clock-face direction (9-12 or 1-3 only) and provide step-by-step guidance "
    "to reach it safely. Use stable structural cues, not color-only landmarks. "
    "If no exit is visible or evidence is insufficient, state that clearly."
)

SCENE_SIMILARITY_THRESHOLD = 0.98


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
    """Analyze a single photo to locate the nearest exit."""
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
    """Initialize streaming session state."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)


async def on_frame(runtime, frame):
    """Process the latest frame when the scene changes."""
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    # Check if this is the same scene we already analyzed
    if anchor is not None and is_same_scene(anchor, embedding):
        if runtime.get_state("analyzing_generation") == generation:
            return
    else:
        # New scene detected
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)

    # Skip if already analyzing this generation
    if runtime.get_state("analyzing_generation") == generation:
        return

    runtime.set_state("analyzing_generation", generation)

    try:
        # Check cancellation before calling model
        if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
            return

        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            [frame.image],
            {"timeout": 60, "num_retries": 0},
        )

        # Check cancellation before emitting
        if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
            return

        text = extract_text(response)
        if text:
            await runtime.emit(
                text,
                final=True,
                metadata={
                    "scene_generation": generation,
                    "frame_id": frame.frame_id,
                },
            )
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    """Mark current scene work as stale when streaming stops."""
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
