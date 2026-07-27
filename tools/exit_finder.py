"""Find indoor exits for blind and low-vision users with scene-gated streaming."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "exit_finder"
TOOL_PROMPT = (
    "Assist a blind user in finding an indoor exit. "
    "1. Find visible doors, doorways, or exit signs. "
    "2. Decide which one is the most likely exit. "
    "3. Report its clock-face direction (use only 9–12 or 1–3) and give concise guidance toward it. "
    "If no exit is visible, say so plainly."
)

SCENE_SIMILARITY_THRESHOLD = 0.985


def _compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized appearance embedding for scene-change detection."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(512, dtype=np.float32)
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]
    ).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm else histogram


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def _is_same_scene(reference: np.ndarray, current: np.ndarray) -> bool:
    return _cosine_similarity(reference, current) >= SCENE_SIMILARITY_THRESHOLD


async def on_take_photo(runtime, image, input_data):
    """Analyze one image to locate an exit."""
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
    """Initialize per-session state for scene gating."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("completed_generation", None)


async def on_frame(runtime, frame):
    """Analyze each distinct scene once and emit the exit-finding result."""
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(_compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is not None and _is_same_scene(anchor, embedding):
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
        text = await asyncio.to_thread(
            lambda: extract_text(
                call_model(
                    "gemini/gemini-3.1-flash-lite-preview",
                    [{"role": "user", "content": TOOL_PROMPT}],
                    [frame.image],
                    {"timeout": 60, "num_retries": 0},
                )
            )
        )
        if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
            return
        if text:
            await runtime.emit(
                text,
                final=True,
                metadata={
                    "model": "gemini/gemini-3.1-flash-lite-preview",
                    "scene_generation": generation,
                    "frame_id": frame.frame_id,
                },
            )
        if runtime.get_state("scene_generation") == generation:
            runtime.set_state("completed_generation", generation)
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    """Invalidate any in-progress analysis before the backend cancels tasks."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing_generation", None)


