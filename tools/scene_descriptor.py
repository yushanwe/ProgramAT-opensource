"""Progressive scene description: brief object list then detailed description."""

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

DETAILED_PROMPT = (
    "Provide a detailed description of the scene for a blind user. "
    "Describe the spatial relationships between objects, their positions, and any notable details. "
    "Use body-relative directions such as left, right, straight ahead, or clock-face positions (9-12 and 1-3 only). "
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


async def on_take_photo(runtime, image, input_data):
    """Return brief scene description for a single photo."""
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
    text = extract_text(response).strip()
    return text or "Unable to describe the scene."


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
        if current_stage == "brief":
            # Emit detailed description
            if runtime.is_cancelled():
                return

            response = await asyncio.to_thread(
                call_model,
                "gemini/gemini-3.1-flash-lite-preview",
                [{"role": "user", "content": DETAILED_PROMPT}],
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
                    "stage": "detailed",
                    "scene_generation": generation,
                    "frame_id": frame.frame_id,
                },
            )
            runtime.set_state("progression_stage", "detailed")
        # If already at detailed stage, do nothing
        return
    else:
        # New scene detected
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("progression_stage", None)

    # Emit brief description for new scene
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
    text = text or "Unable to describe the scene."

    await runtime.emit(
        text,
        final=True,
        metadata={
            "stage": "brief",
            "scene_generation": generation,
            "frame_id": frame.frame_id,
        },
    )
    runtime.set_state("progression_stage", "brief")


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
]
