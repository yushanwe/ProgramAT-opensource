"""Progressive scene description tool that provides increasingly detailed descriptions."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "scene_descriptor"
TOOL_PROMPT = (
    "For a blind or low-vision user viewing this scene, provide a concise description. "
    "Return one concise user-facing response suitable for spoken output. Prefer one or "
    "two short sentences in a single paragraph with no unnecessary line breaks. State "
    "only the information needed for the user's task."
)

BRIEF_PROMPT = (
    "List the main objects visible in this scene as a brief comma-separated list. "
    "Keep it to one sentence, under 15 words. Return one concise user-facing response "
    "suitable for spoken output."
)

DETAILED_PROMPT = (
    "Describe this scene in detail for a blind or low-vision user, including spatial "
    "relationships, positions, and notable details. Keep it under 40 words. Return one "
    "concise user-facing response suitable for spoken output. Prefer one or two short "
    "sentences in a single paragraph with no unnecessary line breaks."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot recognize anything in this view."
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
    """Compute cosine similarity between two embeddings."""
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = SCENE_SIMILARITY_THRESHOLD,
) -> bool:
    """Determine if two scene embeddings represent the same scene."""
    return cosine_similarity(reference, current) >= threshold


async def analyze_image(image, prompt):
    """Call the vision model with the given prompt."""
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Provide a standard scene description for one-shot usage."""
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image, TOOL_PROMPT)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("brief_emitted", False)
    runtime.set_state("detailed_emitted", False)
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """Emit brief description on new scene, then detailed if scene stays stable."""
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)
    try:
        embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    finally:
        runtime.set_state("in_flight", False)

    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))
    brief_emitted = runtime.get_state("brief_emitted", False)
    detailed_emitted = runtime.get_state("detailed_emitted", False)

    # Check if this is a new scene
    if anchor is None or not is_same_scene(anchor, embedding):
        # New scene detected - reset state
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("brief_emitted", False)
        runtime.set_state("detailed_emitted", False)
        brief_emitted = False
        detailed_emitted = False

    # Emit brief description if not already done for this scene
    if not brief_emitted:
        runtime.set_state("in_flight", True)
        try:
            text = await analyze_image(frame.image, BRIEF_PROMPT)
            if not runtime.is_cancelled() and runtime.get_state("scene_generation") == generation:
                await runtime.emit(text, partial=True)
                runtime.set_state("brief_emitted", True)
        finally:
            runtime.set_state("in_flight", False)
        return

    # If brief is done and detailed is not, emit detailed
    if brief_emitted and not detailed_emitted:
        runtime.set_state("in_flight", True)
        try:
            text = await analyze_image(frame.image, DETAILED_PROMPT)
            if not runtime.is_cancelled() and runtime.get_state("scene_generation") == generation:
                await runtime.emit(text, final=True)
                runtime.set_state("detailed_emitted", True)
        finally:
            runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up state on stream stop."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
]
