"""Progressive scene description that starts brief and adds detail over time."""

from __future__ import annotations

import asyncio
import logging

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

logger = logging.getLogger(__name__)

TOOL_NAME = "scene_descriptor"
TOOL_PROMPT = (
    "Describe this scene for a blind user. Respond in two parts: "
    "BRIEF: List the main objects you see, comma-separated (e.g., 'A desk, a cabinet, a printer, and a cat'). "
    "DETAILED: Provide a fuller description with spatial relationships and notable details "
    "(e.g., 'A cat sitting on top of a white desk, looking out the window'). "
    "If the scene is unclear or nothing is visible, state that plainly."
)

SCENE_SIMILARITY_THRESHOLD = 0.985
DETAIL_DELAY_SECONDS = 2.0


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


def extract_brief_part(response_text: str) -> str:
    """Extract the BRIEF part from the model response."""
    text = response_text.strip()
    if "BRIEF:" in text:
        parts = text.split("DETAILED:", 1)
        brief_section = parts[0].replace("BRIEF:", "").strip()
        return brief_section
    return text.split("\n")[0].strip() if text else "Scene unclear"


def extract_detailed_part(response_text: str) -> str:
    """Extract the DETAILED part from the model response."""
    text = response_text.strip()
    if "DETAILED:" in text:
        parts = text.split("DETAILED:", 1)
        if len(parts) > 1:
            return parts[1].strip()
    lines = text.split("\n")
    if len(lines) > 1:
        return " ".join(lines[1:]).strip()
    return text


async def on_take_photo(runtime, image, input_data):
    """Return a brief object list for a single snapshot."""
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
    full_text = extract_text(response).strip()
    brief = extract_brief_part(full_text)
    return brief or "Scene unclear"


async def on_stream_start(runtime, input_data):
    """Initialize state for progressive streaming session."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("progression_level", 0)
    runtime.set_state("analyzing", False)


async def on_frame(runtime, frame):
    """Emit progressive scene descriptions: brief first, then detailed if scene stable."""
    if runtime.is_cancelled():
        return

    if runtime.get_state("analyzing"):
        return

    runtime.set_state("analyzing", True)

    try:
        await _analyze_progressive(runtime, frame)
    finally:
        runtime.set_state("analyzing", False)


async def _analyze_progressive(runtime, frame):
    """Helper that performs all awaited work for scene analysis."""
    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is not None and is_same_scene(anchor, embedding):
        progression = int(runtime.get_state("progression_level", 0))
        if progression >= 2:
            return
    else:
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("progression_level", 0)
        runtime.set_state("scene_first_analyzed_at", None)

    progression = int(runtime.get_state("progression_level", 0))

    if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
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

    full_text = extract_text(response).strip()

    if progression == 0:
        brief = extract_brief_part(full_text)
        result = brief or "Scene unclear"

        await runtime.emit(
            result,
            final=True,
            metadata={
                "scene_generation": generation,
                "progression_level": 1,
                "frame_id": frame.frame_id,
            },
        )
        runtime.set_state("progression_level", 1)
        runtime.set_state("cached_full_response", full_text)

        asyncio.create_task(_schedule_detailed(runtime, generation, full_text, frame))

    elif progression == 1:
        cached = runtime.get_state("cached_full_response", "")
        text_to_use = cached if cached else full_text
        detailed = extract_detailed_part(text_to_use)
        result = detailed or extract_brief_part(text_to_use) or "Scene unclear"

        await runtime.emit(
            result,
            final=True,
            metadata={
                "scene_generation": generation,
                "progression_level": 2,
                "frame_id": frame.frame_id,
            },
        )
        runtime.set_state("progression_level", 2)


async def _schedule_detailed(runtime, generation, cached_response, frame):
    """Wait for the scene to remain stable, then emit detailed description."""
    await asyncio.sleep(DETAIL_DELAY_SECONDS)

    if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
        return

    if int(runtime.get_state("progression_level", 0)) < 1:
        return

    detailed = extract_detailed_part(cached_response)
    result = detailed or extract_brief_part(cached_response) or "Scene unclear"

    await runtime.emit(
        result,
        final=True,
        metadata={
            "scene_generation": generation,
            "progression_level": 2,
            "frame_id": frame.frame_id,
        },
    )
    runtime.set_state("progression_level", 2)


async def on_stream_stop(runtime):
    """Invalidate current scene generation so pending work becomes stale."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing", False)


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "extract_brief_part",
    "extract_detailed_part",
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
