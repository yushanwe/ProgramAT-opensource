"""Progressive scene description that provides brief then detailed descriptions."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "scene_descriptor"

TOOL_PROMPT = (
    "Describe the scene for a blind user. Identify objects, people, and spatial "
    "relationships. Be concise and speech-ready. If uncertain, state what cannot "
    "be determined."
)

BRIEF_PROMPT = (
    "List the main objects in this scene very briefly, like 'A desk, a cabinet, "
    "a printer, and a cat.' Keep it to one short sentence naming the key items."
)

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
    runtime.set_state("detailed_done", False)
    runtime.set_state("in_flight_brief", None)
    runtime.set_state("in_flight_detailed", None)


async def on_frame(runtime, frame):
    """Emit brief description on new scene, detailed on same scene continuation."""
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
        runtime.set_state("detailed_done", False)

        # Cancel any in-flight detailed analysis from previous scene
        detailed_task = runtime.get_state("in_flight_detailed")
        if detailed_task and not detailed_task.done():
            detailed_task.cancel()
        runtime.set_state("in_flight_detailed", None)

    # Decide what to emit
    brief_done = runtime.get_state("brief_done", False)
    detailed_done = runtime.get_state("detailed_done", False)

    if not brief_done:
        # Emit brief description for new or unprocessed scene
        brief_task = runtime.get_state("in_flight_brief")
        if brief_task and not brief_task.done():
            return  # Already analyzing

        async def run_brief():
            try:
                response = await asyncio.to_thread(
                    call_model,
                    "gemini/gemini-3.1-flash-lite-preview",
                    [{"role": "user", "content": BRIEF_PROMPT}],
                    [frame.image],
                    {"timeout": 60, "num_retries": 0},
                )
                text = extract_text(response)
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

    elif not detailed_done:
        # Scene is stable - emit detailed description
        detailed_task = runtime.get_state("in_flight_detailed")
        if detailed_task and not detailed_task.done():
            return  # Already analyzing

        async def run_detailed():
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
                await runtime.emit(text, final=True, replace=True)
                runtime.set_state("detailed_done", True)
            finally:
                if runtime.get_state("in_flight_detailed") is task:
                    runtime.set_state("in_flight_detailed", None)

        task = asyncio.create_task(run_detailed())
        runtime.set_state("in_flight_detailed", task)


async def on_stream_stop(runtime):
    """Clean up state when streaming stops."""
    generation = int(runtime.get_state("scene_generation", 0))
    runtime.set_state("scene_generation", generation + 1)

    # Cancel any in-flight tasks
    brief_task = runtime.get_state("in_flight_brief")
    if brief_task and not brief_task.done():
        brief_task.cancel()
    detailed_task = runtime.get_state("in_flight_detailed")
    if detailed_task and not detailed_task.done():
        detailed_task.cancel()

    runtime.set_state("in_flight_brief", None)
    runtime.set_state("in_flight_detailed", None)


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
