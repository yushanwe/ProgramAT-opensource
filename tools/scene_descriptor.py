"""Progressive scene descriptor that names objects briefly, then describes in detail if scene is steady."""

from __future__ import annotations

import asyncio
import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "scene_descriptor"

SCENE_SIMILARITY_THRESHOLD = 0.985
DETAIL_DELAY_SECONDS = 2.0  # Wait this long before detailed description


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
    """Provide a brief object listing for a single photo."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."

    # For Take Photo, we only have one image, so provide brief listing
    brief_prompt = (
        "List the main objects in this scene briefly for a blind user, "
        "like 'A desk, a cabinet, a printer, and a cat'. Keep it concise."
    )

    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": brief_prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("scene_start_time", None)
    runtime.set_state("brief_emitted", False)
    runtime.set_state("detail_emitted", False)
    runtime.set_state("in_flight_brief", None)
    runtime.set_state("in_flight_detail", None)


async def on_frame(runtime, frame):
    """Progressive scene analysis: brief listing first, then detailed if scene is steady."""
    if runtime.is_cancelled():
        return

    # Compute scene embedding
    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    # Check if scene changed
    if anchor is not None and is_same_scene(anchor, embedding):
        # Same scene - check if we need to emit detailed description
        scene_start_time = runtime.get_state("scene_start_time")
        brief_emitted = runtime.get_state("brief_emitted", False)
        detail_emitted = runtime.get_state("detail_emitted", False)

        if not brief_emitted or not scene_start_time:
            return  # Wait for brief to complete

        # Check if enough time has passed for detailed description
        time_elapsed = frame.timestamp - scene_start_time
        if time_elapsed >= DETAIL_DELAY_SECONDS and not detail_emitted:
            # Check if detail analysis already in flight
            detail_task = runtime.get_state("in_flight_detail")
            if detail_task is not None and not detail_task.done():
                return

            # Launch detailed description
            async def analyze_detail():
                try:
                    detail_prompt = (
                        "Describe this scene in detail for a blind user, including spatial "
                        "relationships and notable details (e.g., 'A cat sitting on top of a "
                        "white desk, looking out the window'). Be concise and natural for speech."
                    )
                    response = await asyncio.to_thread(
                        call_model,
                        "gemini/gemini-3.1-flash-lite-preview",
                        [{"role": "user", "content": detail_prompt}],
                        [frame.image],
                        {"timeout": 60, "num_retries": 0},
                    )
                    text = extract_text(response)

                    if runtime.is_cancelled():
                        return
                    if runtime.get_state("scene_generation") != generation:
                        return

                    await runtime.emit(text, final=True, replace=False)
                    runtime.set_state("detail_emitted", True)
                finally:
                    if runtime.get_state("in_flight_detail") is detail_task:
                        runtime.set_state("in_flight_detail", None)

            detail_task = asyncio.create_task(analyze_detail())
            runtime.set_state("in_flight_detail", detail_task)
    else:
        # New scene detected
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("scene_start_time", frame.timestamp)
        runtime.set_state("brief_emitted", False)
        runtime.set_state("detail_emitted", False)

        # Cancel any in-flight work from previous scene
        old_brief = runtime.get_state("in_flight_brief")
        old_detail = runtime.get_state("in_flight_detail")
        if old_brief and not old_brief.done():
            old_brief.cancel()
        if old_detail and not old_detail.done():
            old_detail.cancel()
        runtime.set_state("in_flight_brief", None)
        runtime.set_state("in_flight_detail", None)

        # Launch brief object listing
        async def analyze_brief():
            try:
                brief_prompt = (
                    "List the main objects in this scene briefly for a blind user, "
                    "like 'A desk, a cabinet, a printer, and a cat'. Keep it concise."
                )
                response = await asyncio.to_thread(
                    call_model,
                    "gemini/gemini-3.1-flash-lite-preview",
                    [{"role": "user", "content": brief_prompt}],
                    [frame.image],
                    {"timeout": 60, "num_retries": 0},
                )
                text = extract_text(response)

                if runtime.is_cancelled():
                    return
                if runtime.get_state("scene_generation") != generation:
                    return

                await runtime.emit(text, final=False, replace=True)
                runtime.set_state("brief_emitted", True)
            finally:
                if runtime.get_state("in_flight_brief") is brief_task:
                    runtime.set_state("in_flight_brief", None)

        brief_task = asyncio.create_task(analyze_brief())
        runtime.set_state("in_flight_brief", brief_task)


async def on_stream_stop(runtime):
    """Clean up when streaming stops."""
    # Cancel any in-flight work
    brief_task = runtime.get_state("in_flight_brief")
    detail_task = runtime.get_state("in_flight_detail")

    if brief_task and not brief_task.done():
        brief_task.cancel()
    if detail_task and not detail_task.done():
        detail_task.cancel()

    # Invalidate current scene
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
