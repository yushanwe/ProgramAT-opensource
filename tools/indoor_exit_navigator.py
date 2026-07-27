"""Indoor Exit Navigator for blind and low-vision users.

Identifies visible exits, emergency exits, doors leading outside, and directional
exit signs in the current frame, then gives concise spoken navigation guidance
using clock-face directions and approximate distances.
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "indoor_exit_navigator"

TOOL_PROMPT = (
    "You are assisting a blind or low-vision user who needs to find an exit inside a building. "
    "Examine the image and identify any exits, emergency exits, doors leading outside, stairwells, "
    "or exit signs. Provide concise, spoken-word navigation guidance: "
    "use clock-face directions only between 9 o'clock and 3 o'clock (9, 10, 11, 12, 1, 2, 3), "
    "body-relative terms (straight ahead, left, right), and approximate distance in steps or feet. "
    "If an exit or exit sign is visible, say where it is and how to reach it. "
    "If no exit is visible, say so plainly and suggest the user slowly turn to scan the room. "
    "Never use color as the only landmark. Keep the response under 30 words and suitable for text-to-speech."
)

SCENE_SIMILARITY_THRESHOLD = 0.982
_HIST_HUE_BINS = 32
_HIST_SAT_BINS = 16
_HIST_SIZE = _HIST_HUE_BINS * _HIST_SAT_BINS


def _scene_embedding(image: np.ndarray) -> np.ndarray:
    """Compact HSV histogram embedding for scene-change detection."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(_HIST_SIZE, dtype=np.float32)
    small = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [_HIST_HUE_BINS, _HIST_SAT_BINS], [0, 180, 0, 256]
    ).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm else hist


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return -1.0
    return float(np.dot(a, b))


def _same_scene(ref: np.ndarray, cur: np.ndarray) -> bool:
    return _cosine_sim(ref, cur) >= SCENE_SIMILARITY_THRESHOLD


def _analyze_frame(image: np.ndarray) -> str:
    """Call the vision model and return navigation guidance text."""
    response = call_model(
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


# ---------------------------------------------------------------------------
# Take Photo entry point
# ---------------------------------------------------------------------------

async def on_take_photo(runtime, image, input_data):
    """Analyze one image and return exit navigation guidance."""
    del runtime, input_data
    if image is None:
        return "No camera image available."
    result = await asyncio.to_thread(_analyze_frame, image)
    return result if result else "No exit detected. Try slowly turning to scan the room."


# ---------------------------------------------------------------------------
# Streaming entry points
# ---------------------------------------------------------------------------

async def on_stream_start(runtime, input_data):
    """Initialize per-session state."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("completed_generation", None)


async def on_frame(runtime, frame):
    """Emit guidance when the scene changes; skip near-duplicate frames."""
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is not None and _same_scene(anchor, embedding):
        # Scene unchanged – skip if this generation is already handled.
        if (
            runtime.get_state("analyzing_generation") == generation
            or runtime.get_state("completed_generation") == generation
        ):
            return
    else:
        # New scene detected.
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("completed_generation", None)

    if runtime.get_state("analyzing_generation") == generation:
        return
    runtime.set_state("analyzing_generation", generation)

    try:
        text = await asyncio.to_thread(_analyze_frame, frame.image)

        if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
            return

        output = text if text else "No exit visible. Try slowly turning to scan the room."
        await runtime.emit(
            output,
            final=True,
            metadata={"scene_generation": generation, "frame_id": frame.frame_id},
        )
        runtime.set_state("completed_generation", generation)
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    """Invalidate any in-flight analysis when the session ends."""
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
