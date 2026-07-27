"""Recognize whether a visible vehicle is likely the user's Uber ride."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "recognize_uber_car"
TOOL_PROMPT = (
    "For a blind or low-vision user, decide whether the visible vehicle is likely "
    "their Uber ride based only on visible evidence such as Uber signs/decals, "
    "readable license plate details, and pickup context. Respond with one short "
    "sentence that starts with exactly one of: 'Likely Uber:', 'Unlikely Uber:', "
    "or 'Not enough evidence:'. If no vehicle is visible, say so clearly."
)

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
# Mean grayscale pixel delta (0-255) required to treat the latest frame as a new scene.
SCENE_DIFF_THRESHOLD = 2.0
# Keep responses quick enough for live guidance while allowing occasional slower calls.
MODEL_TIMEOUT_SECONDS = 45


def _scene_fingerprint(image: np.ndarray) -> np.ndarray:
    """Create a compact grayscale fingerprint for scene-change gating."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def _scene_difference(previous: np.ndarray, current: np.ndarray) -> float:
    if previous is None or current is None or previous.shape != current.shape:
        return float("inf")
    return float(np.mean(np.abs(previous - current)))


def _normalize_response(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned:
        return cleaned
    return "Not enough evidence: I can't confirm whether this is your Uber from this view."


def _run_model(image: np.ndarray) -> str:
    response = call_model(
        MODEL_NAME,
        [{"role": "user", "content": TOOL_PROMPT}],
        images=[image],
        metadata={"timeout": MODEL_TIMEOUT_SECONDS, "num_retries": 0},
    )
    return _normalize_response(extract_text(response))


async def on_take_photo(runtime, image, input_data):
    del runtime, input_data
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "Not enough evidence: No camera image is available."
    return await asyncio.to_thread(_run_model, image)


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("scene_fingerprint", None)
    runtime.set_state("last_result", None)
    runtime.set_state("processing", False)
    runtime.set_state("generation", 0)


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return
    image = frame.image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return

    current_fingerprint = await asyncio.to_thread(_scene_fingerprint, image)
    previous_fingerprint = runtime.get_state("scene_fingerprint")
    if _scene_difference(previous_fingerprint, current_fingerprint) < SCENE_DIFF_THRESHOLD:
        return
    if runtime.get_state("processing"):
        return

    generation = int(runtime.get_state("generation", 0)) + 1
    runtime.set_state("generation", generation)
    runtime.set_state("processing", True)

    try:
        result = await asyncio.to_thread(_run_model, image)
        if runtime.is_cancelled() or runtime.get_state("generation") != generation:
            return
        if result == runtime.get_state("last_result"):
            runtime.set_state("scene_fingerprint", current_fingerprint)
            return
        runtime.set_state("scene_fingerprint", current_fingerprint)
        runtime.set_state("last_result", result)
        await runtime.emit(
            result,
            final=True,
            replace=True,
            metadata={"model": MODEL_NAME, "frame_id": frame.frame_id},
        )
    finally:
        if runtime.get_state("generation") == generation:
            runtime.set_state("processing", False)


async def on_stream_stop(runtime):
    runtime.set_state("generation", int(runtime.get_state("generation", 0)) + 1)
    runtime.set_state("processing", False)
