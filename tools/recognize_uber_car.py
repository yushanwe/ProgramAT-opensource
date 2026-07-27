"""Recognize whether a visible vehicle is likely the user's Uber ride."""

from __future__ import annotations

import asyncio
import logging

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "recognize_uber_car"
TOOL_PROMPT = (
    "For a blind or low-vision user, analyze visible vehicles using only the current "
    "frame. 1) Determine whether the target vehicle is visible and, when relevant, "
    "whether it likely matches the requested Uber. 2) If the request context indicates the "
    "target car is already confirmed and the target appears visible, locate the passenger-side "
    "door and describe its direction using only 9-12 or 1-3 o'clock plus approximate "
    "distance and one concise action to "
    "reach it. 3) If the target is not yet confirmed, respond with one short sentence that "
    "starts with exactly one of: 'Likely Uber:', 'Unlikely Uber:', or "
    "'Not enough evidence:'. If the target is confirmed, start with 'Door guidance:' "
    "or 'Not enough evidence:' when the door cannot be located confidently."
)

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
# Mean grayscale pixel delta (0-255) that filters tiny jitter while still reacting
# when a user points the camera at a different car.
SCENE_DIFF_THRESHOLD = 2.0
# Keep responses quick enough for live guidance while allowing occasional slower calls.
MODEL_TIMEOUT_SECONDS = 45

LOGGER = logging.getLogger(__name__)
CONFIRMED_VEHICLE_KEY = "confirmed_vehicle"
VEHICLE_DETAIL_FIELDS = (
    ("make", "vehicle_make"),
    ("model", "vehicle_model"),
    ("color", "vehicle_color"),
    ("plate", "vehicle_plate"),
)


def _scene_fingerprint(image: np.ndarray) -> np.ndarray:
    """Create a compact grayscale fingerprint for scene-change gating."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def _scene_difference(
    previous: np.ndarray | None, current: np.ndarray | None
) -> float | None:
    if previous is None or current is None or previous.shape != current.shape:
        return None
    return float(np.mean(np.abs(previous - current)))


def _normalize_response(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned:
        return cleaned
    return "Not enough evidence: I can't confirm whether this is your Uber from this view."


def _ensure_dict(input_data: object) -> dict:
    return input_data if isinstance(input_data, dict) else {}


def _build_request_context(config: dict) -> str:
    confirmed_vehicle = bool(config.get(CONFIRMED_VEHICLE_KEY, False))
    details = []
    for label, key in VEHICLE_DETAIL_FIELDS:
        value = str(config.get(key, "")).strip()
        if value:
            details.append(f"{label}: {value}")
    details_text = "; ".join(details) if details else "none provided"
    return (
        f"Vehicle confirmation: {'yes' if confirmed_vehicle else 'no'}. "
        f"Target vehicle details: {details_text}."
    )


def _run_model(image: np.ndarray, request_context: str) -> str:
    try:
        # call_model accepts a list of images; this task uses exactly one frame.
        response = call_model(
            MODEL_NAME,
            [{"role": "user", "content": f"{TOOL_PROMPT}\n{request_context}"}],
            images=[image],
            metadata={"timeout": MODEL_TIMEOUT_SECONDS, "num_retries": 0},
        )
    except Exception as exc:
        LOGGER.warning("recognize_uber_car model call failed: %s", exc)
        return "Not enough evidence: Unable to analyze the image at this time."
    return _normalize_response(extract_text(response))


async def on_take_photo(runtime, image, input_data):
    del runtime
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "Not enough evidence: No camera image is available."
    config = _ensure_dict(input_data)
    return await asyncio.to_thread(_run_model, image, _build_request_context(config))


async def on_stream_start(runtime, input_data):
    config = _ensure_dict(input_data)
    runtime.set_state("scene_fingerprint", None)
    runtime.set_state("last_result", None)
    runtime.set_state("processing", False)
    runtime.set_state("generation", 0)
    runtime.set_state("request_context", _build_request_context(config))


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return
    image = frame.image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return

    current_fingerprint = _scene_fingerprint(image)
    previous_fingerprint = runtime.get_state("scene_fingerprint")
    scene_difference = _scene_difference(previous_fingerprint, current_fingerprint)
    if scene_difference is not None and scene_difference < SCENE_DIFF_THRESHOLD:
        return
    if runtime.get_state("processing"):
        return

    # The runtime dispatches frames serially per stream session; this update is kept
    # together to gate stale responses.
    generation = runtime.get_state("generation", 0) + 1
    runtime.set_state("generation", generation)
    runtime.set_state("processing", True)

    try:
        request_context = runtime.get_state(
            "request_context", _build_request_context({})
        )
        result = await asyncio.to_thread(_run_model, image, request_context)
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
    runtime.set_state("generation", runtime.get_state("generation", 0) + 1)
    runtime.set_state("processing", False)
