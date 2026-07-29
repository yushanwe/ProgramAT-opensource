"""Identify the Uber car in the middle of the scene."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import cv2
import numpy as np

from litellm_utils import call_model, call_openai_responses_model, extract_text

TOOL_NAME = "find_uber_car_info"
TOOL_PROMPT = (
    "Help a blind user find their Uber car. 1. If the target car is not yet "
    "confirmed, identify the car in the middle of the scene and report its make, "
    "model, color, and license plate in one concise spoken sentence. 2. If the "
    "user has confirmed the target car, tell where the passenger-side door is "
    "using left, right, or visible clock direction only, and give a short path "
    "to reach it. If any detail is unclear or not visible, say what is uncertain "
    "instead of guessing."
)

FAST_MODEL = "gemini/gemini-3.1-flash-lite-preview"
STRONG_MODEL = "gpt-5"
SCENE_SIMILARITY_THRESHOLD = 0.985


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized embedding for scene change detection."""
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


def _needs_escalation(text: str) -> bool:
    if not text:
        return True
    normalized = text.lower()
    uncertain_markers = (
        "unclear",
        "uncertain",
        "not visible",
        "not readable",
        "cannot",
        "can't",
        "unable",
        "unknown",
        "sorry",
    )
    return any(marker in normalized for marker in uncertain_markers)


def _normalize_input_data(input_data: Any) -> Dict[str, Any]:
    if isinstance(input_data, dict):
        return input_data
    return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "confirmed"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _get_confirmed_target(input_data: Any) -> Optional[str]:
    config = _normalize_input_data(input_data)
    confirmed = any(
        _as_bool(config.get(key))
        for key in (
            "confirmed",
            "is_confirmed",
            "confirm_target",
            "target_confirmed",
            "uber_confirmed",
            "vehicle_confirmed",
        )
    )
    if not confirmed:
        return None

    for key in ("target_vehicle", "vehicle_description", "uber_description", "target"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    parts = []
    for key in ("color", "make", "model"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    plate = config.get("plate") or config.get("license_plate")
    if isinstance(plate, str) and plate.strip():
        parts.append(f"with license plate {plate.strip()}")
    if parts:
        return " ".join(parts)
    return "the confirmed Uber car"


def _build_prompt(target_vehicle: Optional[str]) -> str:
    if not target_vehicle:
        return TOOL_PROMPT
    return (
        f"{TOOL_PROMPT} The user has confirmed that the target vehicle is "
        f"{target_vehicle}. Only guide the user to that confirmed car's "
        "passenger-side door if it is visible in this view."
    )


def _call_fast_model(image: np.ndarray, prompt: str) -> str:
    response = call_model(
        FAST_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    return extract_text(response).strip()


def _call_strong_model(image: np.ndarray, prompt: str) -> str:
    return call_openai_responses_model(
        STRONG_MODEL,
        prompt,
        image,
        reasoning_effort="medium",
        timeout=60,
    ).strip()


def _fallback_message(target_vehicle: Optional[str]) -> str:
    if target_vehicle:
        return (
            "I can't clearly locate the confirmed Uber car's passenger-side door "
            "from this view."
        )
    return "I can't determine the car details from this view."


def _analyze_image(
    image: np.ndarray, prefer_fast_only: bool, target_vehicle: Optional[str]
) -> str:
    prompt = _build_prompt(target_vehicle)
    text = _call_fast_model(image, prompt)
    if prefer_fast_only or not _needs_escalation(text):
        return text or _fallback_message(target_vehicle)

    strong_text = _call_strong_model(image, prompt)
    return strong_text or text or _fallback_message(target_vehicle)


async def on_take_photo(runtime, image, input_data):
    del runtime
    if image is None:
        return "No camera image is available."
    target_vehicle = _get_confirmed_target(input_data)
    return await asyncio.to_thread(_analyze_image, image, False, target_vehicle)


async def on_stream_start(runtime, input_data):
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("completed_generation", None)
    runtime.set_state("last_emitted_text", None)
    runtime.set_state("target_vehicle", _get_confirmed_target(input_data))


async def on_frame(runtime, frame):
    if runtime.is_cancelled() or frame is None or frame.image is None:
        return

    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    if anchor is not None and cosine_similarity(anchor, embedding) >= SCENE_SIMILARITY_THRESHOLD:
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
        target_vehicle = runtime.get_state("target_vehicle")
        prefer_fast_only = target_vehicle is None
        text = await asyncio.to_thread(
            _analyze_image, frame.image, prefer_fast_only, target_vehicle
        )
        if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
            return
        if text == runtime.get_state("last_emitted_text"):
            runtime.set_state("completed_generation", generation)
            return
        runtime.set_state("completed_generation", generation)
        runtime.set_state("last_emitted_text", text)
        await runtime.emit(
            text or _fallback_message(target_vehicle),
            final=True,
            metadata={"frame_id": frame.frame_id, "scene_generation": generation},
        )
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "TOOL_NAME",
    "TOOL_PROMPT",
    "compute_scene_embedding",
    "cosine_similarity",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
