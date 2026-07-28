"""Identify Uber Car Details - Make, Model, Color, and License Plate.

Helps blind users identify their Uber by analyzing the car in the middle of the scene
and reporting its make, model, color, and license plate number in a concise,
speech-ready format.
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "identify_uber_car_details"
TOOL_PROMPT = (
    "Find the car in the middle of the scene and identify its make, model, color, "
    "and license plate number. Report the information concisely in this format: "
    "'[Color] [Make] [Model] with license plate [Number]'. For example: 'White Toyota "
    "Camry with license plate ABC123'. If any detail is unclear or unavailable, state "
    "what you can determine and note uncertainty for the rest."
)


def _compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized appearance embedding for scene change detection."""
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


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def _is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.985,
) -> bool:
    """Check if two scene embeddings represent the same scene."""
    return _cosine_similarity(reference, current) >= threshold


async def on_take_photo(runtime, image, input_data):
    """Analyze a single photo to identify the Uber car details.

    Uses Gemini 3.1 Flash Lite for accurate car identification including
    make, model, color, and license plate OCR.
    """
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
    """Initialize streaming state for car identification."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("last_result", None)


async def on_frame(runtime, frame):
    """Process each distinct scene change and identify car details.

    Only analyzes when the scene changes significantly to avoid redundant
    processing of the same car view. Uses latest frame for car identification.
    """
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(_compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")

    # Check if this is a new scene
    if anchor is not None and _is_same_scene(anchor, embedding):
        # Same scene, skip processing
        return

    # New scene detected, update anchor
    runtime.set_state("scene_anchor", embedding)

    # Analyze the car in the new scene
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [frame.image],
        {"timeout": 60, "num_retries": 0},
    )
    result_text = extract_text(response)

    # Check if cancelled during processing
    if runtime.is_cancelled():
        return

    # Store and emit the result
    runtime.set_state("last_result", result_text)
    await runtime.emit(
        result_text,
        final=True,
        metadata={"frame_id": frame.frame_id},
    )


async def on_stream_stop(runtime):
    """Clear state when streaming stops."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
