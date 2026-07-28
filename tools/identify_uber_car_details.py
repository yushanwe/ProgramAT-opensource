"""Identify Uber Car Details and Provide Navigation to Passenger Door.

Helps blind users identify their Uber by analyzing the car in the middle of the scene
and reporting its make, model, color, and license plate number. After the car remains
stable in view for several frames, automatically switches to navigation mode and
provides spatial guidance to the passenger-side door using clock-face directions.
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

NAVIGATION_PROMPT = (
    "A blind user has confirmed this is their Uber. Locate the passenger-side door "
    "(right side if facing the front of the vehicle). Report its position using "
    "clock-face directions (9-12-3 only, relative to the camera), approximate distance, "
    "and brief guidance on how to reach it. For example: 'Passenger door is at 2 o'clock, "
    "about 3 meters away. Move slightly right and forward.' If the door is not clearly "
    "visible, guide toward the visible side of the vehicle."
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
    runtime.set_state("mode", "identification")
    runtime.set_state("stable_frames", 0)
    runtime.set_state("car_details", None)


async def on_frame(runtime, frame):
    """Process each frame for car identification and navigation guidance.

    Operates in two modes:
    1. Identification: Detects car details on scene changes
    2. Navigation: After car is stable for 3 frames, provides door location guidance
    """
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(_compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    mode = runtime.get_state("mode", "identification")
    stable_frames = runtime.get_state("stable_frames", 0)

    # Check if this is the same scene
    is_same = anchor is not None and _is_same_scene(anchor, embedding)

    if is_same:
        # Same scene - increment stability counter
        stable_frames += 1
        runtime.set_state("stable_frames", stable_frames)

        # After 3 stable frames in identification mode, switch to navigation
        if mode == "identification" and stable_frames >= 3:
            runtime.set_state("mode", "navigation")
            mode = "navigation"
            # Continue to navigation processing below
        elif mode == "identification":
            # Still in identification mode, skip processing
            return
    else:
        # New scene detected - reset to identification mode
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("stable_frames", 0)
        runtime.set_state("mode", "identification")
        mode = "identification"
        stable_frames = 0

    # Process based on current mode
    if mode == "identification" or (mode == "navigation" and stable_frames == 3):
        # Run identification (either new scene or first time entering navigation)
        if mode == "identification":
            prompt = TOOL_PROMPT
        else:
            # Transitioning to navigation - run identification one more time
            prompt = TOOL_PROMPT

        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": prompt}],
            [frame.image],
            {"timeout": 60, "num_retries": 0},
        )
        result_text = extract_text(response)

        if runtime.is_cancelled():
            return

        runtime.set_state("last_result", result_text)
        runtime.set_state("car_details", result_text)

        await runtime.emit(
            result_text,
            final=True,
            metadata={"frame_id": frame.frame_id, "mode": mode},
        )

    elif mode == "navigation" and stable_frames > 3:
        # Navigation mode - provide guidance to passenger door
        # Only update every 2-3 frames to avoid too frequent updates
        if stable_frames % 3 != 0:
            return

        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": NAVIGATION_PROMPT}],
            [frame.image],
            {"timeout": 60, "num_retries": 0},
        )
        result_text = extract_text(response)

        if runtime.is_cancelled():
            return

        runtime.set_state("last_result", result_text)

        await runtime.emit(
            result_text,
            final=True,
            metadata={"frame_id": frame.frame_id, "mode": "navigation"},
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
