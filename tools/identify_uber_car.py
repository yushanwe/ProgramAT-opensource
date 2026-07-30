"""Identify Uber Car Details for finding a ride.

Helps blind and low-vision users identify their Uber or rideshare vehicle by
reporting make, model, color, and license plate number of cars in the scene.
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "identify_uber_car"
TOOL_PROMPT = (
    "Identify the car at the center of the scene for a blind user waiting for their Uber. "
    "Report the make, model, color, and license plate number in a concise, speech-ready format. "
    "Example: 'White Toyota Camry, license plate ABC123.' "
    "If multiple cars are visible, focus on the most prominent or centered one. "
    "If no car is visible or details cannot be determined, state what is uncertain."
)

NAVIGATION_PROMPT = (
    "Guide a blind user to the passenger-side door of the vehicle at the center of the scene. "
    "1. Identify where the passenger-side door is located relative to the camera view. "
    "2. Provide the direction using clock-face position (9-12 or 1-3 only, restricted to what's visible in front of the camera). "
    "3. Estimate approximate distance if possible (e.g., 'about 10 feet'). "
    "4. Give concise walking instructions to reach the door handle. "
    "Example: 'Passenger door is at your 2 o'clock, about 12 feet away. Walk forward and slightly right.' "
    "If the passenger door is not visible or identifiable, state that clearly."
)

SCENE_SIMILARITY_THRESHOLD = 0.985


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
    """Analyze one current image to identify car details."""
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
    """Initialize state for streaming session."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("last_result", None)
    runtime.set_state("navigation_mode", False)
    runtime.set_state("vehicle_confirmed", False)
    runtime.set_state("consecutive_same_vehicle_count", 0)


async def on_frame(runtime, frame):
    """Process changed latest frame to identify car details or provide navigation."""
    if runtime.is_cancelled():
        return

    # Check if scene has changed
    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")

    if anchor is not None and is_same_scene(anchor, embedding):
        # Same scene detected
        # If we're in navigation mode and vehicle is confirmed, continue providing navigation
        if runtime.get_state("navigation_mode") and runtime.get_state("vehicle_confirmed"):
            # Process navigation for the stable scene
            navigation_response = await asyncio.to_thread(
                call_model,
                "gemini/gemini-3.1-flash-lite-preview",
                [{"role": "user", "content": NAVIGATION_PROMPT}],
                [frame.image],
                {"timeout": 60, "num_retries": 0},
            )

            if runtime.is_cancelled():
                return

            navigation_result = extract_text(navigation_response)
            await runtime.emit(navigation_result, final=True)
        return

    # New scene detected, update anchor
    runtime.set_state("scene_anchor", embedding)

    # Analyze the frame for vehicle identification
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [frame.image],
        {"timeout": 60, "num_retries": 0},
    )

    if runtime.is_cancelled():
        return

    result = extract_text(response)
    last_result = runtime.get_state("last_result")

    # Track consecutive same vehicle sightings to infer confirmation
    if result == last_result and last_result is not None:
        count = runtime.get_state("consecutive_same_vehicle_count", 0) + 1
        runtime.set_state("consecutive_same_vehicle_count", count)

        # After seeing the same vehicle details 3 times, assume confirmed and enter navigation mode
        if count >= 3 and not runtime.get_state("vehicle_confirmed"):
            runtime.set_state("vehicle_confirmed", True)
            runtime.set_state("navigation_mode", True)
            await runtime.emit(
                f"{result} Vehicle confirmed. Switching to navigation mode.",
                final=True
            )
            return
    else:
        # Different result, reset counter
        runtime.set_state("consecutive_same_vehicle_count", 0)
        runtime.set_state("vehicle_confirmed", False)
        runtime.set_state("navigation_mode", False)

    # Only emit if result is different from last time
    if result != last_result:
        runtime.set_state("last_result", result)
        await runtime.emit(result, final=True)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
