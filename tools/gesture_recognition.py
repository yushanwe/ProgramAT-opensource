"""Recognize hand gestures from motion sequences over recent history."""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "gesture_recognition"

TOOL_PROMPT = (
    "Observe hand movements to identify gestures or short meanings. "
    "When multiple chronological frames are available, track the hand motion "
    "sequence from earliest to latest to determine what gesture was performed. "
    "Some gestures are recognizable from a single stable pose (thumbs up, peace sign), "
    "while others require motion evidence (waving, pointing direction changes, "
    "finger wagging side to side). "
    "When only one image is available, identify any clearly visible static hand posture "
    "but do not infer motion that cannot be observed. "
    "Return one concise spoken sentence naming the gesture, such as 'Waving hello' "
    "or 'Thumbs up' or 'Finger wagging no'. "
    "If the gesture cannot be recognized or hands are not clearly visible, "
    "return exactly 'Cannot recognize gesture'."
)

SCENE_SIMILARITY_THRESHOLD = 0.92
MOTION_THRESHOLD = 0.08
COLLECTION_DURATION = 6.0
MINIMUM_FRAMES = 8
COOLDOWN_DURATION = 4.0


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


def has_motion(reference: np.ndarray, current: np.ndarray) -> bool:
    """Detect motion by checking if similarity falls below threshold."""
    similarity = cosine_similarity(reference, current)
    return similarity < (1.0 - MOTION_THRESHOLD)


async def on_take_photo(runtime, image, input_data):
    """Analyze a single frame for static gestures only."""
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
    """Initialize event-driven state for gesture collection and analysis."""
    del input_data
    runtime.set_state("state", "idle")
    runtime.set_state("collection_start_time", None)
    runtime.set_state("motion_anchor", None)
    runtime.set_state("event_generation", 0)
    runtime.set_state("cooldown_until", None)
    runtime.set_state("in_flight_task", None)


async def on_frame(runtime, frame):
    """Event-driven gesture recognition with collection → analyze → cooldown lifecycle."""
    if runtime.is_cancelled():
        return

    state = runtime.get_state("state", "idle")
    current_time = frame.timestamp
    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)

    # Check cooldown
    cooldown_until = runtime.get_state("cooldown_until")
    if cooldown_until and current_time < cooldown_until:
        return

    # If cooldown expired, reset to idle
    if state == "waiting_for_reset" and cooldown_until and current_time >= cooldown_until:
        runtime.set_state("state", "idle")
        runtime.set_state("motion_anchor", None)
        runtime.set_state("collection_start_time", None)
        runtime.set_state("cooldown_until", None)
        state = "idle"

    if state == "idle":
        # Look for motion to start collecting
        motion_anchor = runtime.get_state("motion_anchor")
        if motion_anchor is None:
            runtime.set_state("motion_anchor", embedding)
            return

        if has_motion(motion_anchor, embedding):
            # Motion detected - start collecting
            runtime.set_state("state", "collecting")
            runtime.set_state("collection_start_time", current_time)
            runtime.set_state("event_generation", runtime.get_state("event_generation", 0) + 1)
        else:
            # Update anchor to current stable scene
            runtime.set_state("motion_anchor", embedding)
        return

    if state == "collecting":
        collection_start = runtime.get_state("collection_start_time")
        if collection_start is None:
            runtime.set_state("state", "idle")
            return

        duration = current_time - collection_start

        # Check if we've collected enough
        recent_frames = runtime.get_recent_frames(seconds=COLLECTION_DURATION)

        if duration >= COLLECTION_DURATION and len(recent_frames) >= MINIMUM_FRAMES:
            # Ready to analyze
            runtime.set_state("state", "analyzing")
            event_generation = int(runtime.get_state("event_generation", 0))

            # Check if already analyzing this event
            in_flight = runtime.get_state("in_flight_task")
            if in_flight and not in_flight.done():
                return

            # Select chronological frames for analysis
            selected_frames = recent_frames[-MINIMUM_FRAMES:]
            selected_images = [f.image for f in selected_frames]

            async def analyze_gesture():
                try:
                    response = await asyncio.to_thread(
                        call_model,
                        "gemini/gemini-3.1-flash-lite-preview",
                        [{"role": "user", "content": TOOL_PROMPT}],
                        selected_images,
                        {"timeout": 60, "num_retries": 0},
                    )
                    text = extract_text(response)

                    if (
                        runtime.is_cancelled()
                        or runtime.get_state("event_generation") != event_generation
                    ):
                        return

                    await runtime.emit(text, final=True, replace=True)

                    # Enter cooldown
                    runtime.set_state("state", "waiting_for_reset")
                    runtime.set_state("cooldown_until", current_time + COOLDOWN_DURATION)

                except Exception as exc:
                    if not runtime.is_cancelled():
                        await runtime.emit(
                            "Cannot recognize gesture",
                            final=True,
                            replace=True,
                            metadata={"error": str(exc)},
                        )
                        runtime.set_state("state", "waiting_for_reset")
                        runtime.set_state("cooldown_until", current_time + COOLDOWN_DURATION)
                finally:
                    if runtime.get_state("in_flight_task") is task:
                        runtime.set_state("in_flight_task", None)

            task = asyncio.create_task(analyze_gesture())
            runtime.set_state("in_flight_task", task)
        return

    if state == "analyzing":
        # Wait for analysis to complete
        return

    if state == "waiting_for_reset":
        # Wait for cooldown
        return


async def on_stream_stop(runtime):
    """Cancel in-flight work and reset state."""
    in_flight = runtime.get_state("in_flight_task")
    if in_flight and not in_flight.done():
        in_flight.cancel()
        try:
            await in_flight
        except asyncio.CancelledError:
            pass

    runtime.set_state("in_flight_task", None)
    runtime.set_state("state", "idle")
    runtime.set_state("event_generation", runtime.get_state("event_generation", 0) + 1)


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
