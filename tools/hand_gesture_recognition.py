"""Hand gesture recognition tool analyzing motion over time."""

from __future__ import annotations

import asyncio
import cv2
import numpy as np
from typing import Optional

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_recognition"
TOOL_PROMPT = (
    "When multiple chronological frames are available, observe the hand movements "
    "from earliest to latest and identify the gesture or short meaning just expressed. "
    "Use the full motion sequence, not only the final hand pose. For example, waving "
    "may mean 'hello' or 'goodbye,' while moving one finger side to side may mean 'no' "
    "or 'incorrect.' Some gestures may be recognizable from a single stable pose, while "
    "others are defined by how the hands move across several frames. When only one image "
    "is available, describe any clearly recognizable static hand pose but do not infer "
    "motion that is not visible. Report the identified gesture concisely for a blind or "
    "low-vision user, for example 'Waving hello' or 'Thumbs up.' If the gesture is "
    "unclear or hands are not visible, say so plainly."
)

# Temporal analysis configuration
COLLECTION_WINDOW_SECONDS = 3.0  # Collect frames for up to 3 seconds
MIN_FRAMES_FOR_GESTURE = 6  # Minimum frames to analyze motion
COOLDOWN_FRAMES = 10  # Frames to wait after analysis before next gesture
MOTION_THRESHOLD = 0.02  # Threshold for detecting significant hand motion


def compute_motion_score(frames: list) -> float:
    """Compute a simple motion score by comparing consecutive frames."""
    if len(frames) < 2:
        return 0.0

    motion_scores = []
    for i in range(len(frames) - 1):
        try:
            # Convert to grayscale for faster processing
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            # Resize for faster computation
            gray1 = cv2.resize(gray1, (128, 128))
            gray2 = cv2.resize(gray2, (128, 128))

            # Compute absolute difference
            diff = cv2.absdiff(gray1, gray2)
            motion = np.mean(diff) / 255.0  # Normalize to 0-1
            motion_scores.append(motion)
        except Exception:
            continue

    return np.mean(motion_scores) if motion_scores else 0.0


async def on_take_photo(runtime, image, input_data):
    """Analyze a single image for static hand poses."""
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
    """Initialize streaming state for gesture collection."""
    del input_data
    runtime.set_state("state", "idle")  # idle, collecting, analyzing, cooldown
    runtime.set_state("collected_frames", [])
    runtime.set_state("collection_start_time", None)
    runtime.set_state("cooldown_counter", 0)
    runtime.set_state("in_flight_task", None)
    runtime.set_state("gesture_generation", 0)


async def on_frame(runtime, frame):
    """Process frames for gesture recognition with event-driven temporal analysis."""
    if runtime.is_cancelled():
        return

    current_state = runtime.get_state("state", "idle")

    # Cooldown state: wait before accepting new gestures
    if current_state == "cooldown":
        cooldown_counter = runtime.get_state("cooldown_counter", 0)
        if cooldown_counter >= COOLDOWN_FRAMES:
            runtime.set_state("state", "idle")
            runtime.set_state("cooldown_counter", 0)
            runtime.set_state("collected_frames", [])
        else:
            runtime.set_state("cooldown_counter", cooldown_counter + 1)
            return

    # Analyzing state: skip new frames while analysis is in flight
    if current_state == "analyzing":
        in_flight = runtime.get_state("in_flight_task")
        if in_flight is not None and not in_flight.done():
            return
        # Analysis completed, move to cooldown
        runtime.set_state("state", "cooldown")
        runtime.set_state("cooldown_counter", 0)
        return

    # Idle or collecting state: gather frames
    collected = list(runtime.get_state("collected_frames", []))

    # Detect motion to start collection
    if current_state == "idle":
        if len(collected) == 0:
            # Start collecting with first frame
            collected.append(frame.image.copy())
            runtime.set_state("collected_frames", collected)
            runtime.set_state("collection_start_time", frame.timestamp)
            runtime.set_state("state", "collecting")
            return

    # Collecting state: continue gathering frames
    if current_state == "collecting":
        collected.append(frame.image.copy())
        runtime.set_state("collected_frames", collected)

        collection_start = runtime.get_state("collection_start_time")
        elapsed = frame.timestamp - collection_start if collection_start else 0

        # Check if we have enough frames and time has elapsed
        if len(collected) >= MIN_FRAMES_FOR_GESTURE and elapsed >= COLLECTION_WINDOW_SECONDS:
            # Check for significant motion
            motion_score = compute_motion_score(collected)

            # If minimal motion, reset and wait for actual gesture
            if motion_score < MOTION_THRESHOLD:
                runtime.set_state("state", "idle")
                runtime.set_state("collected_frames", [])
                runtime.set_state("collection_start_time", None)
                return

            # Ready to analyze - create temporal window
            generation = int(runtime.get_state("gesture_generation", 0)) + 1
            runtime.set_state("gesture_generation", generation)
            runtime.set_state("state", "analyzing")

            # Select representative frames (first, middle points, last)
            indices = [
                0,
                len(collected) // 3,
                len(collected) * 2 // 3,
                len(collected) - 1,
            ]
            selected_frames = [collected[i] for i in indices if i < len(collected)]

            # Launch analysis task
            async def analyze_gesture():
                try:
                    # Call model with chronological frames
                    response = await asyncio.to_thread(
                        call_model,
                        "gemini/gemini-3.1-flash-lite-preview",
                        [{"role": "user", "content": TOOL_PROMPT}],
                        selected_frames,
                        {"timeout": 60, "num_retries": 0},
                    )

                    if runtime.is_cancelled():
                        return

                    if runtime.get_state("gesture_generation") != generation:
                        return

                    text = extract_text(response)
                    if text:
                        await runtime.emit(text, final=True, replace=True)

                finally:
                    # Clear in-flight marker
                    if runtime.get_state("in_flight_task") is task:
                        runtime.set_state("in_flight_task", None)

            task = asyncio.create_task(analyze_gesture())
            runtime.set_state("in_flight_task", task)


async def on_stream_stop(runtime):
    """Clean up state when streaming stops."""
    generation = int(runtime.get_state("gesture_generation", 0)) + 1
    runtime.set_state("gesture_generation", generation)
    in_flight = runtime.get_state("in_flight_task")
    if in_flight is not None and not in_flight.done():
        in_flight.cancel()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
