"""Identify hand gestures from static poses or motion sequences."""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_recognition"
TOOL_PROMPT = (
    "Identify the hand gesture or meaning of the recent hand movement. "
    "When multiple chronological frames are available, observe the hand motion "
    "sequence from earliest to latest to recognize gestures like waving (hello/goodbye) "
    "or finger movements (no/incorrect). When only one image is available, identify "
    "any recognizable static hand pose. If hands are overlapping or the gesture "
    "cannot be clearly recognized, state this plainly. Give a concise answer suitable "
    "for speech output, for example 'Waving hello' or 'Thumbs up' or 'Hands are "
    "overlapping, cannot recognize gesture.'"
)

FRAME_COLLECTION_SECONDS = 3.0
MIN_FRAMES_FOR_GESTURE = 5
GESTURE_COOLDOWN_SECONDS = 2.0


def compute_motion_score(prev_image: np.ndarray, curr_image: np.ndarray) -> float:
    """Compute a simple motion metric between two frames."""
    if prev_image is None or curr_image is None:
        return 0.0
    try:
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        prev_resized = cv2.resize(prev_gray, (160, 120))
        curr_resized = cv2.resize(curr_gray, (160, 120))
        diff = cv2.absdiff(prev_resized, curr_resized)
        return float(np.mean(diff))
    except Exception:
        return 0.0


async def on_take_photo(runtime, image, input_data):
    """Analyze a single image for static hand gestures."""
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
    runtime.set_state("state", "idle")
    runtime.set_state("collection_start_time", None)
    runtime.set_state("collected_frames", [])
    runtime.set_state("last_frame", None)
    runtime.set_state("gesture_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("last_result_time", None)


async def on_frame(runtime, frame):
    """
    Event-driven gesture recognition with state machine:
    idle -> collecting -> analyzing -> cooldown -> idle
    """
    if runtime.is_cancelled():
        return

    current_time = time.time()
    state = runtime.get_state("state", "idle")
    last_frame = runtime.get_state("last_frame")

    # Compute motion to detect when hands are moving
    motion_score = 0.0
    if last_frame is not None:
        motion_score = await asyncio.to_thread(
            compute_motion_score, last_frame, frame.image
        )
    runtime.set_state("last_frame", frame.image)

    # State: idle - waiting for motion to start collection
    if state == "idle":
        MOTION_THRESHOLD = 15.0
        if motion_score > MOTION_THRESHOLD:
            runtime.set_state("state", "collecting")
            runtime.set_state("collection_start_time", current_time)
            runtime.set_state("collected_frames", [frame])
        return

    # State: collecting - gathering frames for gesture analysis
    if state == "collecting":
        collected = runtime.get_state("collected_frames", [])
        collected.append(frame)
        runtime.set_state("collected_frames", collected)

        start_time = runtime.get_state("collection_start_time")
        elapsed = current_time - start_time

        # Check if we've collected enough frames
        if elapsed >= FRAME_COLLECTION_SECONDS and len(collected) >= MIN_FRAMES_FOR_GESTURE:
            # Move to analyzing state
            runtime.set_state("state", "analyzing")
            generation = int(runtime.get_state("gesture_generation", 0)) + 1
            runtime.set_state("gesture_generation", generation)

            # Check if already analyzing this generation
            if runtime.get_state("analyzing_generation") == generation:
                return

            runtime.set_state("analyzing_generation", generation)

            # Select representative frames from the collection
            num_frames = len(collected)
            indices = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)
            selected_frames = [collected[i].image for i in indices]

            async def analyze_gesture():
                try:
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
                        await runtime.emit(text, final=True)
                        runtime.set_state("last_result_time", current_time)

                    # Move to cooldown state
                    runtime.set_state("state", "cooldown")
                    runtime.set_state("cooldown_start_time", current_time)
                    runtime.set_state("collected_frames", [])

                finally:
                    if runtime.get_state("analyzing_generation") == generation:
                        runtime.set_state("analyzing_generation", None)

            asyncio.create_task(analyze_gesture())
        return

    # State: cooldown - prevent immediate re-triggering
    if state == "cooldown":
        cooldown_start = runtime.get_state("cooldown_start_time", current_time)
        if current_time - cooldown_start >= GESTURE_COOLDOWN_SECONDS:
            runtime.set_state("state", "idle")
        return


async def on_stream_stop(runtime):
    """Clean up state when streaming stops."""
    runtime.set_state("state", "idle")
    runtime.set_state("collected_frames", [])
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
