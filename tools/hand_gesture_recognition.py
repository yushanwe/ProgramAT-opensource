"""Hand Gesture Recognition Tool

Recognizes hand gestures from camera input, supporting both static poses and
motion-based gestures. Designed for blind and low-vision users to understand
hand movements and their meanings.

Features:
- Static pose recognition from single frames
- Motion-based gesture recognition from recent frame history
- Fingerspelling word recognition from letter sequences
- Concise audio-friendly output for text-to-speech
- Handles uncertainty when evidence is insufficient
"""

import asyncio
from typing import Any, Dict, Optional

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_recognition"

TOOL_PROMPT = (
    "Observe the hand movements in the image(s) and identify the gesture or meaning expressed. "
    "When multiple chronological frames are provided, track hand position, shape, and movement "
    "over time to recognize motion-based gestures like waving (hello/goodbye), beckoning (come), "
    "or finger wagging (no/stop). When only one image is provided, identify any clear static "
    "hand pose such as thumbs up, peace sign, or pointing, but do not infer motion that is not "
    "visible. Report the gesture name and its common meaning in one concise sentence for a "
    "blind or low-vision user. If no clear gesture is visible or evidence is insufficient, "
    "state that uncertainty plainly."
)

FINGERSPELLING_PROMPT = (
    "Analyze the hand configuration in the image to identify if it represents a fingerspelled "
    "letter from the American Sign Language (ASL) alphabet (A through Z). If you recognize a "
    "clear letter, respond with ONLY the single uppercase letter (e.g., 'L' or 'O'). If the "
    "hand configuration is unclear, ambiguous, not a recognizable ASL letter, or the hand is "
    "in a transition state between letters, respond with 'NONE'. Be precise and only report "
    "letters you are confident about."
)

# Frame selection for streaming mode
HISTORY_WINDOW_SECONDS = 4.0  # Analyze recent 4 seconds for motion
MIN_FRAMES_FOR_MOTION = 4  # Minimum frames needed for motion analysis

# Fingerspelling configuration
LETTER_HOLD_TIME = 1.0  # Seconds to hold a letter before confirmation
WORD_BOUNDARY_PAUSE = 2.5  # Seconds of no new letters to complete a word
MIN_WORD_LENGTH = 2  # Minimum letters to form a word


async def _try_fingerspelling(runtime, frame):
    """Attempt to detect fingerspelling and track letter sequences.

    Returns True if fingerspelling mode is active, False otherwise.
    """
    current_timestamp = frame.timestamp

    # Check for letter in current frame
    try:
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": FINGERSPELLING_PROMPT}],
            [frame.image],
            {"timeout": 30, "num_retries": 0},
        )

        detected = extract_text(response).strip().upper()

        # Valid single letter response
        if len(detected) == 1 and detected.isalpha():
            await _handle_detected_letter(runtime, detected, current_timestamp)
            return True
        elif detected == "NONE":
            # No letter detected, check for word completion
            await _check_word_completion(runtime, current_timestamp)
            return runtime.get_state("fingerspelling_mode")
        else:
            # Unexpected response, treat as no letter
            await _check_word_completion(runtime, current_timestamp)
            return runtime.get_state("fingerspelling_mode")

    except Exception as e:
        print(f"Fingerspelling detection error: {e}")
        return False


async def _handle_detected_letter(runtime, letter, timestamp):
    """Handle a detected letter: track it and add to buffer when confirmed."""
    current_letter = runtime.get_state("current_letter")
    letter_start_time = runtime.get_state("letter_start_time")
    letter_buffer = runtime.get_state("letter_buffer")

    if current_letter == letter:
        # Same letter, check if held long enough
        if letter_start_time and (timestamp - letter_start_time) >= LETTER_HOLD_TIME:
            # Letter confirmed, add to buffer if not already added
            if not letter_buffer or letter_buffer[-1] != letter:
                letter_buffer.append(letter)
                runtime.set_state("letter_buffer", letter_buffer)
                runtime.set_state("last_letter_time", timestamp)
                runtime.set_state("fingerspelling_mode", True)

                # Emit progress
                word_progress = "-".join(letter_buffer)
                await runtime.emit(
                    f"Spelling: {word_progress}",
                    final=False,
                    metadata={
                        "frame_id": runtime.get_state("current_frame_id"),
                        "timestamp": timestamp,
                        "letter_count": len(letter_buffer),
                    },
                )
    else:
        # New letter detected, start tracking
        runtime.set_state("current_letter", letter)
        runtime.set_state("letter_start_time", timestamp)


async def _check_word_completion(runtime, timestamp):
    """Check if enough time has passed to complete the current word."""
    last_letter_time = runtime.get_state("last_letter_time")
    letter_buffer = runtime.get_state("letter_buffer")

    if last_letter_time and letter_buffer:
        time_since_last = timestamp - last_letter_time

        if time_since_last >= WORD_BOUNDARY_PAUSE:
            # Word complete
            if len(letter_buffer) >= MIN_WORD_LENGTH:
                word = "".join(letter_buffer).lower()
                await runtime.emit(
                    f"Word spelled: {word}",
                    final=False,
                    metadata={
                        "timestamp": timestamp,
                        "word": word,
                        "letter_count": len(letter_buffer),
                    },
                )

            # Reset fingerspelling state
            runtime.set_state("fingerspelling_mode", False)
            runtime.set_state("current_letter", None)
            runtime.set_state("letter_start_time", None)
            runtime.set_state("letter_buffer", [])
            runtime.set_state("last_letter_time", None)


async def on_take_photo(runtime, image, input_data):
    """Analyze a single frame for static hand gestures.

    Take Photo mode can only detect clear static poses like thumbs up,
    peace sign, or pointing. Motion-based gestures cannot be determined
    from a single frame.
    """
    del runtime, input_data

    if image is None:
        return "No camera image available for gesture recognition."

    # Use Gemini Flash Lite for quick analysis
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 30, "num_retries": 0},
    )

    return extract_text(response)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state for gesture recognition."""
    del input_data
    runtime.set_state("last_result", None)
    runtime.set_state("last_timestamp", None)

    # Fingerspelling state
    runtime.set_state("fingerspelling_mode", False)
    runtime.set_state("current_letter", None)
    runtime.set_state("letter_start_time", None)
    runtime.set_state("letter_buffer", [])
    runtime.set_state("last_letter_time", None)


async def on_frame(runtime, frame):
    """Process frames and analyze recent history for motion-based gestures."""
    if runtime.is_cancelled():
        return

    current_timestamp = frame.timestamp
    runtime.set_state("current_frame_id", frame.frame_id)

    # Try fingerspelling detection first (lightweight single-frame check)
    fingerspelling_result = await _try_fingerspelling(runtime, frame)

    if fingerspelling_result:
        # Fingerspelling mode active, skip general gesture recognition
        return

    # General gesture recognition (original behavior)
    # Get recent frames for motion analysis
    recent_frames = runtime.get_recent_frames(seconds=HISTORY_WINDOW_SECONDS)

    if not recent_frames:
        return

    # Decide whether to analyze based on frame count and time since last result
    last_timestamp = runtime.get_state("last_timestamp")

    # Only analyze if we have enough frames and it's been at least 2 seconds
    # since last result to avoid repetition
    if last_timestamp is not None and (current_timestamp - last_timestamp) < 2.0:
        return

    # Select frames for analysis - prefer motion analysis when we have enough frames
    if len(recent_frames) >= MIN_FRAMES_FOR_MOTION:
        # Use chronological frames for motion detection
        # Select evenly spaced frames across the window
        step = max(1, len(recent_frames) // MIN_FRAMES_FOR_MOTION)
        selected_frames = recent_frames[::step][:MIN_FRAMES_FOR_MOTION]
        images = [f.image for f in selected_frames]
    else:
        # Use just the latest frame for static pose detection
        images = [frame.image]

    # Call model with selected frames
    try:
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            images,
            {"timeout": 30, "num_retries": 0},
        )

        result_text = extract_text(response)

        # Check cancellation before emitting
        if runtime.is_cancelled():
            return

        # Only emit if result is different or meaningful
        last_result = runtime.get_state("last_result")

        # Normalize results for comparison (lowercase, strip)
        normalized_current = result_text.lower().strip()
        normalized_last = last_result.lower().strip() if last_result else ""

        # Emit if result changed or if it's a clear gesture (not "no gesture")
        if (normalized_current != normalized_last and
            "no gesture" not in normalized_current and
            "no clear gesture" not in normalized_current):

            await runtime.emit(
                result_text,
                final=False,
                metadata={
                    "frame_count": len(images),
                    "frame_id": frame.frame_id,
                    "timestamp": current_timestamp,
                },
            )

            runtime.set_state("last_result", result_text)
            runtime.set_state("last_timestamp", current_timestamp)

    except Exception as e:
        # Log error but don't crash
        print(f"Hand gesture recognition error: {e}")


async def on_stream_stop(runtime):
    """Clean up streaming state when stopped."""
    runtime.clear_state()


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
