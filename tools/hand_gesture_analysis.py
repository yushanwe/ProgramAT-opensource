"""
Hand Gesture Analysis Tool

Identifies hand gestures and their meanings by analyzing hand movements over time.
Supports both static gestures (recognizable from a single pose) and dynamic gestures
(defined by motion across multiple frames).

Examples:
- Waving: hello or goodbye
- Finger moving side to side: no or incorrect
- Thumbs up: approval
- Peace sign: static pose
- Beckoning: come here

This tool uses recent frame history to analyze motion-based gestures while also
handling single-frame static gestures.
"""

import asyncio
from typing import Any, Dict, Optional

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_analysis"

TOOL_PROMPT = (
    "Observe the person's hand movements and identify the gesture or short meaning expressed. "
    "When multiple chronological frames are available, analyze the full motion sequence from "
    "earliest to latest, tracking hand position, orientation, and movement patterns. "
    "When only one image is available, identify any clearly recognizable static hand pose. "
    "Report the identified gesture and its likely meaning in one concise sentence for a blind user. "
    "If the gesture is unclear or hands are not visible, state what cannot be determined."
)


async def on_take_photo(runtime, image, input_data):
    """
    Analyze a single frame for static hand gestures.

    Motion-based gestures cannot be identified from a single image,
    so this reports only static poses when recognizable.
    """
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
    )

    result = extract_text(response)

    # Ensure we return a user-friendly message
    if not result or result.strip() == "":
        return "No clear hand gesture detected"

    return result


async def on_stream_start(runtime, input_data):
    """Initialize streaming session for gesture analysis."""
    runtime.set_state("last_emitted_gesture", None)
    runtime.set_state("frames_analyzed", 0)


async def on_frame(runtime, frame):
    """
    Process frames and analyze hand gesture motion over recent history.

    Uses a temporal window of recent frames to detect motion-based gestures
    while also identifying static poses.
    """
    if runtime.is_cancelled():
        return

    frames_analyzed = runtime.get_state("frames_analyzed", 0)
    runtime.set_state("frames_analyzed", frames_analyzed + 1)

    # Analyze every 10 frames to balance responsiveness with efficiency
    if frames_analyzed % 10 != 0:
        return

    # Get recent frames for motion analysis (last 3 seconds)
    recent_frames = runtime.get_recent_frames(seconds=3)

    if not recent_frames:
        return

    # For gesture analysis, we want to see motion, so use multiple frames
    # when available, but can also work with just the latest frame
    if len(recent_frames) >= 3:
        # Use multiple frames in chronological order for motion analysis
        # Select a sparse sample to show motion without overwhelming the model
        if len(recent_frames) > 6:
            # Sample evenly across the time window
            step = len(recent_frames) // 6
            selected_frames = [recent_frames[i] for i in range(0, len(recent_frames), step)][:6]
        else:
            selected_frames = recent_frames

        images = [f.image for f in selected_frames]
    else:
        # Single frame or very few frames - analyze static pose
        images = [runtime.current_frame.image]

    # Call model with the selected frames
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        images,
    )

    result = extract_text(response)

    if not result or result.strip() == "":
        return

    # Avoid repeating the same gesture detection
    last_emitted = runtime.get_state("last_emitted_gesture")

    # Simple deduplication: check if the core gesture is the same
    # Extract key words to compare (e.g., "waving", "thumbs up")
    current_normalized = result.lower().strip()
    last_normalized = last_emitted.lower().strip() if last_emitted else ""

    # Emit if substantially different or first detection
    if not last_emitted or current_normalized != last_normalized:
        runtime.set_state("last_emitted_gesture", result)
        await runtime.emit(result, final=False)


async def on_stream_stop(runtime):
    """Clean up when streaming session ends."""
    runtime.clear_state()
