"""Identify a playing card held in hand, with swipe-action temporal awareness."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "identify_card_in_hand"

TOOL_PROMPT = (
    "Identify the playing card or cards held in the person's hand. "
    "When multiple chronological recent frames are available, inspect them from "
    "earliest to latest to understand whether a swipe gesture just revealed a "
    "new card. Report the card or cards now visible after any swipe. "
    "State the rank and suit of each visible card clearly, for example "
    "\"You are holding the jack of spades\" or \"You just swiped to the ace of hearts.\" "
    "If no card is clearly readable from the available frames, say so plainly. "
    "Keep the response concise and natural when spoken aloud."
)

_ANALYSIS_COOLDOWN_SECONDS = 2.0
_HISTORY_WINDOW_SECONDS = 3.0


async def on_take_photo(runtime, image, input_data):
    """Identify a card in a single captured photo."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 30, "num_retries": 0},
    )
    return extract_text(response) or "No card could be identified."


async def on_stream_start(runtime, input_data):
    """Initialize streaming state for card identification."""
    del input_data
    runtime.set_state("last_analyzed_timestamp", None)
    runtime.set_state("analysis_in_progress", False)


async def on_frame(runtime, frame):
    """Analyze recent frame history to detect cards and understand swipe gestures."""
    if runtime.is_cancelled():
        return

    if runtime.get_state("analysis_in_progress"):
        return

    last_ts = runtime.get_state("last_analyzed_timestamp")
    if last_ts is not None and (frame.timestamp - last_ts) < _ANALYSIS_COOLDOWN_SECONDS:
        return

    runtime.set_state("analysis_in_progress", True)
    current_timestamp = frame.timestamp
    try:
        recent_frames = runtime.get_recent_frames(seconds=_HISTORY_WINDOW_SECONDS)
        if not recent_frames:
            return

        images = [f.image for f in recent_frames if f.image is not None]
        if not images:
            return

        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            images,
            {"timeout": 30, "num_retries": 0},
        )
        text = extract_text(response) or "No card could be identified."

        if runtime.is_cancelled():
            return

        runtime.set_state("last_analyzed_timestamp", current_timestamp)
        await runtime.emit(text, final=True)
    finally:
        runtime.set_state("analysis_in_progress", False)


async def on_stream_stop(runtime):
    """Clean up all streaming state."""
    runtime.clear_state()
