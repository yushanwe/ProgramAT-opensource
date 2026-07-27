"""Read playing cards held in a user's hand, with temporal swipe detection in streaming."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "card_hand_reader"
TOOL_PROMPT = (
    "Identify all playing cards currently held in the hand shown. "
    "List each readable card by rank and suit, for example: ace of spades, "
    "ten of hearts. "
    "If multiple chronological frames are provided, also note any card that "
    "was added to or removed from the hand since the earliest frame, "
    "for example: 'Queen of clubs was played.' "
    "Keep the response concise and spoken-ready for a blind or low-vision user. "
    "If cards are not visible or readable, say so."
)

_DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite-preview"
_RECENT_FRAME_SECONDS = 4
_MIN_CHANGE_FRAMES = 2


async def on_take_photo(runtime, image, input_data):
    """Read cards in a single camera image."""
    del runtime, input_data
    if image is None:
        return "No camera image available."
    response = await asyncio.to_thread(
        call_model,
        _DEFAULT_MODEL,
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 30, "num_retries": 0},
    )
    return extract_text(response) or "Could not read any cards."


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    del input_data
    runtime.set_state("last_result", None)
    runtime.set_state("analyzing_frame_id", None)


async def on_frame(runtime, frame):
    """Analyze recent frames to detect the current hand and any card swipes."""
    if runtime.is_cancelled():
        return

    # Avoid overlapping analyses
    if runtime.get_state("analyzing_frame_id") is not None:
        return
    runtime.set_state("analyzing_frame_id", frame.frame_id)

    try:
        recent = runtime.get_recent_frames(seconds=_RECENT_FRAME_SECONDS)
        if not recent:
            recent = [frame]

        images = [f.image for f in recent if f.image is not None]
        if not images:
            return

        response = await asyncio.to_thread(
            call_model,
            _DEFAULT_MODEL,
            [{"role": "user", "content": TOOL_PROMPT}],
            images,
            {"timeout": 30, "num_retries": 0},
        )
        text = extract_text(response) or "Could not read any cards."

        if runtime.is_cancelled():
            return

        last = runtime.get_state("last_result")
        if text != last:
            runtime.set_state("last_result", text)
            await runtime.emit(text, final=True, metadata={"frame_id": frame.frame_id})
    finally:
        if runtime.get_state("analyzing_frame_id") == frame.frame_id:
            runtime.set_state("analyzing_frame_id", None)


async def on_stream_stop(runtime):
    """Reset streaming state."""
    runtime.set_state("analyzing_frame_id", None)


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
