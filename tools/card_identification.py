"""Identify playing cards by pointing or swiping through a hand."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "card_identification"
TOOL_PROMPT = (
    "Identify visible playing cards clearly. When one image is available, report "
    "the card the user is pointing to concisely, such as 'Nine of hearts.' When "
    "multiple chronological frames are available showing a swipe through cards, "
    "report all distinct cards in the hand, such as 'Nine of hearts, two of spades, "
    "jack of diamonds.' If no cards are visible or evidence is insufficient, say so."
)


async def on_take_photo(runtime, image, input_data):
    """Identify a single card from one image."""
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
    """Initialize state for tracking cards across frames."""
    del input_data
    runtime.set_state("seen_cards", set())
    runtime.set_state("last_result", None)
    runtime.set_state("frames_processed", 0)


async def on_frame(runtime, frame):
    """Process each frame and accumulate cards from swipe gesture."""
    if runtime.is_cancelled():
        return

    frames_processed = int(runtime.get_state("frames_processed", 0))

    # Get recent frames for temporal understanding
    recent_frames = runtime.get_recent_frames(count=5)

    if not recent_frames:
        return

    # Use multiple frames to understand swipe motion
    images = [f.image for f in recent_frames]

    # Call model with recent frames to detect cards
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        images,
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response)

    # Update frames processed count
    runtime.set_state("frames_processed", frames_processed + 1)

    # Parse detected cards and update seen_cards state
    seen_cards = runtime.get_state("seen_cards", set())

    # Extract card names from the response
    # The model should return something like "Nine of hearts, two of spades"
    if text and text.lower() not in ["no cards visible", "insufficient evidence"]:
        # Split by common delimiters
        cards = [card.strip() for card in text.replace(" and ", ", ").split(",")]
        for card in cards:
            if card:
                seen_cards.add(card)
        runtime.set_state("seen_cards", seen_cards)

    # Emit partial result showing accumulated cards
    if seen_cards:
        accumulated_result = ", ".join(sorted(seen_cards))
        last_result = runtime.get_state("last_result")
        if accumulated_result != last_result:
            runtime.set_state("last_result", accumulated_result)
            await runtime.emit(
                accumulated_result,
                partial=True,
                metadata={"frame_id": frame.frame_id, "card_count": len(seen_cards)},
            )


async def on_stream_stop(runtime):
    """Emit final summary of all cards detected during the swipe."""
    seen_cards = runtime.get_state("seen_cards", set())

    if seen_cards:
        final_result = ", ".join(sorted(seen_cards))
        await runtime.emit(
            final_result,
            final=True,
            metadata={"total_cards": len(seen_cards)},
        )
    else:
        await runtime.emit(
            "No cards detected",
            final=True,
            metadata={"total_cards": 0},
        )


__all__ = [
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
