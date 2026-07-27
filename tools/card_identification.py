"""Identify playing cards by pointing or swiping through a hand."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "card_identification"
TOOL_PROMPT = (
    "Identify visible playing cards clearly. If one card is prominently visible "
    "or stands out from others (such as being pulled forward), report only that "
    "card concisely, such as 'Nine of hearts.' If multiple cards are being shown "
    "in a swipe motion across frames, report all distinct cards seen, such as "
    "'Nine of hearts, two of spades, jack of diamonds.' If no cards are visible "
    "or evidence is insufficient, say so."
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
    runtime.set_state("last_single_card", None)
    runtime.set_state("frames_processed", 0)


async def on_frame(runtime, frame):
    """Process each frame for single-card or swipe-gesture detection."""
    if runtime.is_cancelled():
        return

    frames_processed = int(runtime.get_state("frames_processed", 0))

    # Get recent frames for temporal understanding
    recent_frames = runtime.get_recent_frames(count=5)

    if not recent_frames:
        return

    # Use multiple frames to understand context
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

    if not text or text.lower() in ["no cards visible", "insufficient evidence"]:
        return

    # Parse the response - check if it's a single card or multiple cards
    # Split by common delimiters
    cards = [card.strip() for card in text.replace(" and ", ", ").split(",")]
    cards = [card for card in cards if card]

    if not cards:
        return

    # If only one card detected, it's likely a "pulled out" card - report immediately
    if len(cards) == 1:
        single_card = cards[0]
        last_single = runtime.get_state("last_single_card")

        # Only emit if it's a different card than last time
        if single_card != last_single:
            runtime.set_state("last_single_card", single_card)
            await runtime.emit(
                single_card,
                partial=True,
                metadata={"frame_id": frame.frame_id, "mode": "single_card"},
            )
    else:
        # Multiple cards detected - accumulate for swipe mode
        runtime.set_state("last_single_card", None)  # Clear single card state
        seen_cards = runtime.get_state("seen_cards", set())

        for card in cards:
            seen_cards.add(card)
        runtime.set_state("seen_cards", seen_cards)

        # Emit partial result showing accumulated cards
        accumulated_result = ", ".join(sorted(seen_cards))
        last_result = runtime.get_state("last_result")
        if accumulated_result != last_result:
            runtime.set_state("last_result", accumulated_result)
            await runtime.emit(
                accumulated_result,
                partial=True,
                metadata={"frame_id": frame.frame_id, "card_count": len(seen_cards), "mode": "swipe"},
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
