"""Card Hand Identifier — name pointed or spread playing cards for blind users."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "card_hand_identifier"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "You are assisting a blind user who is holding playing cards. "
    "Examine the image carefully and do one of the following:\n"
    "1. If a finger is clearly pointing at or touching a single card, name only "
    "that card using natural spoken English, for example: 'Nine of hearts.' or "
    "'Jack of spades.'\n"
    "2. If multiple cards are spread or fanned out with no single card being "
    "pointed to, list every readable card from left to right, for example: "
    "'Two of clubs, seven of diamonds, queen of hearts, ace of spades.'\n"
    "Use only: ace, two, three, four, five, six, seven, eight, nine, ten, jack, "
    "queen, or king followed by 'of clubs', 'of diamonds', 'of hearts', or "
    "'of spades'. Do not mention jokers. "
    "If no playing cards are visible or the image is too unclear, say: "
    "'No cards visible.'"
)


async def on_take_photo(runtime, image, input_data):
    """Identify the pointed-at card or list all spread cards for a blind user."""
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


__all__ = ["on_take_photo"]
