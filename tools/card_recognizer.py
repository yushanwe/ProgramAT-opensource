"""Card recognizer: identifies playing cards in hand for blind and low-vision users."""

from __future__ import annotations

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "card_recognizer"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "You are helping a blind or low-vision user identify playing cards held in their hand. "
    "Examine the image and identify every visible playing card. "
    "If a finger is clearly pointing at one specific card, name that card first. "
    "List all visible cards using full spoken names such as 'nine of hearts', "
    "'two of spades', or 'jack of diamonds'. "
    "Use only: ace, two, three, four, five, six, seven, eight, nine, ten, jack, queen, or king "
    "followed by 'of clubs', 'of diamonds', 'of hearts', or 'of spades'. "
    "If multiple cards are visible, list them separated by commas. "
    "If no playing cards are visible or the cards are unreadable, say so plainly."
)


async def on_take_photo(runtime, image, input_data):
    """Identify playing cards in the image using Gemini Flash Lite."""
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60},
    )
    return extract_text(response)
