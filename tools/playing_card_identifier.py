"""Playing card identifier for blind and low-vision users.

Identifies specific playing cards pointed to by a finger, or lists all cards
in a hand when swiped through. Uses recent frames to detect temporal patterns.
"""

import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "playing_card_identifier"
TOOL_PROMPT = """
You are helping a blind or low-vision user identify playing cards.

When analyzing the image or images:
1. Look for a finger pointing at a specific card. If you see pointing, identify only that single card.
2. If you see a finger moving across multiple cards (a swiping motion visible across frames), list all the cards the finger passes over in order.
3. Each card should be identified by rank and suit, for example "Nine of hearts", "Two of spades", "Jack of diamonds".
4. Use full words: ace, two, three, four, five, six, seven, eight, nine, ten, jack, queen, or king.
5. Use full suit names: hearts, diamonds, clubs, or spades.

For a single pointed card, respond with just the card name: "Nine of hearts."
For multiple cards during a swipe, list them separated by commas: "Nine of hearts, two of spades, jack of diamonds."

If the cards are unclear due to poor lighting, reflections, obstructions, or if no cards are visible, respond with: "I cannot recognize the cards clearly from this view."

Return one concise user-facing response suitable for spoken output. Prefer one or two short sentences in a single paragraph with no unnecessary line breaks. State only the information needed for the user's task.
"""

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot recognize the cards clearly from this view."
USE_RECENT_FRAMES = True


async def analyze_frames(frames):
    """
    Analyze one or more frames to identify playing cards.

    Args:
        frames: Single image or list of recent frames in chronological order

    Returns:
        Spoken-friendly card identification text
    """
    if isinstance(frames, list):
        images = [f.image for f in frames if f.image is not None]
    else:
        images = [frames] if frames is not None else []

    if not images:
        return FALLBACK_TEXT

    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": TOOL_PROMPT}],
        images,
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Handle single-frame Take Photo mode for identifying a pointed card."""
    if image is None:
        return FALLBACK_TEXT
    return await analyze_frames(image)


async def on_stream_start(runtime, input_data):
    """Initialize streaming state."""
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    """
    Handle streaming frames to detect pointing or swiping gestures.
    Uses recent frames to distinguish between single-card pointing and multi-card swiping.
    """
    if runtime.is_cancelled():
        return

    if runtime.get_state("in_flight"):
        return

    runtime.set_state("in_flight", True)

    try:
        # Get recent frames to detect temporal patterns (pointing vs swiping)
        recent_frames = runtime.get_recent_frames(seconds=2)

        if not recent_frames:
            # Fall back to current frame if no history available
            text = await analyze_frames(frame)
        else:
            # Use multiple frames to help model detect swiping motion
            text = await analyze_frames(recent_frames)

        if not runtime.is_cancelled() and text:
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up streaming state."""
    runtime.clear_state()
