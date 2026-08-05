import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "card_identifier"
TOOL_PROMPT = (
    "You are helping a blind or low-vision user identify playing cards in their hand. "
    "When several chronological frames are given, inspect them in order from earliest to "
    "latest. If a finger is pointing to and resting on one card in the latest frame, "
    "identify only that single card by rank and suit, for example 'Nine of hearts.' If the "
    "finger instead moves across several cards during the sequence, report every card you "
    "can clearly recognize that the finger swept over, in the order visited, as one spoken "
    "list such as 'Nine of hearts, two of spades, jack of diamonds.' When only one static "
    "image is available, identify only the single card a visible pointing finger rests on; "
    "do not claim a swipe happened from one image. Use ranks ace, two, three, four, five, "
    "six, seven, eight, nine, ten, jack, queen, or king together with a suit of clubs, "
    "diamonds, hearts, or spades; there are no jokers. Do not guess a card that is tilted, "
    "partly hidden, or poorly lit; report only the cards you can confidently read, and if "
    "none are readable say so plainly. Return one concise spoken sentence suitable for "
    "text-to-speech with no lists, labels, or extra commentary."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I could not clearly recognize a card from this view."
RECENT_WINDOW_SECONDS = 3.0
MAX_FRAMES_PER_CALL = 6


def _select_frames(frames):
    if len(frames) <= MAX_FRAMES_PER_CALL:
        return frames
    step = len(frames) / MAX_FRAMES_PER_CALL
    return [frames[int(i * step)] for i in range(MAX_FRAMES_PER_CALL)]


async def analyze_frames(images):
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
    del runtime, input_data
    if image is None:
        return FALLBACK_TEXT
    return await analyze_frames([image])


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)
    try:
        recent = runtime.get_recent_frames(seconds=RECENT_WINDOW_SECONDS) or [frame]
        images = [selected.image for selected in _select_frames(recent)]
        text = await analyze_frames(images)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    runtime.clear_state()
