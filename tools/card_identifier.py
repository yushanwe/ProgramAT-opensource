"""Declarative temporal tool for identifying pointed or swiped playing cards."""

TOOL_NAME = "card_identifier"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 1,
    "minimum_span_seconds": 4,
    "minimum_unique_frames": 6,
}
TOOL_PROMPT = """
When multiple chronological recent frames are available, inspect them from
earliest to latest to determine the user's intent. If the user points at one
card, identify only that card. If the user swipes across multiple cards,
identify each clearly readable card in swipe order and return them as a comma-
separated list. Use only standard ranks (ace through king) and suits (clubs,
diamonds, hearts, spades).

When only one image is available, use only clear static evidence: identify a
single clearly indicated card, or list multiple clearly readable cards from
left to right if no motion cue is available. Do not invent unseen motion or
cards.

Return only the concise spoken answer, for example "Nine of hearts." or "Nine
of hearts, two of spades, jack of diamonds." If no card can be read clearly,
return exactly "No clear card detected."
"""
