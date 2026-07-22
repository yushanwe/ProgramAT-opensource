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
earliest to latest and decide between pointing and swiping. Only treat it as a
swipe when there is a clear continuous left-to-right or right-to-left motion
across multiple cards. For a swipe, return all clearly readable cards in that
swipe direction order as a comma-separated list.

Otherwise treat it as pointing, including when the user shifts from one pointed
card to another without a continuous sweep. For pointing, return exactly one
card: the last clearly indicated card from the latest stable frames.

When only one image is available, use only static evidence and return exactly
one clearly indicated card; do not list multiple cards without motion evidence.
Use only standard ranks (ace through king) and suits (clubs, diamonds, hearts,
spades), and do not invent unseen cards or motion.

Return only the concise spoken answer, for example "Nine of hearts." or "Nine
of hearts, two of spades, jack of diamonds." If evidence is insufficient,
return exactly "No clear card detected."
"""
