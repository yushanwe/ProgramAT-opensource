"""Declarative hosted-video tool for temporal sign language recognition."""

TOOL_NAME = "temporal_sign_language_gesture_recognition"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "minimum_span_seconds": 5,
    "minimum_unique_frames": 12,
}
TOOL_PROMPT = (
    "Watch this short chronological video of the signer and use early, middle, "
    "and late moments to identify the sign or short signed phrase that was just "
    "completed. Return only a concise, audio-friendly final answer naming that "
    "sign or phrase. If hands are not clearly visible or confidence is low, say: "
    "\"I couldn't confidently identify the recent sign.\""
)
