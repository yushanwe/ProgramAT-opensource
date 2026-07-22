"""Declarative hosted-video tool for recent hand gesture identification."""

TOOL_NAME = "recent_hand_gesture_identifier"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "minimum_span_seconds": 4,
    "minimum_unique_frames": 8,
}
TOOL_PROMPT = (
    "Watch this chronological short video of the most recent few seconds and "
    "identify the hand gesture or brief meaning the person just expressed, "
    "using both movement across frames and any stable hand pose near the end "
    "when helpful. Return one concise, audio-friendly sentence only. If no hand "
    "gesture is visible or the evidence is unclear, say that clearly."
)
