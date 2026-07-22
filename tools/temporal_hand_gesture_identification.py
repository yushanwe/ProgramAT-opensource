"""Declarative hosted-video tool for identifying a recent hand gesture."""

TOOL_NAME = "temporal_hand_gesture_identification"
EXECUTION_MODE = "hosted_video_streaming"
VIDEO_CONFIG = {
    "window_seconds": 6,
    "interval_seconds": 3,
    "minimum_span_seconds": 5,
    "minimum_unique_frames": 12,
}
TOOL_PROMPT = (
    "Watch this chronological short video of the signer's recent hand movements. "
    "Use the gesture's motion and the late stable frames to identify the hand "
    "gesture they just made, whether it was brief or held for a few seconds. "
    "Return only the concise gesture name; if the hands are not visible clearly "
    "enough or no distinct gesture can be determined from the recent history, "
    "say \"No clear hand gesture.\""
)
