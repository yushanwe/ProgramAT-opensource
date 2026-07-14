"""Hand Gesture Interpreter — take-photo tool.

Identifies nonverbal hand gestures in the camera frame and describes the
likely conversational meaning for a blind or low-vision user.
"""

from litellm_utils import call_take_photo_baseline_vlm

TOOL_PROMPT = (
    "Look at the hands in this image and identify any nonverbal hand gesture being made. "
    "Name the gesture (for example: pointing, waving, thumbs up, thumbs down, open palm, "
    "peace sign, beckoning, stop, crossed fingers) and briefly explain its likely meaning "
    "in a conversation. If multiple gestures are visible focus on the most prominent one. "
    "If no clear hand gesture is visible, say so. Keep the answer to one or two short sentences."
)


def main(image, input_data=None):
    """Return an audio-friendly description of any hand gesture in the image."""
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
