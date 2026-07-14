"""Exit Locator Tool

Locates the nearest exit door in the current camera frame and provides
concise, step-by-step spoken directions to help a blind or low-vision
user reach it safely in an unfamiliar indoor environment.
"""

from typing import Any

from litellm_utils import call_take_photo_baseline_vlm

TOOL_PROMPT = (
    "You are an assistive navigation assistant for a blind user in an unfamiliar indoor space. "
    "Examine this image and locate the nearest visible exit door or emergency exit sign. "
    "Describe what you see, then give clear, step-by-step spoken directions to reach that exit safely. "
    "Use simple language and clock-face positions to indicate direction: "
    "12 o'clock means straight ahead, 1-3 o'clock means to the right, 9-12 o'clock means to the left or straight ahead. "
    "Only use positions 1-3 and 9-12; positions 4-8 would be behind the camera. "
    "Keep the total response under 60 words and audio-friendly."
)


def main(image: Any, input_data: Any = None) -> str:
    """Locate the nearest exit and return spoken directions.

    Args:
        image: Camera frame supplied by the backend.
        input_data: Unused; reserved for future configuration.

    Returns:
        Audio-friendly string with exit location and step-by-step directions.
    """
    if image is None:
        return "No camera image available. Please point your camera at the room."

    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
