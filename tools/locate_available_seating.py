"""
Locate Available Seating Tool

Helps blind and low-vision users find an empty chair or seat in crowded spaces
by analysing a camera frame and returning concise audio-friendly guidance about
where to walk.
"""

from typing import Any

from litellm_utils import call_take_photo_baseline_vlm

TOOL_NAME = "locate_available_seating"

TOOL_PROMPT = (
    "You are assisting a blind person who needs to sit down in a crowded space. "
    "Look at this image and identify any empty chairs, benches, or seats that are not occupied by a person. "
    "If you find empty seating, describe its location in simple directional terms (e.g. 'slightly to your left', "
    "'straight ahead', 'to your right') and estimate roughly how far away it is (e.g. 'about five steps'). "
    "Then give one short navigation instruction to guide the user toward it. "
    "If no empty seating is visible, say so clearly and suggest the user scan around. "
    "Keep the entire response under three sentences and suitable for text-to-speech."
)


def main(image: Any, input_data: Any = None) -> str:
    """Find an empty seat and guide the user toward it.

    Args:
        image: Camera frame as a numpy ndarray (BGR), PIL Image, or data URI.
        input_data: Unused; accepted for interface compatibility.

    Returns:
        Audio-friendly string describing available seating and navigation guidance.
    """
    if image is None:
        return "No camera image available. Please point your camera at the seating area."

    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
