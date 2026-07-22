"""
Locate Empty Chairs Tool for Blind Users

Helps blind or low vision users find available seating in crowded spaces by
identifying empty chairs, benches, or couches and giving concise spoken
guidance toward the nearest one.
"""

from litellm_utils import call_take_photo_vlm

TOOL_NAME = "locate_empty_chairs"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Identify any chairs, benches, couches, or other seating visible in the scene, "
    "determine which seats appear unoccupied, select the nearest suitable option, "
    "and give concise spoken guidance toward it. "
    "If no empty seat is visible, say so. Return only the final spoken guidance."
)


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
