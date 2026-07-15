"""
Find Nearest Exit Tool for Blind Users

Locates the nearest exit door in the camera view and provides step-by-step
spoken directions to help the user reach it safely.

This tool uses a two-stage AI pipeline:
  Stage 1 — object_detection_localization: Find and locate the nearest exit door.
  Stage 2 — navigation: Generate step-by-step directions to guide the user there.

## Usage from Mobile App:
The tool automatically receives a camera frame from the mobile app and returns
spoken directions to the nearest exit.

## Example Return Format:
"Exit door detected ahead and slightly to your right. Walk forward about ten
steps, then turn right to reach the exit."
"""

import numpy as np

from model_router_client import copilot_llm_call

TOOL_NAME = "find_nearest_exit"
TOOL_PROMPT = (
    "Locate the nearest exit door in the image and provide step-by-step directions "
    "to help a blind user reach it safely."
)


def main(image: np.ndarray, input_data: dict) -> str:
    """
    Main entry point for the Find Nearest Exit tool.

    Args:
        image: Camera frame as an OpenCV BGR numpy array.
        input_data: Dictionary of optional runtime parameters (unused).

    Returns:
        Concise, audio-friendly spoken directions to the nearest exit door,
        or an error message if no image is available.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available. Please point the camera at your surroundings."

    # Stage 1: Locate the nearest exit door.
    stage1 = copilot_llm_call(
        capability="object_detection_localization",
        images=[image],
        goal="Locate the nearest exit door.",
    )

    # Stage 2: Generate spoken directions to the exit using the detection result.
    stage2 = copilot_llm_call(
        capability="navigation",
        images=[image],
        metadata={"previous_stage_artifact": stage1["artifact"]},
        goal="Guide me to the exit.",
    )

    return stage2["response"]
