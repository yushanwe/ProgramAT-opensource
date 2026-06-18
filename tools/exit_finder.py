"""Exit Finder Tool for Blind and Low-Vision Users

Identifies the nearest exit door visible in the camera frame and provides
concise, audio-friendly directional guidance to help the user navigate toward it.
"""

from __future__ import annotations

import numpy as np

from litellm_utils import extract_text
from model_router_client import copilot_llm_call

TOOL_NAME = "exit_finder"
TASK_CATEGORY = "visual_reasoning"

SYSTEM_PROMPT = (
    "You assist blind users navigating indoors. "
    "Be concise — responses will be spoken aloud via text-to-speech. "
    "Use body-relative directions (left, right, ahead, behind) and simple distance cues."
)

EXIT_PROMPT = (
    "Look at this image and identify the nearest exit. "
    "An exit may be a door marked EXIT or with an exit sign, a fire escape door, "
    "an emergency exit, or any clearly visible doorway leading outside. "
    "If you see an exit, describe in one or two short sentences: "
    "(1) where it is relative to the viewer (e.g., 'directly ahead', 'to your left'), "
    "and (2) any distinguishing features (e.g., 'green exit sign above', 'red door'). "
    "If no exit is visible, say so briefly and suggest the user slowly turn to scan the area. "
    "Keep the response under 30 words."
)


def main(image: np.ndarray, input_data=None) -> str:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available. Please point your camera around the room."

    try:
        response = copilot_llm_call(
            task_category=TASK_CATEGORY,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EXIT_PROMPT},
            ],
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "identify nearest exit door and provide directional guidance for blind user",
            },
        )
        result = extract_text(response)
        return result if result else "No exit visible. Slowly turn to scan the area."
    except Exception:
        return "Could not analyze the scene. Please try again."
