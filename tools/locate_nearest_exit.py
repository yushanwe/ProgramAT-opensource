"""
Locate Nearest Exit Tool for Blind Users

Identifies the nearest exit door in a building and guides the user toward it.

Stages:
  1. Detect exit doors using object_detection
  2. Provide navigation guidance toward the detected exit using navigation
"""

from __future__ import annotations

from model_router_client import copilot_llm_call
from litellm_utils import extract_text

TOOL_NAME = "locate_nearest_exit"


def _get_text(response) -> str:
    """Extract plain text from a copilot_llm_call response."""
    if hasattr(response, "choices"):
        return extract_text(response)
    return str(response)


def main(image, input_data=None):
    """
    Locate the nearest exit door and guide the user toward it.

    Args:
        image: Camera frame as numpy array (BGR format from OpenCV)
        input_data: Optional dict (unused)

    Returns:
        Audio-friendly navigation guidance string.
    """
    if image is None:
        return "No camera image available. Please point the camera at the room."

    # Stage 1: Locate the nearest exit door
    detection_response = copilot_llm_call(
        task_category="object_detection",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a vision assistant helping a blind user locate the nearest exit in a building. "
                    "Identify any exit signs, emergency exit doors, or exit door indicators visible in the image. "
                    "Describe what you see: the number of exits, their approximate position (left, right, center, "
                    "near or far), and any relevant details like signage color or distance. "
                    "Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": "Locate any exit doors or exit signs in this image. Describe their position and any key details.",
            },
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "stage_name": "Stage 1",
            "stage_goal": "Locate the nearest exit door",
            "route_text": "detect and locate exit doors or exit signs in a building interior",
        },
    )

    detection_text = _get_text(detection_response)

    # Stage 2: Guide the user toward the detected exit
    navigation_response = copilot_llm_call(
        task_category="navigation",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a navigation assistant helping a blind user reach the nearest exit door. "
                    "Based on the exit detection results and the camera image, provide clear, step-by-step "
                    "walking directions. Use clock positions (1 to 3 or 9 to 12 o'clock) or simple directions "
                    "(left, right, straight ahead) to guide the user. Keep guidance brief and actionable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Exit detection result: {detection_text}\n\n"
                    "Using the camera image and the detection results, give me walking directions to reach the nearest exit."
                ),
            },
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "stage_name": "Stage 2",
            "stage_goal": "Guide the user toward the detected exit door",
            "route_text": "provide walking navigation guidance toward a detected exit door",
            "previous_stage_output": detection_text,
        },
    )

    navigation_text = _get_text(navigation_response)

    if not navigation_text or navigation_text.strip() == "":
        return "No exit found in view. Try turning slowly to scan for exit signs."

    return navigation_text
