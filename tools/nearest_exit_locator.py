"""
Nearest Exit Locator Tool

Identifies and guides a blind or low-vision user to the nearest exit door
inside a large building.

Stages:
  Stage 1 — object_detection: Detect exit doors and exit signs in the frame
            and determine their location and direction.
  Stage 2 — navigation: Produce turn-by-turn verbal instructions that guide
            the user to the detected exit.

Returns audio-friendly guidance suitable for text-to-speech.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from PIL import Image
from litellm_utils import extract_text, pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "nearest_exit_locator"


def main(image: np.ndarray, input_data: Any = None) -> str:
    """Detect the nearest exit and guide the user toward it."""

    # Convert numpy BGR image to a data URI for the model calls
    rgb = image[..., ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
    pil_image = Image.fromarray(rgb.astype("uint8"))
    image_uri = pil_image_to_data_uri(pil_image)

    # ------------------------------------------------------------------
    # Stage 1 — object_detection
    # Detect exit doors and exit signs and report their location/direction.
    # ------------------------------------------------------------------
    detection_result = copilot_llm_call(
        task_category="object_detection",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistive-technology tool for blind users. "
                    "Respond concisely in plain English, suitable for text-to-speech."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Look at this building interior image. "
                    "Detect any exit doors, emergency exit signs, 'EXIT' signs, "
                    "or emergency exit indicators. "
                    "For each one found, describe: (1) what it is, "
                    "(2) its approximate clock-face direction (using only 1, 2, 3, "
                    "9, 10, 11, or 12 o'clock), and (3) its estimated distance "
                    "(very close, close, medium, or far). "
                    "If no exit is visible, say 'No exit detected'. "
                    "Focus on the nearest or most prominent exit."
                ),
            },
        ],
        images=[image_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "stage_name": "Stage 1",
            "stage_goal": "Detect exit doors and signs and determine their direction",
            "route_text": "detect exit doors and exit signs in a building interior",
        },
    )

    detection_text = extract_text(detection_result)

    # If no exit was found, return early with a helpful message
    if "no exit" in detection_text.lower():
        return (
            "No exit sign or exit door detected in the current view. "
            "Try turning slowly to scan your surroundings."
        )

    # ------------------------------------------------------------------
    # Stage 2 — navigation
    # Use the detection output to produce turn-by-turn walking guidance.
    # ------------------------------------------------------------------
    navigation_result = copilot_llm_call(
        task_category="navigation",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a navigation assistant for a blind user inside a building. "
                    "Give clear, step-by-step verbal directions. "
                    "Respond in plain English suitable for text-to-speech. "
                    "Be concise but complete."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The exit detection analysis says:\n{detection_text}\n\n"
                    "Based on this, give step-by-step navigation instructions "
                    "to guide a blind user from their current position to the nearest exit. "
                    "Use clock-face directions (e.g. '12 o'clock is straight ahead', "
                    "'3 o'clock is to your right'). "
                    "Include any turns, distance estimates, and when to stop. "
                    "Keep instructions clear and actionable."
                ),
            },
        ],
        images=[image_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "stage_name": "Stage 2",
            "stage_goal": "Provide turn-by-turn navigation instructions to the nearest exit",
            "route_text": "turn-by-turn navigation to guide a blind user to the nearest building exit",
            "previous_stage_output": detection_text,
        },
    )

    navigation_text = extract_text(navigation_result)
    return navigation_text
