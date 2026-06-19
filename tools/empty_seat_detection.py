"""
Empty Seat Detection Tool for Visually Impaired Users

Helps blind or low vision users locate available seating in crowded spaces.
Points a camera at the room to get a spoken report of empty seats and
directional guidance to reach the nearest one.

Example output:
"There are 3 empty chairs. The closest one is slightly to your left, about
 5 steps away. Walk forward and slightly left to reach it."
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from litellm_utils import extract_text, pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "empty_seat_detection"
TASK_CATEGORY = "visual_reasoning"

SYSTEM_PROMPT = (
    "You are an assistive-technology guide for a blind user. "
    "Describe only what is relevant to finding an empty seat. "
    "Keep the response under 40 words and suitable for text-to-speech."
)

USER_PROMPT = (
    "Look at this image and tell me: are there any empty seats, chairs, benches, "
    "or couches visible? If yes, how many, roughly where are they relative to me "
    "(left, right, center, near, far), and what direction should I walk to reach "
    "the closest one? If no empty seats are visible, say so briefly."
)


def _to_data_uri(image: np.ndarray) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return pil_image_to_data_uri(pil_image)


def main(image: np.ndarray, input_data: dict = None) -> str:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available. Please point the camera at the room."

    try:
        data_uri = _to_data_uri(image)
    except Exception:
        return "Unable to process the camera image. Please try again."

    try:
        response = copilot_llm_call(
            task_category=TASK_CATEGORY,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            images=[data_uri],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "locate empty seats and guide a blind user toward the nearest available one",
            },
        )
        result = extract_text(response)
        return result if result else "No empty seats detected in the current view."
    except Exception:
        return "Seat detection failed. Please try again."
