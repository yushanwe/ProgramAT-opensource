"""Card Hand Identification Tool for Blind Users

Identifies playing cards visible in the user's hand and reports each card's
rank and suit in natural spoken language for card game assistance.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from litellm_utils import pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "card_hand_identification"


def _image_to_data_uri(image: np.ndarray) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return pil_image_to_data_uri(pil_img)


def main(image: np.ndarray, input_data: dict = None) -> str:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available. Please point the camera at your cards."

    try:
        data_uri = _image_to_data_uri(image)
    except Exception:
        return "Could not process the camera image."

    result = copilot_llm_call(
        capability="structured_visual_understanding",
        goal="Identify the cards and their properties",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistive tool for blind card-game players. "
                    "List every playing card visible in the hand. "
                    "For each card state its rank and suit clearly. "
                    "If no cards are visible, say so. "
                    "Keep the response concise and natural for text-to-speech."
                ),
            },
            {
                "role": "user",
                "content": "What playing cards do you see in my hand? List each card's rank and suit.",
            },
        ],
        images=[data_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "identify playing cards in a hand and report rank and suit of each card",
        },
    )

    response = result.get("response", "").strip()
    if not response:
        return "No cards detected. Please point the camera at your hand of cards."
    return response
