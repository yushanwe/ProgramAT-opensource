"""
Mail Categorizer Tool

Distinguishes important from junk mail by analyzing photos of mail items.
Detects mail objects, reads visible text, then categorizes each piece
as important or junk and speaks the result to the user.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from litellm_utils import pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "mail_categorizer"


def _image_to_data_uri(image: np.ndarray) -> str:
    """Convert an OpenCV BGR image to a data URI for LLM calls."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return pil_image_to_data_uri(pil_img)


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Categorize mail items in the image as important or junk.

    Parameters
    ----------
    image : np.ndarray
        OpenCV BGR camera frame. May be None.
    input_data : dict, optional
        Unused; reserved for future configuration.

    Returns
    -------
    str or dict
        Audio-friendly categorization result spoken to the user.
    """
    if image is None:
        return "No image received. Please point the camera at your mail."

    try:
        image_uri = _image_to_data_uri(image)
    except Exception:
        return "Could not process the image. Please try again."

    # Stage 1 – Detect mail objects in the frame
    detection = copilot_llm_call(
        capability="object_detection_localization",
        goal="Detect mail items in the image such as envelopes, letters, packages, and postcards.",
        images=[image_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "detect envelopes, letters, packages, and postcards in the camera frame",
            "target_labels": ["envelope", "letter", "package", "postcard", "mail", "box"],
        },
    )

    # Stage 2 – Read text visible on the detected mail items
    ocr_result = copilot_llm_call(
        capability="ocr",
        goal="Read all text visible on the mail items, including sender names, return addresses, subject lines, and any promotional labels.",
        images=[image_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "extract text from mail items including sender, address, and subject",
            "previous_stage_artifact": detection.get("artifact"),
        },
    )

    # Stage 3 – Reason about importance of each mail piece
    reasoning = copilot_llm_call(
        capability="general_reasoning",
        goal=(
            "Based on the detected mail items and the text read from them, "
            "categorize each piece of mail as important or junk. "
            "Important mail includes bills, legal notices, government correspondence, "
            "bank statements, medical notices, and personal letters. "
            "Junk mail includes advertisements, promotional offers, and unsolicited catalogs. "
            "Provide a concise, audio-friendly summary telling the user which pieces are "
            "important and which are junk. Speak as if addressing a blind user directly."
        ),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistive technology helper for a blind user. "
                    "Keep your response concise and natural for text-to-speech. "
                    "Lead with the most important finding."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Detected mail objects: {detection.get('artifact')}\n"
                    f"Text on mail: {ocr_result.get('artifact')}"
                ),
            },
        ],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "categorize mail as important or junk based on detected objects and OCR text",
            "previous_stage_artifact": ocr_result.get("artifact"),
        },
    )

    response = reasoning.get("response", "")
    if not response:
        return "Could not categorize the mail. Please try again with better lighting."

    return response
