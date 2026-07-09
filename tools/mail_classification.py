"""
Mail Classification Tool for Blind and Low Vision Users

Classifies mail items in a photo as 'important' or 'junk' using a three-stage
structured visual pipeline. Helps blind and low vision users quickly sort mail
by hearing a spoken summary of each item's classification.

Stages:
  1. Identify personal mail or bills (structured_visual_understanding)
  2. Exclude likely advertisements (structured_visual_understanding)
  3. Categorize remaining letters as 'important' or 'junk' (general_reasoning)
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from litellm_utils import pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "mail_classification"


def _to_data_uri(image: np.ndarray) -> str:
    """Convert a BGR numpy array to a JPEG data URI."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return pil_image_to_data_uri(pil_image)


def _get_artifact(result: dict) -> str:
    """Extract the most useful text from a copilot_llm_call result."""
    return result.get("artifact") or result.get("response", "")


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Classify mail in a photo as important or junk and return an audio-friendly result.

    Args:
        image: Camera frame as a numpy array (BGR format from OpenCV).
        input_data: Unused; reserved for future configuration.

    Returns:
        An audio-friendly string describing the classification of each visible
        mail item, e.g. "Phone bill: important. Pizza coupon: junk."
        If the image cannot be processed, returns a spoken error message.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available for mail classification."

    image_uri = _to_data_uri(image)

    # Stage 1 — Identify personal mail and bills
    stage1 = copilot_llm_call(
        capability="structured_visual_understanding",
        goal="Identify letters that are like personal mail or bills.",
        messages=[
            {
                "role": "user",
                "content": (
                    "Look at this photo of mail. "
                    "List every item that appears to be personal mail, a bill, "
                    "a bank statement, an official notice, or similar important correspondence. "
                    "For each item, briefly note what it is (e.g. 'electric bill', 'bank statement'). "
                    "If none are found, say 'none'."
                ),
            }
        ],
        images=[image_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "identify personal mail and bills in a photo",
        },
    )
    personal_mail_artifact = _get_artifact(stage1)

    # Stage 2 — Exclude advertisements
    stage2 = copilot_llm_call(
        capability="structured_visual_understanding",
        goal="Exclude letters that are like advertisements.",
        messages=[
            {
                "role": "user",
                "content": (
                    "Look at this photo of mail. "
                    "List every item that appears to be an advertisement, promotional flyer, "
                    "coupon, catalog, or unsolicited marketing material. "
                    "For each item, briefly note what it is (e.g. 'pizza coupon', 'store catalog'). "
                    "If none are found, say 'none'. "
                    f"Previously identified personal mail or bills: {personal_mail_artifact}"
                ),
            }
        ],
        images=[image_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "identify advertisements and junk mail in a photo",
            "previous_stage_artifact": personal_mail_artifact,
        },
    )
    ads_artifact = _get_artifact(stage2)

    # Stage 3 — Categorize into 'important' or 'junk'
    stage3 = copilot_llm_call(
        capability="general_reasoning",
        goal="Categorize letters into 'important' or 'junk'.",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant helping a blind user sort their mail. "
                    "Respond in plain, concise spoken language. "
                    "For each mail item visible, state its label and whether it is important or junk. "
                    "If you cannot identify any mail, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Based on the mail photo and the analysis below, "
                    "produce a final spoken summary categorizing each visible mail item "
                    "as 'important' or 'junk'. "
                    "Keep the response short and natural for text-to-speech. "
                    f"Personal mail and bills found: {personal_mail_artifact}. "
                    f"Advertisements and junk found: {ads_artifact}."
                ),
            },
        ],
        images=[image_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "categorize mail items as important or junk",
            "previous_stage_artifact": ads_artifact,
        },
    )

    response = stage3.get("response", "").strip()
    if not response:
        return "I could not recognize the mail in this image."
    return response
