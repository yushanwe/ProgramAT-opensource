"""Clothing matching assistant for blind and low-vision users.

Point the camera at your outfit to hear whether your clothing items match.
"""

from __future__ import annotations

from model_router_client import llm_call

TOOL_NAME = "clothing_match"
TASK_CATEGORY = "visual_reasoning"


def main(image, input_data=None):
    if image is None:
        return "No image received. Please point the camera at your outfit."

    prompt = (
        "Look at the clothing items visible in this image. "
        "Briefly assess whether the colors, patterns, and styles of the outfit match well. "
        "Mention the specific items and their colors, then give a clear verdict. "
        "Keep the response concise and natural for text-to-speech."
    )

    return llm_call(
        task_category=TASK_CATEGORY,
        messages=[
            {"role": "system", "content": "You are a fashion assistant helping a blind user. Keep responses concise and audio-friendly."},
            {"role": "user", "content": prompt},
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "assess whether clothing items in an outfit match in color, pattern, and style",
        },
    )
