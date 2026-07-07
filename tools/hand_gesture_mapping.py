"""
Hand Gesture Mapping Tool

Identifies visible hand gestures and explains their likely emotional or
conversational meaning for blind or low vision users.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from model_router_client import copilot_llm_call
from litellm_utils import pil_image_to_data_uri

TOOL_NAME = "hand_gesture_mapping"
TASK_CATEGORY = "general_reasoning"
TRANSCRIPT_KEYS = (
    "spoken_text",
    "speech_text",
    "transcript",
    "conversation_text",
    "context_text",
)


def extract_conversation_context(input_data: Optional[Dict[str, Any]]) -> str:
    """Pull optional spoken context from common input keys."""
    if not isinstance(input_data, dict):
        return ""

    for key in TRANSCRIPT_KEYS:
        raw_value = input_data.get(key)
        context = " ".join(str(raw_value).split()).strip() if raw_value is not None else ""
        if context:
            return context

    return ""


def build_gesture_prompt(conversation_context: str = "") -> str:
    """Build the routed prompt for gesture interpretation."""
    base_prompt = (
        "You are describing another person's hand gesture for a blind user during a conversation. "
        "Identify the most likely visible hand gesture, such as pointing, waving, thumbs up, "
        "thumbs down, open palm, raised hand, or crossed/closed-off hands. "
        "Briefly explain what the gesture likely means about the speaker's tone, attitude, or intent. "
        "If the hand is only partly visible or the gesture is ambiguous, say that clearly and then "
        "describe your best guess instead of pretending to be certain. "
        "Highlight both consistent and conflicting nonverbal cues when relevant. "
        "Keep the answer concise, natural, and audio-friendly."
    )

    if not conversation_context:
        return base_prompt

    return (
        f"{base_prompt} Compare the gesture with this spoken context and mention any mismatch or reinforcement: "
        f"\"{conversation_context}\"."
    )


def format_response(response: str) -> str:
    """Normalize model output for speech."""
    formatted = " ".join(str(response).split()).strip()
    if not formatted:
        return "I could not confidently identify a hand gesture."
    if formatted[-1] not in ".!?":
        formatted += "."
    return formatted


def analyze_hand_gesture(
    image: np.ndarray,
    conversation_context: str = "",
    api_key: Optional[str] = None,
    color_order: str = "bgr",
) -> Dict[str, Any]:
    """Run the routed gesture analysis."""
    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    elif image.ndim == 3 and image.shape[2] == 1:
        pil_image = Image.fromarray(image.squeeze(axis=2))
    elif color_order.lower() == "rgb":
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image[:, :, ::-1])

    width, height = pil_image.size
    if width > 1024 or height > 1024:
        scale = min(1024 / width, 1024 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    image_data_uri = pil_image_to_data_uri(pil_image)
    prompt = build_gesture_prompt(conversation_context)

    metadata = {
        "tool_name": TOOL_NAME,
        "route_text": "identify visible hand gestures and explain their likely emotional meaning",
    }
    if api_key:
        metadata["api_key"] = api_key

    result = copilot_llm_call(
        capability=TASK_CATEGORY,
        goal="Identify the visible hand gesture and explain its likely conversational meaning.",
        messages=[
            {"role": "system", "content": "Keep responses concise, natural, and audio-friendly."},
            {"role": "user", "content": prompt},
        ],
        images=[image_data_uri],
        metadata=metadata,
    )

    return {
        "success": True,
        "description": format_response(result.get("response", "")),
        "artifact": result.get("artifact"),
        "capability": result.get("capability", TASK_CATEGORY),
    }


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
    """Entry point for the hand gesture mapping tool."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {
                "type": "error",
                "text": "No camera image available for hand gesture mapping.",
            },
            "text": "No camera image available for hand gesture mapping.",
        }

    input_data = input_data or {}

    try:
        result = analyze_hand_gesture(
            image=image,
            conversation_context=extract_conversation_context(input_data),
            api_key=input_data.get("api_key"),
            color_order=input_data.get("color_order", "bgr"),
        )
        return result["description"]
    except Exception as exc:
        message = format_response(f"Hand gesture mapping failed: {exc}")
        return {
            "audio": {
                "type": "error",
                "text": message,
            },
            "text": message,
        }


__all__ = [
    "TOOL_NAME",
    "TASK_CATEGORY",
    "TRANSCRIPT_KEYS",
    "analyze_hand_gesture",
    "build_gesture_prompt",
    "extract_conversation_context",
    "format_response",
    "main",
]
