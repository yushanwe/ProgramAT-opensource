"""Room lighting status tool."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np

from model_router_client import copilot_llm_call

TOOL_NAME = "room_lighting_status"
TASK_CATEGORY = "general_reasoning"
LIGHT_LEVELS = ("dark", "dim", "normal", "bright")
LIGHT_ORDER = {level: index for index, level in enumerate(LIGHT_LEVELS)}


def build_prompt(previous_level: Optional[str] = None) -> str:
    """Build the ambient-light classification prompt."""
    change_instruction = ""
    if previous_level in LIGHT_LEVELS:
        change_instruction = (
            f" Previous level was '{previous_level}'. Mention if change is significant."
        )

    return (
        "Determine current ambient room lighting. "
        "Classify exactly one level: dark, dim, normal, or bright. "
        "Respond in two lines:\n"
        "LEVEL: <dark|dim|normal|bright>\n"
        "MESSAGE: <short, audio-friendly sentence>."
        f"{change_instruction}"
    )


def extract_light_level(text: str) -> Optional[str]:
    """Extract normalized light level from model response."""
    match = re.search(r"\b(dark|dim|normal|bright)\b", text.lower())
    return match.group(1) if match else None


def extract_message(text: str, fallback_level: Optional[str]) -> str:
    """Extract user-facing spoken message."""
    message_match = re.search(r"^\s*MESSAGE:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if message_match:
        message = message_match.group(1).strip()
        if message:
            return message

    if fallback_level:
        return f"Lighting is {fallback_level}."
    return "Unable to determine lighting level."


def is_significant_change(previous_level: Optional[str], current_level: Optional[str]) -> bool:
    """Treat changes of two or more steps as significant."""
    if previous_level not in LIGHT_ORDER or current_level not in LIGHT_ORDER:
        return False
    return abs(LIGHT_ORDER[current_level] - LIGHT_ORDER[previous_level]) >= 2


def analyze_room_lighting(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run staged lighting analysis with one routed capability call."""
    payload = input_data or {}
    previous_level = payload.get("previous_level")
    if isinstance(previous_level, str):
        previous_level = previous_level.lower().strip()
    else:
        previous_level = None

    prompt = build_prompt(previous_level=previous_level)
    result = copilot_llm_call(
        capability=TASK_CATEGORY,
        goal="Determine the ambient brightness of the room.",
        messages=[
            {"role": "system", "content": "Keep responses concise and audio-friendly."},
            {"role": "user", "content": prompt},
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "classify room lighting and report dark, dim, normal, or bright",
        },
    )

    raw_response = str(result.get("response", "")).strip()
    level = extract_light_level(raw_response)
    spoken = extract_message(raw_response, fallback_level=level)

    significant_change = is_significant_change(previous_level, level)
    if significant_change and previous_level and level:
        spoken = f"Lighting changed from {previous_level} to {level}. {spoken}"

    return {
        "level": level,
        "message": spoken,
        "significant_change": significant_change,
    }


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Any:
    """Entry point for room lighting status."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {"type": "error", "text": "No camera image available."},
            "text": "No camera image available.",
        }

    try:
        analysis = analyze_room_lighting(image=image, input_data=input_data)
    except Exception as exc:
        return {
            "audio": {"type": "error", "text": f"Lighting check failed: {type(exc).__name__}."},
            "text": f"Lighting check failed: {type(exc).__name__}.",
        }

    level = analysis.get("level")
    message = analysis.get("message", "Unable to determine lighting level.")
    significant_change = bool(analysis.get("significant_change"))

    if significant_change:
        return {
            "audio": {"type": "warning", "text": message, "interrupt": True},
            "text": message,
        }
    if level in {"dark", "dim"}:
        return {
            "audio": {"type": "warning", "text": message},
            "text": message,
        }
    return message


__all__ = [
    "main",
    "analyze_room_lighting",
    "build_prompt",
    "extract_light_level",
    "extract_message",
    "is_significant_change",
]
