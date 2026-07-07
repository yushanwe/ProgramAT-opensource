"""Detect and explain common hand gestures for nonverbal communication."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from model_router_client import copilot_llm_call

TOOL_NAME = "hand_gesture_detection"
DEFAULT_GESTURES = ("pointing", "waving", "thumbs up", "raised hand")


def _normalize_requested_gestures(input_data: Dict[str, Any]) -> str:
    raw = input_data.get("gestures") if isinstance(input_data, dict) else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray, str)):
        items = [str(item).strip() for item in raw if str(item).strip()]
        if items:
            return ", ".join(items)
    return ", ".join(DEFAULT_GESTURES)


def _has_usable_stage_artifact(stage_result: Dict[str, Any]) -> bool:
    if not isinstance(stage_result, dict):
        return False

    if stage_result.get("success") is False:
        return False

    artifact = stage_result.get("artifact")
    if not artifact:
        return False

    confidence = stage_result.get("confidence")
    if confidence is None and isinstance(artifact, dict):
        confidence = artifact.get("confidence")

    if isinstance(confidence, (int, float)) and confidence < 0.4:
        return False

    return True


def _build_stage1_goal(gestures: str) -> str:
    return (
        "Provide information about nonverbal hand gestures during conversations. "
        f"Focus on: {gestures}. Include where each gesture appears in the scene."
    )


def _build_stage2_goal(gestures: str) -> str:
    return (
        "Briefly explain the meaning of each hand gesture. "
        f"Prioritize: {gestures}. Keep response concise and audio-friendly."
    )


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Any:
    """Run two-stage gesture detection and meaning explanation."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {
                "type": "error",
                "text": "No camera image available for hand gesture detection.",
            },
            "text": "No camera image available for hand gesture detection.",
        }

    config = input_data if isinstance(input_data, dict) else {}
    gestures = _normalize_requested_gestures(config)

    try:
        stage1 = copilot_llm_call(
            capability="spatial_reasoning",
            goal=_build_stage1_goal(gestures),
            images=[image],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "Detect visible hand gestures and their relative positions.",
            },
        )

        stage2_metadata: Dict[str, Any] = {
            "tool_name": TOOL_NAME,
            "route_text": "Explain likely meanings of detected gestures for conversation context.",
        }
        stage2_messages = [
            {
                "role": "system",
                "content": "Respond with natural, concise speech suitable for text-to-speech.",
            }
        ]

        if _has_usable_stage_artifact(stage1):
            stage2_metadata["previous_stage_artifact"] = stage1["artifact"]
            stage1_summary = str(stage1.get("response", "")).strip()
            if stage1_summary:
                stage2_messages.append(
                    {
                        "role": "user",
                        "content": f"Detected gesture details: {stage1_summary}",
                    }
                )

        stage2 = copilot_llm_call(
            capability="general_reasoning",
            goal=_build_stage2_goal(gestures),
            messages=stage2_messages,
            images=[image],
            metadata=stage2_metadata,
        )

        response = str(stage2.get("response", "")).strip()
        if response:
            return response
        return "I could not confidently identify a clear hand gesture right now."
    except Exception:
        return {
            "audio": {
                "type": "error",
                "text": "I could not process hand gestures right now. Please try again.",
            },
            "text": "I could not process hand gestures right now. Please try again.",
        }


__all__ = ["main"]
