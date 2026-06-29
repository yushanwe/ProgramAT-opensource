from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from model_router_client import copilot_llm_call


TOOL_NAME = "nearest_exit_locator"
EXIT_DETECTION_CATEGORY = "object_detection"
NAVIGATION_CATEGORY = "navigation"


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _pil_image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}"


def _response_to_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        text = response.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        audio = response.get("audio")
        if isinstance(audio, dict):
            audio_text = audio.get("text")
            if isinstance(audio_text, str) and audio_text.strip():
                return audio_text.strip()
    if hasattr(response, "choices"):
        try:
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        parts.append(part.get("text", ""))
                    elif hasattr(part, "text"):
                        parts.append(getattr(part, "text", "") or "")
                return "".join(parts).strip()
        except Exception:
            pass
    return str(response).strip()


def _build_exit_detection_prompt() -> str:
    return (
        "Identify the nearest visible exit door for a blind user. "
        "Describe only the nearest exit door location relative to the camera, using straight ahead, left, or right, "
        "plus an approximate distance cue if possible. "
        "If no exit door is visible, say 'No exit door visible.' "
        "Keep the response concise and audio-friendly."
    )


def _build_navigation_prompt(exit_detection_text: str) -> str:
    return (
        f"Stage 1 found: {exit_detection_text}\n"
        "Guide a blind user toward that exit door with short body-relative instructions. "
        "Use left, right, or straight and mention when to stop. "
        "If stage 1 found no exit door, say you cannot guide yet and ask the user to scan again. "
        "Keep the response concise and audio-friendly."
    )


def main(image, input_data=None):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {"type": "error", "text": "No camera image available for exit guidance."},
            "text": "No camera image available for exit guidance.",
        }

    if input_data is None:
        input_data = {}

    try:
        image_data_uri = _pil_image_to_data_uri(convert_cv2_to_pil(image))

        exit_detection_result = copilot_llm_call(
            task_category=EXIT_DETECTION_CATEGORY,
            messages=[
                {"role": "system", "content": "Keep responses concise and audio-friendly."},
                {"role": "user", "content": _build_exit_detection_prompt()},
            ],
            images=[image_data_uri],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "identify the nearest visible exit door",
                "stage_name": "Stage 1",
                "stage_goal": "Identify the nearest exit door.",
            },
        )
        exit_detection_text = _response_to_text(exit_detection_result) or "No exit door visible."

        navigation_result = copilot_llm_call(
            task_category=NAVIGATION_CATEGORY,
            messages=[
                {"role": "system", "content": "Keep responses concise and audio-friendly."},
                {"role": "user", "content": _build_navigation_prompt(exit_detection_text)},
            ],
            images=[image_data_uri],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": "guide the user toward the detected exit door",
                "stage_name": "Stage 2",
                "stage_goal": "Guide me toward the detected exit door.",
                "previous_stage_output": exit_detection_text,
            },
        )
        navigation_text = _response_to_text(navigation_result)
        return navigation_text or "I could not guide you to an exit. Please scan again."
    except Exception as exc:
        message = f"Exit guidance failed: {exc}"
        return {
            "audio": {"type": "error", "text": message},
            "text": message,
        }
