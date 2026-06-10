"""
Clothing Match Checker Tool

Checks whether top and bottom clothing in view match well and returns
concise, audio-friendly guidance for blind and low-vision users.
"""

import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from litellm_utils import (
    extract_text,
    pil_image_to_data_uri,
    resolve_api_key,
    resolve_model_name,
)

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False


DEFAULT_MODEL = "gemini-3-flash-preview"
STREAMING_WORD_LIMIT = 15
_last_stream_message = None


def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    """Resize oversized images while preserving aspect ratio."""
    height, width = image.shape[:2]
    max_width, max_height = max_size

    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def build_match_prompt(is_streaming: bool = False) -> str:
    """Build the prompt used to assess outfit matching."""
    length_rule = (
        "Keep your response to 15 words or fewer." if is_streaming else "Use one short sentence."
    )

    return (
        "You are helping a blind user decide whether their outfit matches. "
        "Look for the main shirt/top and pants/bottoms. "
        "Judge color harmony, pattern compatibility, and overall coordination. "
        "If only one clothing piece is visible, clearly say you need both shirt and pants visible. "
        "Respond directly to the user in natural speech. "
        f"{length_rule} "
        "Examples: 'Your shirt and pants match well.' "
        "or 'Your shirt and pants clash; try neutral pants.'"
    )


def format_for_audio(text: str, max_words: Optional[int] = None) -> str:
    """Normalize whitespace and optionally enforce a word limit."""
    formatted = " ".join((text or "").replace("\n", " ").split()).strip()

    if not formatted:
        return "I could not determine whether your outfit matches."

    if max_words and max_words > 0:
        words = formatted.split()
        if len(words) > max_words:
            formatted = " ".join(words[:max_words]).rstrip(".,;:") + "."

    return formatted


def analyze_clothing_match(
    image: np.ndarray,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    is_streaming: bool = False,
) -> Dict[str, Any]:
    """Run Gemini vision analysis to decide whether clothing matches."""
    if not LITELLM_AVAILABLE:
        return {
            "success": False,
            "message": "Outfit check is unavailable because LiteLLM is not installed.",
        }

    api_key = resolve_api_key(model_name, api_key)
    if not api_key:
        return {
            "success": False,
            "message": "API key not configured for outfit checking.",
        }

    try:
        processed = resize_image_if_needed(image)
        pil_image = convert_cv2_to_pil(processed)
        image_data_uri = pil_image_to_data_uri(pil_image)
        model = resolve_model_name(model_name, default_model=DEFAULT_MODEL)

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_match_prompt(is_streaming=is_streaming)},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            api_key=api_key,
        )

        message = extract_text(response)
        return {"success": True, "message": message}
    except Exception as exc:
        return {
            "success": False,
            "message": f"Outfit check failed: {str(exc)}",
        }


def _build_response(message: str, audio_type: str = "speech") -> Dict[str, Any]:
    """Build standard ProgramAT audio/text response dictionary."""
    return {
        "audio": {
            "type": audio_type,
            "text": message,
            "rate": 1.0,
            "interrupt": False,
        },
        "text": message,
    }


def main(image: np.ndarray, input_data: Optional[Dict] = None):
    """Main entry point for clothing match checker."""
    global _last_stream_message

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return _build_response("No camera image available for outfit checking.", audio_type="error")

    config = input_data if isinstance(input_data, dict) else {}
    is_streaming = bool(config.get("is_streaming", config.get("streaming", False)))
    api_key = config.get("api_key")
    model_name = config.get(
        "model",
        os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)),
    )

    result = analyze_clothing_match(
        image=image,
        api_key=api_key,
        model_name=model_name,
        is_streaming=is_streaming,
    )

    if not result.get("success"):
        if not is_streaming:
            _last_stream_message = None
        return _build_response(result.get("message", "Outfit check failed."), audio_type="error")

    max_words = STREAMING_WORD_LIMIT if is_streaming else None
    spoken_text = format_for_audio(result.get("message", ""), max_words=max_words)

    if is_streaming:
        if spoken_text == _last_stream_message:
            return ""
        _last_stream_message = spoken_text
    else:
        _last_stream_message = None

    return _build_response(spoken_text, audio_type="speech")


__all__ = [
    "main",
    "analyze_clothing_match",
    "build_match_prompt",
    "format_for_audio",
    "resize_image_if_needed",
    "convert_cv2_to_pil",
]
