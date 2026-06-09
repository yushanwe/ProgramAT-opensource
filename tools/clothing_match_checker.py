"""
Clothing Match Checker Tool

Checks whether visible shirt and pants/outfit pieces match well for quick feedback.
"""

import json
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from litellm_utils import extract_text, pil_image_to_data_uri, resolve_api_key, resolve_model_name

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False


_last_stream_feedback = None


def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    """Resize image while keeping aspect ratio."""
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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def limit_words(text: str, max_words: int = 15) -> str:
    """Limit output length for streaming speech."""
    clean_text = " ".join((text or "").replace("\n", " ").split()).strip()
    if not clean_text:
        return ""

    words = clean_text.split()
    if len(words) <= max_words:
        return clean_text

    shortened = " ".join(words[:max_words]).rstrip(".,;:")
    return shortened + "."


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output."""
    candidate = (text or "").strip()
    if not candidate:
        return {}

    try:
        return json.loads(candidate)
    except Exception:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start:end + 1])
        except Exception:
            return {}
    return {}


def build_match_prompt(is_streaming: bool = False) -> str:
    """Build prompt for outfit match evaluation."""
    length_instruction = (
        "Keep response under 15 words when writing spoken feedback."
        if is_streaming
        else "Spoken feedback can be one concise sentence."
    )
    return (
        "You are helping a blind user decide if clothing matches.\n"
        "Analyze visible upper-body clothing and lower-body clothing on the same person.\n"
        "If shirt and pants are not both visible, return unknown.\n"
        "Evaluate color harmony, pattern clash, and style compatibility.\n"
        f"{length_instruction}\n"
        "Return ONLY valid JSON with keys:\n"
        "shirt, pants, match, reason, spoken_feedback\n"
        "Where match is one of: good, okay, poor, unknown."
    )


def analyze_clothing_match(
    image: np.ndarray,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3-flash-preview",
    is_streaming: bool = False,
) -> Dict[str, Any]:
    """Analyze whether visible clothing pieces match."""
    if not LITELLM_AVAILABLE:
        return {
            "success": False,
            "match": "unknown",
            "spoken_feedback": "I cannot check clothing match right now.",
            "reason": "litellm not available",
        }

    api_key = resolve_api_key(model_name, api_key)
    if not api_key:
        return {
            "success": False,
            "match": "unknown",
            "spoken_feedback": "I cannot check clothing match because API key is missing.",
            "reason": "API key not configured",
        }

    try:
        processed_image = resize_image_if_needed(image)
        pil_image = convert_cv2_to_pil(processed_image)
        image_data_uri = pil_image_to_data_uri(pil_image)
        prompt = build_match_prompt(is_streaming=is_streaming)

        resolved_model = resolve_model_name(model_name)
        response = litellm.completion(
            model=resolved_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            api_key=api_key,
        )

        raw_text = extract_text(response)
        parsed = _extract_json_object(raw_text)

        if not parsed:
            return {
                "success": False,
                "match": "unknown",
                "spoken_feedback": "I could not confidently judge your outfit match.",
                "reason": "Invalid model response",
            }

        match_value = str(parsed.get("match", "unknown")).lower().strip()
        if match_value not in {"good", "okay", "poor", "unknown"}:
            match_value = "unknown"

        spoken = str(parsed.get("spoken_feedback", "")).strip()
        if not spoken:
            if match_value == "good":
                spoken = "Your shirt and pants match well."
            elif match_value == "okay":
                spoken = "Your outfit mostly matches."
            elif match_value == "poor":
                spoken = "Your shirt and pants do not match well."
            else:
                spoken = "I need a clearer view of your shirt and pants."

        return {
            "success": True,
            "shirt": str(parsed.get("shirt", "")).strip(),
            "pants": str(parsed.get("pants", "")).strip(),
            "match": match_value,
            "reason": str(parsed.get("reason", "")).strip(),
            "spoken_feedback": spoken,
        }
    except Exception as exc:
        return {
            "success": False,
            "match": "unknown",
            "spoken_feedback": "Unable to analyze clothing. Please adjust camera angle or try again.",
            "reason": f"analysis error: {type(exc).__name__}",
        }


def _format_response_text(result: Dict[str, Any], is_streaming: bool) -> str:
    """Format final user-facing feedback."""
    spoken = str(result.get("spoken_feedback", "")).strip()
    if not spoken:
        spoken = "I could not check clothing match right now."

    if is_streaming:
        return limit_words(spoken, max_words=15)
    return spoken


def main(image: np.ndarray, input_data: Optional[Dict] = None):
    """Tool entry point."""
    global _last_stream_feedback

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {"type": "error", "text": "Camera not ready. Please wait a moment and try again.", "rate": 1.0, "interrupt": False},
            "text": "Camera not ready. Please wait a moment and try again.",
        }

    config = input_data or {}
    is_streaming = bool(config.get("is_streaming", False))
    api_key = config.get("api_key")
    model = config.get("model", os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")))

    result = analyze_clothing_match(
        image=image,
        api_key=api_key,
        model_name=model,
        is_streaming=is_streaming,
    )

    response_text = _format_response_text(result, is_streaming=is_streaming)
    if is_streaming and response_text == _last_stream_feedback:
        return ""
    if is_streaming:
        _last_stream_feedback = response_text
    else:
        _last_stream_feedback = None

    match_value = result.get("match", "unknown")
    if not result.get("success", False):
        audio_type = "error"
    elif match_value == "poor":
        audio_type = "warning"
    elif match_value == "good":
        audio_type = "success"
    else:
        audio_type = "speech"

    return {
        "audio": {"type": audio_type, "text": response_text, "rate": 1.0, "interrupt": False},
        "text": response_text,
        "match": match_value,
        "shirt": result.get("shirt", ""),
        "pants": result.get("pants", ""),
        "reason": result.get("reason", ""),
    }
