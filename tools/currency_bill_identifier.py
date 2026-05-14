"""
Currency Bill Identifier Tool

Identifies visible paper currency denominations from camera frames using OCR.
Primary support is for U.S. dollar bills.
"""

import io
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False


STREAMING_WORD_LIMIT = 15

USD_DENOMINATION_PATTERNS = {
    1: [r"\bone\b", r"\bwashington\b", r"\b1\s*dollars?\b", r"\$\s*1\b"],
    2: [r"\btwo\b", r"\bjefferson\b", r"\b2\s*dollars?\b", r"\$\s*2\b"],
    5: [r"\bfive\b", r"\blincoln\b", r"\b5\s*dollars?\b", r"\$\s*5\b"],
    10: [r"\bten\b", r"\bhamilton\b", r"\b10\s*dollars?\b", r"\$\s*10\b"],
    20: [r"\btwenty\b", r"\bjackson\b", r"\b20\s*dollars?\b", r"\$\s*20\b"],
    50: [r"\bfifty\b", r"\bgrant\b", r"\b50\s*dollars?\b", r"\$\s*50\b"],
    100: [r"\bhundred\b", r"\bbenjamin\b", r"\bfranklin\b", r"\b100\s*dollars?\b", r"\$\s*100\b"],
}

USD_CONTEXT_PATTERNS = [
    r"\bdollar\b",
    r"\bdollars\b",
    r"\bfederal reserve note\b",
    r"\bunited states\b",
]


def convert_image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Convert OpenCV-style numpy image to JPEG bytes for OCR."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return b""

    if len(image.shape) == 2:
        pil_image = Image.fromarray(image).convert("L")
    elif len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image).convert("RGB")
    elif len(image.shape) == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(image[:, :, :3][:, :, ::-1]).convert("RGB")
    else:
        return b""

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def detect_text_google_vision(
    image: np.ndarray,
    credentials_hint: Optional[str] = None,
    language_hints: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Extract text blocks from image using Google Cloud Vision OCR."""
    if not VISION_API_AVAILABLE:
        return [{'error': 'Google Cloud Vision is not available on this server.'}]

    if language_hints is None:
        language_hints = ['en']

    credentials_hint = credentials_hint or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if not credentials_hint and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        return [{'error': 'GOOGLE_APPLICATION_CREDENTIALS is not configured for OCR.'}]

    content = convert_image_to_jpeg_bytes(image)
    if not content:
        return []

    try:
        client = vision.ImageAnnotatorClient()
        vision_image = vision.Image(content=content)
        context = vision.ImageContext(language_hints=language_hints)
        response = client.text_detection(image=vision_image, image_context=context)

        if response.error.message:
            return [{'error': response.error.message}]

        if not response.text_annotations:
            return []

        return [{'text': annotation.description} for annotation in response.text_annotations]
    except Exception as exc:
        return [{'error': f'OCR failed: {exc}'}]


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text to improve denomination matching."""
    normalized = text.lower()
    normalized = normalized.replace('\n', ' ')
    normalized = re.sub(r'[^a-z0-9$ ]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def identify_usd_denomination(text: str) -> Optional[int]:
    """Identify a likely U.S. bill denomination from OCR text."""
    normalized = normalize_ocr_text(text)
    if not normalized:
        return None

    has_usd_context = any(re.search(pattern, normalized) for pattern in USD_CONTEXT_PATTERNS)
    scores: Dict[int, int] = {}

    for denomination, patterns in USD_DENOMINATION_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, normalized):
                score += 1
        if score:
            scores[denomination] = score

    if not scores:
        return None

    best_denomination, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if has_usd_context or best_score >= 2:
        return best_denomination
    return None


def limit_words(text: str, max_words: int = STREAMING_WORD_LIMIT) -> str:
    """Trim text to a max word count for streaming mode."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = ' '.join(words[:max_words]).rstrip()
    if truncated.endswith(('.', '!', '?')):
        return truncated
    return truncated.rstrip('.,;:') + '.'


def apply_streaming_limit(text: str, is_streaming: bool) -> str:
    """Apply streaming word limit only when streaming mode is enabled."""
    return limit_words(text) if is_streaming else text


def build_response_text(denomination: int, currency_name: str = "dollars") -> str:
    """Create audio-friendly response text."""
    if denomination == 1:
        return "This looks like 1 dollar."
    return f"This looks like {denomination} {currency_name}."


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """
    Main tool entry point.
    Returns an audio-friendly bill identification result.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {'type': 'error', 'text': 'No camera image available.'},
            'text': 'No camera image available.'
        }

    config = input_data if isinstance(input_data, dict) else {}
    is_streaming = bool(config.get('is_streaming', False))
    language = config.get('language', 'en')
    credentials_hint = config.get('credentials_hint')

    detections = detect_text_google_vision(
        image=image,
        credentials_hint=credentials_hint,
        language_hints=[language]
    )

    if detections and 'error' in detections[0]:
        return {
            'audio': {'type': 'error', 'text': detections[0]['error']},
            'text': detections[0]['error']
        }

    if not detections:
        message = "I couldn't read a bill denomination. Please hold the bill steady."
        return apply_streaming_limit(message, is_streaming)

    full_text = detections[0].get('text', '')
    denomination = identify_usd_denomination(full_text)

    if denomination is None:
        message = "I couldn't determine the bill value. Try better lighting or a closer view."
        return apply_streaming_limit(message, is_streaming)

    response_text = build_response_text(denomination)
    return apply_streaming_limit(response_text, is_streaming)


__all__ = [
    'main',
    'detect_text_google_vision',
    'convert_image_to_jpeg_bytes',
    'normalize_ocr_text',
    'identify_usd_denomination',
    'limit_words',
    'apply_streaming_limit',
    'build_response_text',
]
