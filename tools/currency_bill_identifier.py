"""
Currency Bill Identifier Tool

Identifies cash bill denominations from camera images using Google Cloud Vision OCR.
Designed for audio-first usage in ProgramAT.
"""

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Supported denominations for U.S. paper bills
SUPPORTED_DENOMINATIONS = [1, 2, 5, 10, 20, 50, 100]

WORD_TO_DENOMINATION = {
    'ONE': 1,
    'TWO': 2,
    'FIVE': 5,
    'TEN': 10,
    'TWENTY': 20,
    'FIFTY': 50,
    'HUNDRED': 100,
    'ONE HUNDRED': 100,
}

USD_CONTEXT_MARKERS = [
    'DOLLAR',
    'DOLLARS',
    'UNITED STATES',
    'FEDERAL RESERVE',
    'IN GOD WE TRUST',
]

_last_announced_denomination: Optional[int] = None

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False


_vision_client = None
_vision_client_key = None


def get_vision_client(api_key: Optional[str] = None):
    """Get or create a cached Google Cloud Vision client."""
    if not VISION_AVAILABLE:
        return None

    global _vision_client, _vision_client_key

    if api_key is None:
        api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')

    cache_key = api_key or '__default__'
    if _vision_client is not None and _vision_client_key == cache_key:
        return _vision_client

    if api_key and os.path.isfile(api_key):
        credentials = service_account.Credentials.from_service_account_file(api_key)
        client = vision.ImageAnnotatorClient(credentials=credentials)
    elif api_key and api_key.strip().startswith('{'):
        import json
        credentials_info = json.loads(api_key)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        client = vision.ImageAnnotatorClient(credentials=credentials)
    else:
        client = vision.ImageAnnotatorClient()

    _vision_client = client
    _vision_client_key = cache_key
    return client


def detect_text_google_vision(
    image: np.ndarray,
    api_key: Optional[str] = None,
    language_hints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect text using Google Cloud Vision OCR and return the full detected text."""
    if not VISION_AVAILABLE:
        return {'error': 'Google Cloud Vision is not available on this server.'}

    try:
        client = get_vision_client(api_key)
        if client is None:
            return {'error': 'Google Cloud Vision is not available on this server.'}

        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            return {'text': ''}

        vision_image = vision.Image(content=encoded_image.tobytes())
        context = vision.ImageContext(language_hints=language_hints or ['en'])
        response = client.text_detection(image=vision_image, image_context=context)

        if response.error.message:
            return {'error': response.error.message}

        annotations = response.text_annotations
        if not annotations:
            return {'text': ''}

        return {'text': annotations[0].description}
    except Exception as exc:
        return {'error': f'OCR request failed: {exc}'}


def normalize_text(text: str) -> str:
    """Normalize OCR text for matching denomination patterns."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9$\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_denomination_from_text(text: str) -> Optional[int]:
    """Extract likely U.S. bill denomination from OCR text."""
    if not text:
        return None

    normalized = normalize_text(text)
    if not normalized:
        return None

    candidates: List[int] = []

    # Numeric mentions with optional dollar marker
    numeric_pattern = re.findall(r'\$?\s*(1|2|5|10|20|50|100)\b', normalized)
    for value in numeric_pattern:
        candidates.append(int(value))

    # Word-based mentions
    if 'ONE HUNDRED' in normalized:
        candidates.append(100)

    for word, denomination in WORD_TO_DENOMINATION.items():
        if word == 'ONE HUNDRED':
            continue
        matches = re.findall(rf'\b{re.escape(word)}\b', normalized)
        if matches:
            candidates.extend([denomination] * len(matches))

    if not candidates:
        return None

    marker_found = any(marker in normalized for marker in USD_CONTEXT_MARKERS)
    if not marker_found:
        # If there is no currency context, require a strong signal (same denom repeated)
        counts = Counter(candidates)
        top_denom, top_count = counts.most_common(1)[0]
        return top_denom if top_count >= 2 else None

    counts = Counter(candidates)
    top_count = max(counts.values())
    top_denominations = [den for den, count in counts.items() if count == top_count]

    # Prefer lower denomination when tied to reduce risk of over-reporting value
    return min(top_denominations)


def format_denomination_audio(denomination: int) -> str:
    """Create an audio-friendly denomination phrase."""
    return '1 dollar' if denomination == 1 else f'{denomination} dollars'


def identify_currency_bill(image: np.ndarray, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Run OCR and identify the bill denomination if present."""
    ocr_result = detect_text_google_vision(image, api_key=api_key, language_hints=['en'])
    if 'error' in ocr_result:
        return ocr_result

    detected_text = ocr_result.get('text', '')
    denomination = extract_denomination_from_text(detected_text)

    return {
        'denomination': denomination,
        'detected_text': detected_text,
    }


def reset_tracking() -> None:
    """Reset denomination tracking for streaming mode."""
    global _last_announced_denomination
    _last_announced_denomination = None


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """Main entry point for ProgramAT tool runner."""
    global _last_announced_denomination

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return 'No camera image available'

    config = input_data if isinstance(input_data, dict) else {}

    if config.get('reset', False):
        reset_tracking()
        return 'Currency tracking reset'

    track_mode = bool(config.get('track_mode', False))
    api_key = config.get('api_key')

    result = identify_currency_bill(image, api_key=api_key)

    if 'error' in result:
        return {
            'audio': {
                'type': 'error',
                'text': result['error'],
                'interrupt': True,
            },
            'text': result['error'],
        }

    denomination = result.get('denomination')
    if denomination is None:
        if track_mode:
            _last_announced_denomination = None
        return 'No currency bill detected'

    if track_mode and denomination == _last_announced_denomination:
        return ''

    _last_announced_denomination = denomination
    audio_text = format_denomination_audio(denomination)

    return {
        'audio': {
            'type': 'speech',
            'text': audio_text,
            'interrupt': True,
        },
        'text': audio_text,
    }


__all__ = [
    'main',
    'identify_currency_bill',
    'detect_text_google_vision',
    'extract_denomination_from_text',
    'format_denomination_audio',
    'reset_tracking',
]
