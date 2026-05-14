"""
Currency Bill Identifier Tool

Identifies U.S. dollar bill denominations from camera images using OCR text.
Designed for blind and low-vision users with concise, audio-friendly output.

Interface:
- Entry point: main(image, input_data)
- Input: OpenCV image (numpy array), optional config dict
- Output: audio-friendly string or dict with audio/text keys
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import os
import re

# Try to import Google Cloud Vision
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False

# Cache Vision client for performance in streaming mode
_vision_client = None
_vision_client_key = None
PATTERN_TEXT_WEIGHT = 5
PATTERN_NUMERIC_WEIGHT = 2
LABEL_MATCH_BONUS = 4
MIN_DENOMINATION_SCORE = 4
# Scores without explicit currency context are capped below MIN_DENOMINATION_SCORE
# to avoid false positives from years/serial numbers.
MAX_SCORE_WITHOUT_CONTEXT = 3

# U.S. denomination labels and OCR patterns
USD_DENOMINATIONS = {
    1: {
        'label': '1 dollar',
        'patterns': [
            r'\bONE\s+DOLLAR\b',
            r'\bONE\b',
            r'\$\s*1\b',
            r'\b1\b',
        ],
    },
    2: {
        'label': '2 dollars',
        'patterns': [
            r'\bTWO\s+DOLLARS?\b',
            r'\bTWO\b',
            r'\$\s*2\b',
            r'\b2\b',
        ],
    },
    5: {
        'label': '5 dollars',
        'patterns': [
            r'\bFIVE\s+DOLLARS?\b',
            r'\bFIVE\b',
            r'\$\s*5\b',
            r'\b5\b',
        ],
    },
    10: {
        'label': '10 dollars',
        'patterns': [
            r'\bTEN\s+DOLLARS?\b',
            r'\bTEN\b',
            r'\$\s*10\b',
            r'\b10\b',
        ],
    },
    20: {
        'label': '20 dollars',
        'patterns': [
            r'\bTWENTY\s+DOLLARS?\b',
            r'\bTWENTY\b',
            r'\$\s*20\b',
            r'\b20\b',
        ],
    },
    50: {
        'label': '50 dollars',
        'patterns': [
            r'\bFIFTY\s+DOLLARS?\b',
            r'\bFIFTY\b',
            r'\$\s*50\b',
            r'\b50\b',
        ],
    },
    100: {
        'label': '100 dollars',
        'patterns': [
            r'\bONE\s+HUNDRED\s+DOLLARS?\b',
            r'\bHUNDRED\s+DOLLARS?\b',
            r'\bONE\s+HUNDRED\b',
            r'\$\s*100\b',
            r'\b100\b',
        ],
    },
}


# Building block: image preprocessing

def preprocess_image_for_ocr(image: np.ndarray, max_size: Tuple[int, int] = (1280, 1280)) -> np.ndarray:
    """Resize and lightly enhance image for OCR stability."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image

    h, w = image.shape[:2]
    max_w, max_h = max_size

    # Resize if needed
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Improve OCR contrast while preserving color information
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.equalizeHist(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# Building block: Vision client creation

def _looks_like_service_account_json(raw_json: str) -> bool:
    """Light validation for service-account style JSON credentials."""
    try:
        import json
        parsed = json.loads(raw_json)
        required_keys = {'type', 'client_email', 'private_key'}
        return isinstance(parsed, dict) and required_keys.issubset(parsed.keys())
    except Exception:
        return False


def get_vision_client(api_key: Optional[str] = None):
    """Create or reuse a Google Cloud Vision client."""
    global _vision_client, _vision_client_key

    if not VISION_API_AVAILABLE:
        return None

    if api_key is None:
        api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')

    if _vision_client is not None and _vision_client_key == api_key:
        return _vision_client

    if api_key and os.path.isfile(api_key):
        credentials = service_account.Credentials.from_service_account_file(api_key)
        client = vision.ImageAnnotatorClient(credentials=credentials)
    elif api_key and api_key.startswith('{') and _looks_like_service_account_json(api_key):
        import json
        creds = json.loads(api_key)
        credentials = service_account.Credentials.from_service_account_info(creds)
        client = vision.ImageAnnotatorClient(credentials=credentials)
    else:
        # Uses default credentials chain
        client = vision.ImageAnnotatorClient()

    _vision_client = client
    _vision_client_key = api_key
    return client


# Building block: OCR

def detect_text_google_vision(image: np.ndarray, api_key: Optional[str] = None) -> List[str]:
    """Extract OCR text lines from an image using Google Cloud Vision."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    if not VISION_API_AVAILABLE:
        return []

    try:
        client = get_vision_client(api_key)
        if client is None:
            return []

        ok, encoded = cv2.imencode('.jpg', image)
        if not ok:
            return []

        vision_image = vision.Image(content=encoded.tobytes())
        response = client.text_detection(image=vision_image)

        if response.error.message:
            return []

        annotations = response.text_annotations
        if not annotations:
            return []

        # First annotation is full text block; keep both full and token-level lines
        texts = [a.description for a in annotations if a.description and a.description.strip()]
        return texts

    except Exception:
        return []


# Building block: denomination scoring

def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text to uppercase and clean punctuation/noise."""
    if not text:
        return ''
    cleaned = text.upper()
    cleaned = cleaned.replace('\n', ' ')
    cleaned = re.sub(r'[^A-Z0-9$ ]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def identify_us_bill_denomination(ocr_texts: List[str]) -> Tuple[Optional[int], Optional[str], int]:
    """
    Identify U.S. bill denomination from OCR text.

    Returns:
        (value, label, score)
    """
    if not ocr_texts:
        return None, None, 0

    normalized = normalize_ocr_text(' '.join(ocr_texts))
    if not normalized:
        return None, None, 0

    # Prioritize explicit currency context to reduce false positives from serial numbers/years
    currency_context = bool(re.search(r'\b(DOLLAR|DOLLARS|FEDERAL\s+RESERVE\s+NOTE)\b', normalized))

    best_value = None
    best_label = None
    best_score = 0

    for value, data in USD_DENOMINATIONS.items():
        score = 0
        for idx, pattern in enumerate(data['patterns']):
            if re.search(pattern, normalized):
                # Heavier weight for text phrases, lighter for bare numeric matches
                score += PATTERN_TEXT_WEIGHT if idx <= 1 else PATTERN_NUMERIC_WEIGHT

        # Extra weight if denomination includes explicit "X dollars" phrase
        word_label = data['label'].upper()
        if re.search(re.escape(word_label), normalized):
            score += LABEL_MATCH_BONUS

        # Bare numeric matches are weak evidence when currency context is absent
        if not currency_context:
            score = min(score, MAX_SCORE_WITHOUT_CONTEXT)

        if score > best_score:
            best_score = score
            best_value = value
            best_label = data['label']

    # Require minimum confidence to avoid random number misreads
    if best_score < MIN_DENOMINATION_SCORE:
        return None, None, best_score

    return best_value, best_label, best_score


# Entry point

def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Any:
    """
    Identify U.S. cash denomination from camera image.

    Optional input_data keys:
    - api_key: Optional Google credentials override (path or JSON string)
    - detailed: bool, return dict with score/details if True
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {
                'type': 'error',
                'text': 'No camera image available for bill identification.'
            },
            'text': 'No camera image available for bill identification.'
        }

    if input_data is None or not isinstance(input_data, dict):
        input_data = {}

    api_key = input_data.get('api_key')
    detailed = bool(input_data.get('detailed', False))

    if not VISION_API_AVAILABLE:
        msg = 'Currency recognition needs Google Cloud Vision. Please install dependencies.'
        return {
            'audio': {'type': 'error', 'text': msg},
            'text': msg
        }

    processed = preprocess_image_for_ocr(image)
    ocr_texts = detect_text_google_vision(processed, api_key=api_key)

    if not ocr_texts:
        return 'No bill text detected. Move closer and improve lighting.'

    value, label, score = identify_us_bill_denomination(ocr_texts)

    if label is None:
        return 'I could not identify the bill denomination. Try a clearer view of the bill front.'

    spoken = f'This appears to be {label}.'

    if not detailed:
        return spoken

    return {
        'audio': {
            'type': 'speech',
            'text': spoken,
            'rate': 1.0,
            'interrupt': True
        },
        'text': spoken,
        'denomination': value,
        'confidence_score': score
    }
