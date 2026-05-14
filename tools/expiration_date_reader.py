"""
Expiration Date Reader Tool for Blind Users

Reads printed expiration dates on food or medication packages using OCR
and returns concise audio-friendly results.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import os
import re
from datetime import date

# Try to import Google Cloud Vision
try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False

_MONTH_TO_NUMBER = {
    'JAN': 1, 'JANUARY': 1,
    'FEB': 2, 'FEBRUARY': 2,
    'MAR': 3, 'MARCH': 3,
    'APR': 4, 'APRIL': 4,
    'MAY': 5,
    'JUN': 6, 'JUNE': 6,
    'JUL': 7, 'JULY': 7,
    'AUG': 8, 'AUGUST': 8,
    'SEP': 9, 'SEPT': 9, 'SEPTEMBER': 9,
    'OCT': 10, 'OCTOBER': 10,
    'NOV': 11, 'NOVEMBER': 11,
    'DEC': 12, 'DECEMBER': 12,
}

_MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

_KEYWORD_PATTERNS = [
    r'\bEXP\b',
    r'\bEXPIRES?\b',
    r'\bEXPIRATION\b',
    r'\bUSE BY\b',
    r'\bBEST BY\b',
    r'\bSELL BY\b',
    r'\bBBE\b',
]


def detect_text_google_vision(
    image: np.ndarray,
    api_key: Optional[str] = None,
    language_hints: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Detect text in image using Google Cloud Vision API.

    Returns list with first item containing full text when successful.
    Returns [{'error': ...}] if OCR is unavailable.
    """
    if language_hints is None:
        language_hints = ['en']

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    if not VISION_API_AVAILABLE:
        return [{'error': 'OCR not available. Install google-cloud-vision or configure server image dependencies.'}]

    if api_key is None:
        api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')

    try:
        from google.oauth2 import service_account
        import json

        if api_key and os.path.isfile(api_key):
            credentials = service_account.Credentials.from_service_account_file(api_key)
            client = vision.ImageAnnotatorClient(credentials=credentials)
        elif api_key and api_key.strip().startswith('{'):
            credentials_dict = json.loads(api_key)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            client = vision.ImageAnnotatorClient()

        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            return []

        vision_image = vision.Image(content=encoded_image.tobytes())
        image_context = vision.ImageContext(language_hints=language_hints)
        response = client.text_detection(image=vision_image, image_context=image_context)

        if response.error.message:
            raise Exception(response.error.message)

        if not response.text_annotations:
            return []

        return [{
            'text': response.text_annotations[0].description,
            'confidence': 0.9
        }]
    except Exception as e:
        return [{'error': f'OCR failed: {str(e)}'}]


def _expand_year(year: int) -> int:
    if year >= 100:
        return year
    return 2000 + year if year <= 69 else 1900 + year


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_from_numeric(match: re.Match) -> Optional[date]:
    a = int(match.group(1))
    b = int(match.group(2))
    c = int(match.group(3))

    if len(match.group(1)) == 4:
        return _safe_date(a, b, c)

    c = _expand_year(c)

    if a > 12 and b <= 12:
        return _safe_date(c, b, a)

    return _safe_date(c, a, b)


def _extract_from_month_name(match: re.Match, day_first: bool = False) -> Optional[date]:
    if day_first:
        day = int(match.group(1))
        month_text = match.group(2).upper()
        year = _expand_year(int(match.group(3)))
    else:
        month_text = match.group(1).upper()
        day = int(match.group(2))
        year = _expand_year(int(match.group(3)))

    month = _MONTH_TO_NUMBER.get(month_text)
    if not month:
        return None

    return _safe_date(year, month, day)


def _find_dates_in_text(text: str) -> List[Tuple[date, str]]:
    if not text:
        return []

    candidates: List[Tuple[date, str]] = []

    numeric_pattern = re.compile(r'\b(\d{1,4})[\/-](\d{1,2})[\/-](\d{2,4})\b')
    month_first_pattern = re.compile(
        r'\b(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|'
        r'JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)'
        r'\s+(\d{1,2})(?:,)?\s+(\d{2,4})\b',
        re.IGNORECASE,
    )
    day_first_pattern = re.compile(
        r'\b(\d{1,2})\s+'
        r'(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|'
        r'JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)'
        r'\s+(\d{2,4})\b',
        re.IGNORECASE,
    )

    for m in numeric_pattern.finditer(text):
        parsed = _extract_from_numeric(m)
        if parsed:
            candidates.append((parsed, m.group(0)))

    for m in month_first_pattern.finditer(text):
        parsed = _extract_from_month_name(m, day_first=False)
        if parsed:
            candidates.append((parsed, m.group(0)))

    for m in day_first_pattern.finditer(text):
        parsed = _extract_from_month_name(m, day_first=True)
        if parsed:
            candidates.append((parsed, m.group(0)))

    return candidates


def extract_expiration_date(ocr_text: str) -> Optional[date]:
    """Extract the most likely expiration date from OCR text."""
    if not ocr_text or not ocr_text.strip():
        return None

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    all_candidates: List[Tuple[int, date]] = []

    for line in lines:
        upper_line = line.upper()
        line_dates = _find_dates_in_text(line)
        if not line_dates:
            continue

        has_keyword = _contains_expiration_keyword(upper_line)
        for parsed, _raw in line_dates:
            score = 10 if has_keyword else 0
            if parsed.year >= date.today().year - 1:
                score += 2
            all_candidates.append((score, parsed))

    if not all_candidates:
        text_dates = _find_dates_in_text(ocr_text)
        for parsed, _raw in text_dates:
            score = 1
            if parsed.year >= date.today().year - 1:
                score += 1
            all_candidates.append((score, parsed))

    if not all_candidates:
        return None

    all_candidates.sort(key=lambda x: (-x[0], x[1]))
    return all_candidates[0][1]


def format_expiration_date(expiration: date) -> str:
    """Format date in speech-friendly form."""
    return f"{_MONTH_NAMES[expiration.month]} {expiration.day}, {expiration.year}"


def _is_streaming_mode(config: Dict[str, Any]) -> bool:
    mode = str(config.get('mode', '')).lower()
    return bool(config.get('streaming') or config.get('is_streaming') or mode == 'stream')


def _contains_expiration_keyword(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _KEYWORD_PATTERNS)


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Main entry point.

    Args:
        image: Camera frame as numpy array
        input_data: Optional config:
          - language: OCR language hint (default: en)
          - mode / streaming / is_streaming: stream mode hint
          - api_key: optional credentials override

    Returns:
        Audio-friendly string or dict.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available."

    config = input_data if isinstance(input_data, dict) else {}
    language = config.get('language', 'en')
    api_key = config.get('api_key')

    detections = detect_text_google_vision(image, api_key=api_key, language_hints=[language])

    if detections and len(detections) == 1 and 'error' in detections[0]:
        error_msg = detections[0]['error']
        return {
            'audio': {
                'type': 'error',
                'text': error_msg,
                'interrupt': True
            },
            'text': error_msg
        }

    if not detections:
        return "No text detected on the package."

    ocr_text = detections[0].get('text', '')
    expiration = extract_expiration_date(ocr_text)
    if not expiration:
        return "No expiration date found."

    spoken_date = format_expiration_date(expiration)
    today = date.today()

    if expiration < today:
        return {
            'audio': {
                'type': 'warning',
                'text': f"Warning: expired on {spoken_date}.",
                'rate': 1.2,
                'interrupt': True
            },
            'text': f"Expired on {spoken_date}."
        }

    response = f"Expires {spoken_date}."

    if _is_streaming_mode(config):
        words = response.split()
        if len(words) > 15:
            response = ' '.join(words[:15])

    return response


# Optional alternate entry points for compatibility with future tools
run = main
process_image = main
