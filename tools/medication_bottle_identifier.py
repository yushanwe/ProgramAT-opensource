"""
Medication Bottle Identifier Tool

Identifies medication bottles from camera frames using YOLO11 (COCO bottle class)
and Google Cloud Vision OCR for label text. It can also verify whether the bottle
matches an expected medication name.
"""

import os
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Constants
BOTTLE_CLASS_ID = 39  # COCO: bottle
DEFAULT_DETECTION_CONFIDENCE = 0.35
STREAMING_WORD_LIMIT = 15
MAX_HISTORY_SIZE = 4

# Global state for streaming deduplication
_recent_signatures = deque(maxlen=MAX_HISTORY_SIZE)

# Vision API client cache
_vision_client = None
_vision_client_key = None

try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False


def _normalize_text(text: str) -> str:
    """Normalize text for robust matching and deduplication."""
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _limit_words(text: str, limit: int = STREAMING_WORD_LIMIT) -> str:
    """Limit text to a maximum word count for streaming mode."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def reset_tracking() -> None:
    """Reset per-session streaming memory."""
    _recent_signatures.clear()


def detect_bottle_presence(
    image: np.ndarray,
    confidence_threshold: float = DEFAULT_DETECTION_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """Detect bottle objects using YOLO11 COCO model."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    detections: List[Dict[str, Any]] = []

    try:
        from ultralytics import YOLO

        model_cache = globals().get("yolo_model_cache", {})
        cache_key = "medication_bottle_identifier_yolo11n"

        if cache_key not in model_cache:
            model_cache[cache_key] = YOLO("yolo11n.pt")

        model = model_cache[cache_key]
        results = model(image, conf=confidence_threshold, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id != BOTTLE_CLASS_ID:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    {
                        "class_name": "bottle",
                        "confidence": confidence,
                        "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    }
                )

    except Exception:
        # Non-fatal: OCR can still identify medication text without bottle detection.
        return []

    return detections


def _get_vision_client(credentials_input: str):
    """Get cached Google Vision client for provided credentials input."""
    global _vision_client, _vision_client_key

    if _vision_client and _vision_client_key == credentials_input:
        return _vision_client

    from google.oauth2 import service_account
    import json

    if os.path.isfile(credentials_input):
        credentials = service_account.Credentials.from_service_account_file(credentials_input)
        client = vision.ImageAnnotatorClient(credentials=credentials)
    elif credentials_input.startswith("{"):
        credentials_dict = json.loads(credentials_input)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = vision.ImageAnnotatorClient(credentials=credentials)
    else:
        client = vision.ImageAnnotatorClient()

    _vision_client = client
    _vision_client_key = credentials_input
    return client


def extract_text_google_vision(
    image: np.ndarray,
    credentials_input: Optional[str] = None,
    language_hints: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Extract full text from image using Google Cloud Vision."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "", "No camera image available"

    if not VISION_API_AVAILABLE:
        return "", "Google Cloud Vision is unavailable"

    if language_hints is None:
        language_hints = ["en"]

    if not credentials_input:
        credentials_input = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    if not credentials_input:
        return "", "Google Cloud Vision credentials are missing"

    try:
        client = _get_vision_client(credentials_input)

        success, encoded_image = cv2.imencode(".jpg", image)
        if not success:
            return "", "Could not encode image for OCR"

        response = client.text_detection(
            image=vision.Image(content=encoded_image.tobytes()),
            image_context=vision.ImageContext(language_hints=language_hints),
        )

        if response.error.message:
            return "", response.error.message

        annotations = response.text_annotations
        if not annotations:
            return "", None

        return annotations[0].description.strip(), None

    except Exception as exc:
        return "", f"OCR failed: {str(exc)}"


def extract_medication_name(ocr_text: str) -> str:
    """Extract the most likely medication name line from OCR text."""
    if not ocr_text:
        return ""

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        return ""

    stop_words = {
        "rx", "refill", "qty", "quantity", "doctor", "dr", "pharmacy",
        "take", "tablet", "tablets", "capsule", "capsules", "daily",
        "warning", "instructions", "prescriber", "patient", "name", "dob",
        "expires", "exp", "lot", "manufacturer", "generic", "use",
    }

    dosage_pattern = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu)\b", re.IGNORECASE)

    scored_candidates: List[Tuple[int, str]] = []

    for line in lines:
        normalized = _normalize_text(line)
        if len(normalized) < 3:
            continue

        words = normalized.split()
        if not words:
            continue

        content_words = [w for w in words if w not in stop_words]
        if not content_words:
            continue

        score = len(content_words)
        if dosage_pattern.search(line):
            score += 8
        if any(ch.isalpha() for ch in line):
            score += 2
        if len(line) <= 45:
            score += 1

        scored_candidates.append((score, line))

    if not scored_candidates:
        return ""

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return scored_candidates[0][1]


def is_expected_medication(found_medication: str, expected_medication: str) -> bool:
    """Compare found medication against user-provided expected name."""
    found_norm = _normalize_text(found_medication)
    expected_norm = _normalize_text(expected_medication)

    if not found_norm or not expected_norm:
        return False

    if expected_norm in found_norm or found_norm in expected_norm:
        return True

    found_tokens = set(found_norm.split())
    expected_tokens = set(expected_norm.split())
    if not found_tokens or not expected_tokens:
        return False

    overlap_ratio = len(found_tokens & expected_tokens) / len(expected_tokens)
    return overlap_ratio >= 0.6


def _build_response_text(
    found_medication: str,
    expected_medication: str,
    bottle_detected: bool,
) -> Tuple[str, Optional[str]]:
    """Create user-facing response text and optional severity."""
    if found_medication and expected_medication:
        if is_expected_medication(found_medication, expected_medication):
            return f"Yes, this is {found_medication}.", None
        return (
            f"Warning: This says {found_medication}, not {expected_medication}.",
            "warning",
        )

    if found_medication:
        return f"This is {found_medication}.", None

    if bottle_detected:
        return "Bottle detected. Rotate label toward camera for medication name.", None

    return "No medication bottle detected. Point camera at the bottle label.", None


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Identify medication bottles and verify against expected medication.

    input_data options:
      - expected_medication / target_medication / medication_name: target label text
      - confidence: bottle detection threshold (default 0.35)
      - track_mode: suppress repeated streaming responses (default True)
      - is_streaming: if True, cap spoken responses to 15 words
      - reset: clear tracking memory
      - language: OCR language hint (default en)
      - api_key: optional credentials override for Vision API
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return "No camera image available"

    config = input_data if isinstance(input_data, dict) else {}

    if config.get("reset", False):
        reset_tracking()
        return "Medication tracking reset"

    expected_medication = (
        config.get("expected_medication")
        or config.get("target_medication")
        or config.get("medication_name")
        or ""
    )
    confidence = float(config.get("confidence", DEFAULT_DETECTION_CONFIDENCE))
    track_mode = bool(config.get("track_mode", True))
    is_streaming = bool(config.get("is_streaming", False))
    language = config.get("language", "en")
    credentials_input = config.get("api_key") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    bottle_detections = detect_bottle_presence(image, confidence_threshold=confidence)
    bottle_detected = len(bottle_detections) > 0

    ocr_text, ocr_error = extract_text_google_vision(
        image,
        credentials_input=credentials_input,
        language_hints=[language],
    )

    if ocr_error:
        # Still return bottle guidance if available, otherwise return an actionable error.
        if bottle_detected:
            response_text = "Bottle detected, but OCR unavailable. Check Vision API credentials."
        else:
            response_text = "Cannot read label. Check Vision API credentials and label visibility."

        if is_streaming:
            response_text = _limit_words(response_text)

        signature = f"error|{_normalize_text(response_text)}"
        if track_mode and signature in _recent_signatures:
            return ""
        if track_mode:
            _recent_signatures.append(signature)

        return {
            "audio": {"type": "error", "text": response_text, "interrupt": True},
            "text": response_text,
            "error": ocr_error,
        }

    found_medication = extract_medication_name(ocr_text)
    response_text, severity = _build_response_text(found_medication, expected_medication, bottle_detected)

    if is_streaming:
        response_text = _limit_words(response_text)

    signature = "|".join(
        [
            _normalize_text(found_medication),
            _normalize_text(expected_medication),
            str(bottle_detected),
            severity or "info",
            _normalize_text(response_text),
        ]
    )

    if track_mode and signature in _recent_signatures:
        return ""

    if track_mode:
        _recent_signatures.append(signature)

    if severity == "warning":
        return {
            "audio": {
                "type": "warning",
                "text": response_text,
                "rate": 1.1,
                "interrupt": True,
            },
            "text": response_text,
        }

    return response_text
