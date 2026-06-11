"""
Uber Vehicle and Passenger Door Handle Locator

Capability Pipeline
1) object_detection: Detect likely vehicles in the frame and keep their bounding boxes.
2) visual_reasoning: Match the target Uber description to one detected vehicle and estimate passenger handle point.
3) navigation: Convert handle coordinates into short, audio-friendly guidance.

This tool is designed for blind and low-vision users. It returns concise spoken guidance.
"""

from typing import Any, Dict, List, Optional
import json

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import litellm
except ImportError:
    litellm = None

try:
    from litellm_utils import resolve_model_name, resolve_api_key, extract_text, pil_image_to_data_uri
except Exception:
    def resolve_model_name(model_name: str, default_model: str = 'gemini-3-flash-preview') -> str:
        return (model_name or default_model).strip()

    def resolve_api_key(model_name: str, explicit_api_key: str = '') -> str:
        return explicit_api_key

    def extract_text(response: Any) -> str:
        return str(response) if response is not None else ''

    def pil_image_to_data_uri(pil_image: Any) -> str:
        return ''

DEFAULT_CONFIDENCE = 0.35
STREAMING_WORD_LIMIT = 15
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


# Track recent output to avoid repetition in streaming mode
_last_stream_output = ""


def build_target_description(input_data: Dict[str, Any]) -> str:
    """Build a target-vehicle description from flexible input fields."""
    if not isinstance(input_data, dict):
        return ""

    direct = input_data.get('uber_description') or input_data.get('vehicle_description') or input_data.get('target_description')
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: List[str] = []
    for key in ('color', 'make', 'model', 'plate'):
        value = input_data.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    return " ".join(parts).strip()


def limit_streaming_words(text: str, is_streaming: bool) -> str:
    """Limit streaming text to the repository 15-word guideline."""
    if not is_streaming:
        return text

    words = text.split()
    if len(words) <= STREAMING_WORD_LIMIT:
        return text
    return " ".join(words[:STREAMING_WORD_LIMIT])


def get_clock_position(center_x: int, frame_width: int) -> int:
    """Map horizontal position to allowed clock outputs: 9-12 and 1-3."""
    if frame_width <= 0:
        return 12

    ratio = max(0.0, min(1.0, center_x / frame_width))
    if ratio < 1 / 7:
        return 9
    if ratio < 2 / 7:
        return 10
    if ratio < 3 / 7:
        return 11
    if ratio < 4 / 7:
        return 12
    if ratio < 5 / 7:
        return 1
    if ratio < 6 / 7:
        return 2
    return 3


def detect_vehicle_candidates(image: np.ndarray, confidence_threshold: float) -> List[Dict[str, Any]]:
    """Detect vehicle candidates with YOLO11 + COCO classes."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    if YOLO is None:
        return []

    model_cache = globals().get('yolo_model_cache', {})
    model = model_cache.get('yolo11n.pt')
    if model is None:
        model = YOLO('yolo11n.pt')
        model_cache['yolo11n.pt'] = model

    height, width = image.shape[:2]
    detections: List[Dict[str, Any]] = []

    results = model(image, conf=confidence_threshold, verbose=False)
    for result in results:
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x = max(0, int(x1))
            y = max(0, int(y1))
            w = max(1, int(x2 - x1))
            h = max(1, int(y2 - y1))
            cx = x + w // 2
            cy = y + h // 2

            detections.append(
                {
                    'bbox': [x, y, w, h],
                    'center': [cx, cy],
                    'confidence': float(box.conf[0]),
                    'frame_size': [width, height],
                }
            )

    detections.sort(key=lambda d: d['bbox'][2] * d['bbox'][3], reverse=True)
    return detections


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from free-form model text."""
    if not text:
        return None

    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = text[start:end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _estimate_handle_from_bbox(bbox: List[int]) -> Dict[str, int]:
    """Fallback estimate for passenger handle point from a vehicle bbox."""
    x, y, w, h = bbox
    return {
        'handle_x': int(x + (0.78 * w)),
        'handle_y': int(y + (0.56 * h)),
    }


def reason_about_vehicle_and_handle(
    image: np.ndarray,
    candidates: List[Dict[str, Any]],
    target_description: str,
    model_name: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Use multimodal reasoning to pick the matching Uber and locate passenger handle.
    Falls back to largest vehicle and geometric estimate when model path is unavailable.
    """
    if not candidates:
        return {'selected_index': -1, 'reason': 'no_vehicle'}

    if litellm is None or Image is None:
        estimate = _estimate_handle_from_bbox(candidates[0]['bbox'])
        return {'selected_index': 0, **estimate, 'confidence': 0.4}

    try:
        rgb = image[:, :, ::-1] if len(image.shape) == 3 else image
        pil_image = Image.fromarray(rgb)
        image_data_uri = pil_image_to_data_uri(pil_image)

        candidate_lines = []
        for idx, candidate in enumerate(candidates):
            x, y, w, h = candidate['bbox']
            candidate_lines.append(f"{idx}: bbox=[{x},{y},{w},{h}]")

        prompt = (
            "You help a blind user find their Uber and the passenger door handle. "
            "Select one vehicle candidate and estimate passenger door handle center in absolute image pixels. "
            "Return JSON only with keys: selected_index, handle_x, handle_y, vehicle_summary, confidence. "
            f"Target Uber description: {target_description or 'not provided'}. "
            "Candidates: " + " ; ".join(candidate_lines)
        )

        resolved_model = resolve_model_name(model_name)
        resolved_api_key = resolve_api_key(resolved_model, api_key)

        if not resolved_api_key:
            estimate = _estimate_handle_from_bbox(candidates[0]['bbox'])
            return {'selected_index': 0, **estimate, 'confidence': 0.4}

        response = litellm.completion(
            model=resolved_model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_data_uri}},
                    ],
                }
            ],
            api_key=resolved_api_key,
        )

        parsed = _extract_json(extract_text(response) if response else '') or {}
        selected_index = int(parsed.get('selected_index', 0)) if candidates else -1
        selected_index = max(0, min(selected_index, len(candidates) - 1))

        selected_bbox = candidates[selected_index]['bbox']
        fallback = _estimate_handle_from_bbox(selected_bbox)

        handle_x = int(parsed.get('handle_x', fallback['handle_x']))
        handle_y = int(parsed.get('handle_y', fallback['handle_y']))

        return {
            'selected_index': selected_index,
            'handle_x': handle_x,
            'handle_y': handle_y,
            'vehicle_summary': str(parsed.get('vehicle_summary', '')).strip(),
            'confidence': float(parsed.get('confidence', 0.6)),
        }
    except Exception:
        estimate = _estimate_handle_from_bbox(candidates[0]['bbox'])
        return {'selected_index': 0, **estimate, 'confidence': 0.4}


def build_navigation_response(
    selected_vehicle: Dict[str, Any],
    reasoning: Dict[str, Any],
    frame_width: int,
    is_streaming: bool,
    target_description: str,
) -> str:
    """Create concise audio-first guidance."""
    handle_x = int(reasoning.get('handle_x', selected_vehicle['center'][0]))
    clock = get_clock_position(handle_x, frame_width)

    center_offset = (handle_x / frame_width) - 0.5 if frame_width > 0 else 0
    if center_offset > 0.12:
        direction = 'right of center'
    elif center_offset < -0.12:
        direction = 'left of center'
    else:
        direction = 'near center'

    summary = reasoning.get('vehicle_summary', '').strip()
    if not summary:
        summary = target_description if target_description else 'detected vehicle'

    message = f"Your Uber is {summary} at {clock} o'clock. Passenger handle is {direction}."
    return limit_streaming_words(message, is_streaming)


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """
    Main entry point.

    input_data supports:
    - uber_description / vehicle_description / target_description
    - color, make, model, plate
    - confidence
    - is_streaming
    - model
    - api_key
    """
    global _last_stream_output

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {'type': 'error', 'text': 'No camera image available'},
            'text': 'No camera image available'
        }

    config = input_data if isinstance(input_data, dict) else {}
    is_streaming = bool(config.get('is_streaming', False))
    confidence = float(config.get('confidence', DEFAULT_CONFIDENCE))
    model_name = config.get('model', 'gemini-3-flash-preview')
    api_key = config.get('api_key', '')

    target_description = build_target_description(config)

    detections = detect_vehicle_candidates(image, confidence)
    if not detections:
        message = limit_streaming_words('No vehicle found yet. Slowly pan left and right.', is_streaming)
        if is_streaming and message == _last_stream_output:
            return ""
        _last_stream_output = message
        return message

    if not target_description and len(detections) > 1:
        message = limit_streaming_words('I found multiple vehicles. Say your Uber color, make, model, or plate.', is_streaming)
        if is_streaming and message == _last_stream_output:
            return ""
        _last_stream_output = message
        return message

    height, width = image.shape[:2]
    reasoning = reason_about_vehicle_and_handle(image, detections, target_description, model_name, api_key)

    selected_index = int(reasoning.get('selected_index', 0))
    selected_index = max(0, min(selected_index, len(detections) - 1))
    selected_vehicle = detections[selected_index]

    message = build_navigation_response(selected_vehicle, reasoning, width, is_streaming, target_description)

    if is_streaming and message == _last_stream_output:
        return ""

    _last_stream_output = message

    near_center = abs((int(reasoning.get('handle_x', selected_vehicle['center'][0])) / max(width, 1)) - 0.5) < 0.08
    if near_center:
        return {
            'audio': {
                'type': 'success',
                'text': message,
                'rate': 1.0,
                'interrupt': False,
            },
            'text': message,
        }

    return message


__all__ = [
    'main',
    'build_target_description',
    'limit_streaming_words',
    'get_clock_position',
    'detect_vehicle_candidates',
    'reason_about_vehicle_and_handle',
    'build_navigation_response',
]
