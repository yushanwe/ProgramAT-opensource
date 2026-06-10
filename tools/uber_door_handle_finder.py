"""
Find Uber Vehicle and Passenger Door Handle

Task Pipeline (minimal stages):
1) object_detection: Detect candidate rideshare vehicles with YOLO11 + COCO classes.
2) object_localization: Locate passenger door handle with YoloWorld (non-COCO object),
   with geometric fallback localization when handle detection is unavailable.
3) navigation: Generate concise, audio-friendly guidance with clock-face directions.

This tool runs on the backend server and receives camera frames from the mobile app.
It returns a spoken-friendly string or an audio/text dictionary.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_CONFIDENCE = 0.35
VEHICLE_CLASSES = {'car', 'truck', 'bus'}
HANDLE_CLASSES = [
    'car door handle',
    'vehicle door handle',
    'door handle',
    'passenger door handle',
]
STREAMING_WORD_LIMIT = 15

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


_last_message = None
_local_model_cache: Dict[str, Any] = {}


def get_model_cache() -> Dict[str, Any]:
    """
    Get shared backend cache when available; otherwise use local module cache.
    """
    shared_cache = globals().get('yolo_model_cache')
    if isinstance(shared_cache, dict):
        return shared_cache
    return _local_model_cache


def get_clock_position(center_x: int, center_y: int, frame_width: int, frame_height: int) -> int:
    """Map object center to visible clock-face positions (1-3 and 9-12)."""
    norm_x = center_x / max(frame_width, 1)
    norm_y = center_y / max(frame_height, 1)

    if norm_x < 0.33:
        h_region = 'left'
    elif norm_x < 0.67:
        h_region = 'center'
    else:
        h_region = 'right'

    if norm_y < 0.33:
        v_region = 'top'
    elif norm_y < 0.67:
        v_region = 'middle'
    else:
        v_region = 'bottom'

    if h_region == 'center':
        return 12
    if h_region == 'right':
        return 1 if v_region == 'top' else 2 if v_region == 'middle' else 3
    return 11 if v_region == 'top' else 10 if v_region == 'middle' else 9


def get_horizontal_hint(center_x: int, frame_width: int) -> str:
    """Return short horizontal guidance for audio output."""
    ratio = center_x / max(frame_width, 1)
    if ratio < 0.4:
        return 'left of center'
    if ratio > 0.6:
        return 'right of center'
    return 'near center'


def limit_streaming_words(text: str, limit: int = STREAMING_WORD_LIMIT) -> str:
    """Limit streaming response to a max number of words."""
    words = text.split()
    if len(words) <= limit:
        return text
    return ' '.join(words[:limit])


def estimate_vehicle_color(image: np.ndarray, bbox: List[int]) -> str:
    """Estimate dominant vehicle color from a central crop of the vehicle bounding box."""
    x, y, w, h = bbox
    x0 = max(x + int(w * 0.2), 0)
    y0 = max(y + int(h * 0.2), 0)
    x1 = min(x + int(w * 0.8), image.shape[1])
    y1 = min(y + int(h * 0.8), image.shape[0])

    if x1 <= x0 or y1 <= y0:
        return 'unknown'

    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return 'unknown'

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    if v_mean < 55:
        return 'black'
    if s_mean < 30 and v_mean > 180:
        return 'white'
    if s_mean < 40:
        return 'gray'
    if h_mean < 10 or h_mean >= 170:
        return 'red'
    if h_mean < 25:
        return 'orange'
    if h_mean < 35:
        return 'yellow'
    if h_mean < 85:
        return 'green'
    if h_mean < 130:
        return 'blue'
    return 'dark'


def detect_vehicles(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """Detect vehicle candidates with YOLO11 + COCO classes."""
    if image is None or image.size == 0:
        return []

    detections: List[Dict[str, Any]] = []

    try:
        from ultralytics import YOLO

        model_cache = get_model_cache()
        cache_key = 'uber_vehicle_finder_yolo11n'
        if cache_key not in model_cache:
            model_cache[cache_key] = YOLO('yolo11n.pt')

        yolo_model = model_cache[cache_key]
        results = yolo_model(image, conf=confidence_threshold, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                class_id = int(box.cls[0])
                class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else 'object'
                if class_name not in VEHICLE_CLASSES:
                    continue

                center_x, center_y = x + w // 2, y + h // 2
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(box.conf[0]),
                    'bbox': [x, y, w, h],
                    'center': [center_x, center_y],
                })
    except Exception:
        return []

    return detections


def score_vehicle(vehicle: Dict[str, Any], image: np.ndarray, requested_color: str = '', requested_type: str = '') -> float:
    """Score a vehicle candidate using user hints and prominence."""
    score = 0.0
    vehicle_type = vehicle.get('class_name', '')

    if requested_type and requested_type.lower() in vehicle_type.lower():
        score += 2.5

    detected_color = estimate_vehicle_color(image, vehicle['bbox'])
    vehicle['estimated_color'] = detected_color
    if requested_color and requested_color.lower() == detected_color.lower():
        score += 3.5

    _, _, w, h = vehicle['bbox']
    score += (w * h) / max(image.shape[0] * image.shape[1], 1) * 10.0
    score += float(vehicle.get('confidence', 0.0))

    return score


def select_uber_vehicle(
    vehicles: List[Dict[str, Any]],
    image: np.ndarray,
    requested_color: str = '',
    requested_type: str = ''
) -> Optional[Dict[str, Any]]:
    """Select the best vehicle as the Uber target."""
    if not vehicles:
        return None

    best = max(vehicles, key=lambda v: score_vehicle(v, image, requested_color, requested_type))
    return best


def is_inside_or_near(bbox: List[int], container: List[int], pad_ratio: float = 0.08) -> bool:
    """Check if bbox center is inside or near container bbox."""
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    cx0, cy0, cw, ch = container
    pad_x = int(cw * pad_ratio)
    pad_y = int(ch * pad_ratio)
    return (cx0 - pad_x) <= cx <= (cx0 + cw + pad_x) and (cy0 - pad_y) <= cy <= (cy0 + ch + pad_y)


def detect_passenger_door_handles(
    image: np.ndarray,
    vehicle_bbox: List[int],
    confidence_threshold: float = 0.2
) -> List[Dict[str, Any]]:
    """Detect passenger-side door handles with YoloWorld prompts."""
    if image is None or image.size == 0:
        return []

    detections: List[Dict[str, Any]] = []

    try:
        from ultralytics import YOLO

        model_cache = get_model_cache()
        cache_key = 'uber_handle_finder_yolov8s_world'

        if cache_key not in model_cache:
            yolo_model = YOLO('yolov8s-world.pt')
            yolo_model.set_classes(HANDLE_CLASSES)
            model_cache[cache_key] = {
                'model': yolo_model,
                'classes': tuple(HANDLE_CLASSES),
            }

        model_info = model_cache[cache_key]
        yolo_model = model_info['model']

        classes_key = tuple(HANDLE_CLASSES)
        if model_info['classes'] != classes_key:
            yolo_model.set_classes(HANDLE_CLASSES)
            model_info['classes'] = classes_key

        results = yolo_model(image, conf=confidence_threshold, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                bbox = [x, y, w, h]

                if not is_inside_or_near(bbox, vehicle_bbox):
                    continue

                center_x, center_y = x + w // 2, y + h // 2
                class_id = int(box.cls[0])
                class_name = HANDLE_CLASSES[class_id] if 0 <= class_id < len(HANDLE_CLASSES) else 'door handle'

                detections.append({
                    'class_name': class_name,
                    'confidence': float(box.conf[0]),
                    'bbox': bbox,
                    'center': [center_x, center_y],
                })
    except Exception:
        return []

    return detections


def infer_passenger_handle(vehicle_bbox: List[int], passenger_side: str = 'right') -> Dict[str, Any]:
    """Geometric fallback for passenger handle location when detection is unavailable."""
    x, y, w, h = vehicle_bbox

    side = passenger_side.lower().strip()
    handle_x_ratio = 0.78 if side != 'left' else 0.22
    handle_y_ratio = 0.58

    center_x = int(x + w * handle_x_ratio)
    center_y = int(y + h * handle_y_ratio)

    return {
        'class_name': 'door handle (estimated)',
        'confidence': 0.55,
        'bbox': [center_x - 8, center_y - 8, 16, 16],
        'center': [center_x, center_y],
    }


def describe_uber_and_handle(
    vehicle: Dict[str, Any],
    handle: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    is_streaming: bool = False
) -> str:
    """Build concise user-facing guidance."""
    vehicle_center = vehicle['center']
    vehicle_clock = get_clock_position(vehicle_center[0], vehicle_center[1], frame_width, frame_height)

    color = vehicle.get('estimated_color', 'unknown')
    vehicle_name = vehicle.get('class_name', 'vehicle')
    if color == 'unknown':
        uber_phrase = f"Your Uber is the {vehicle_name} at {vehicle_clock} o'clock."
    else:
        uber_phrase = f"Your Uber is the {color} {vehicle_name} at {vehicle_clock} o'clock."

    handle_center = handle['center']
    handle_hint = get_horizontal_hint(handle_center[0], frame_width)
    handle_phrase = f"Passenger door handle is {handle_hint}."

    text = f"{uber_phrase} {handle_phrase}"
    return limit_streaming_words(text) if is_streaming else text


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """Find the Uber vehicle and guide to the passenger door handle."""
    global _last_message

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {'type': 'error', 'text': 'No camera image available'},
            'text': 'No camera image available'
        }

    config = input_data if isinstance(input_data, dict) else {}
    confidence = float(config.get('confidence', DEFAULT_CONFIDENCE))
    requested_color = str(config.get('uber_color', '')).strip().lower()
    requested_type = str(config.get('uber_type', '')).strip().lower()
    passenger_side = str(config.get('passenger_side', 'right')).strip().lower()
    is_streaming = bool(config.get('is_streaming', False))

    frame_height, frame_width = image.shape[:2]

    vehicles = detect_vehicles(image, confidence)
    if not vehicles:
        message = 'No likely Uber vehicle found yet. Slowly pan left or right.'
        message = limit_streaming_words(message) if is_streaming else message
        if is_streaming and message == _last_message:
            return ''
        _last_message = message
        return message

    target_vehicle = select_uber_vehicle(vehicles, image, requested_color, requested_type)
    if target_vehicle is None:
        message = 'I cannot identify your Uber yet. Keep scanning nearby vehicles.'
        message = limit_streaming_words(message) if is_streaming else message
        if is_streaming and message == _last_message:
            return ''
        _last_message = message
        return message

    handles = detect_passenger_door_handles(image, target_vehicle['bbox'])
    if handles:
        if passenger_side == 'left':
            target_handle = min(handles, key=lambda h: h['center'][0])
        else:
            target_handle = max(handles, key=lambda h: h['center'][0])
    else:
        target_handle = infer_passenger_handle(target_vehicle['bbox'], passenger_side)

    message = describe_uber_and_handle(target_vehicle, target_handle, frame_width, frame_height, is_streaming)

    if is_streaming and message == _last_message:
        return ''

    _last_message = message

    if 'estimated' in target_handle.get('class_name', ''):
        return {
            'audio': {
                'type': 'warning',
                'text': message,
                'rate': 1.0,
            },
            'text': message,
        }

    return message


__all__ = [
    'main',
    'detect_vehicles',
    'select_uber_vehicle',
    'detect_passenger_door_handles',
    'infer_passenger_handle',
    'get_clock_position',
    'limit_streaming_words',
]
