"""
Find Uber Vehicle and Passenger Door Handle Tool

Capability Pipeline (minimal stages)
1) object_detection: Detect vehicles in frame and extract bounding boxes.
2) object_localization: Estimate passenger door-handle location on selected vehicle.
3) navigation: Convert localization to clock-face guidance and concise spoken instructions.

This tool is designed for blind and low-vision users. It returns audio-friendly guidance
that first identifies a likely Uber vehicle, then directs the user to the passenger handle.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIDENCE = 0.35
STREAMING_WORD_LIMIT = 15
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

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


def get_clock_position(center_x: int, center_y: int, frame_width: int, frame_height: int) -> int:
    """Map point to visible clock positions (1-3, 9-12)."""
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

    if h_region == 'center' and v_region == 'top':
        return 12
    if h_region == 'right':
        return 1 if v_region == 'top' else 2 if v_region == 'middle' else 3
    if h_region == 'left':
        return 11 if v_region == 'top' else 10 if v_region == 'middle' else 9
    return 12


def estimate_color_in_bbox(image: np.ndarray, bbox: List[int]) -> str:
    """Estimate dominant color name in a bounding box."""
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]

    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(x1 + 1, min(x + w, w_img))
    y2 = max(y1 + 1, min(y + h, h_img))

    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return "unknown"

    b, g, r = np.mean(region, axis=(0, 1))
    brightness = (r + g + b) / 3

    if brightness > 205:
        return "white"
    if brightness < 55:
        return "black"
    if abs(r - g) < 18 and abs(g - b) < 18:
        return "gray"
    if r > g + 30 and r > b + 30:
        return "red"
    if g > r + 30 and g > b + 30:
        return "green"
    if b > r + 30 and b > g + 30:
        return "blue"
    if r > 140 and g > 140 and b < 110:
        return "yellow"
    if brightness > 150:
        return "silver"
    return "dark"


def detect_vehicles(image: np.ndarray, confidence_threshold: float = DEFAULT_CONFIDENCE) -> List[Dict[str, Any]]:
    """Detect COCO vehicles using YOLO11."""
    if image is None or image.size == 0:
        return []

    detections: List[Dict[str, Any]] = []

    try:
        from ultralytics import YOLO

        model_cache = globals().get('yolo_model_cache', {})
        cache_key = 'find_uber_yolo11n'

        if cache_key not in model_cache:
            model_cache[cache_key] = YOLO('yolo11n.pt')

        model = model_cache[cache_key]
        results = model(image, conf=confidence_threshold, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else "object"
                if class_name not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = max(int(x2 - x1), 1), max(int(y2 - y1), 1)
                center = [x + w // 2, y + h // 2]
                bbox = [x, y, w, h]

                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(box.conf[0]),
                    'bbox': bbox,
                    'center': center,
                    'area': w * h,
                    'color': estimate_color_in_bbox(image, bbox),
                })

    except Exception:
        return []

    return detections


def parse_target_details(input_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Parse user-provided hints for identifying the Uber."""
    data = input_data if isinstance(input_data, dict) else {}
    color = str(data.get('vehicle_color', data.get('color', ''))).strip().lower()
    make = str(data.get('vehicle_make', data.get('make', ''))).strip().lower()
    model = str(data.get('vehicle_model', data.get('model', ''))).strip().lower()
    plate = str(data.get('license_plate', data.get('plate', ''))).strip().lower()

    return {
        'color': color,
        'make': make,
        'model': model,
        'plate': plate,
    }


def score_vehicle_match(vehicle: Dict[str, Any], target: Dict[str, str]) -> float:
    """Score how well a detected vehicle matches target hints."""
    score = float(vehicle.get('area', 0)) / 100000.0

    target_color = target.get('color', '')
    if target_color:
        if vehicle.get('color') == target_color:
            score += 3.0
        elif target_color in ('gray', 'silver') and vehicle.get('color') in ('gray', 'silver'):
            score += 2.0

    return score


def select_target_vehicle(vehicles: List[Dict[str, Any]], target: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Select likely Uber vehicle and whether match confidence is strong."""
    if not vehicles:
        return None, False

    scored = sorted(vehicles, key=lambda v: score_vehicle_match(v, target), reverse=True)
    chosen = scored[0]

    has_target_color = bool(target.get('color'))
    color_matched = (not has_target_color) or chosen.get('color') == target.get('color')
    return chosen, color_matched


def estimate_passenger_handle(vehicle_bbox: List[int], frame_width: int, frame_height: int, side: str = 'right') -> Dict[str, Any]:
    """Estimate passenger door-handle point from vehicle geometry."""
    x, y, w, h = vehicle_bbox

    horizontal_ratio = 0.22 if side == 'left' else 0.78
    handle_x = int(x + w * horizontal_ratio)
    handle_y = int(y + h * 0.56)

    handle_x = max(0, min(handle_x, frame_width - 1))
    handle_y = max(0, min(handle_y, frame_height - 1))

    clock = get_clock_position(handle_x, handle_y, frame_width, frame_height)

    height_ratio = h / max(frame_height, 1)
    if height_ratio > 0.6:
        distance = 'very close'
    elif height_ratio > 0.35:
        distance = 'close'
    elif height_ratio > 0.18:
        distance = 'near'
    else:
        distance = 'far'

    return {
        'point': [handle_x, handle_y],
        'clock': clock,
        'distance': distance,
    }


def _trim_for_streaming(text: str, max_words: int = STREAMING_WORD_LIMIT) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def build_guidance(
    vehicle: Dict[str, Any],
    vehicle_matched: bool,
    handle: Dict[str, Any],
    target: Dict[str, str],
    frame_width: int,
    frame_height: int,
    is_streaming: bool,
) -> Any:
    """Build final audio-friendly response."""
    vehicle_clock = get_clock_position(vehicle['center'][0], vehicle['center'][1], frame_width, frame_height)
    vehicle_color = vehicle.get('color', 'unknown')

    if vehicle_matched and target.get('color'):
        first_sentence = f"Your Uber is the {vehicle_color} {vehicle['class_name']} at {vehicle_clock} o'clock."
    else:
        first_sentence = f"Likely Uber is the {vehicle_color} {vehicle['class_name']} at {vehicle_clock} o'clock."

    second_sentence = f"Passenger handle near {handle['clock']} o'clock, {handle['distance']}."
    message = f"{first_sentence} {second_sentence}"

    if is_streaming:
        return _trim_for_streaming(message)

    if handle['distance'] in ('very close', 'close'):
        return {
            'audio': {
                'type': 'beep_high',
                'text': message,
                'rate': 1.15,
                'interrupt': True,
            },
            'text': message,
        }

    return message


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """Entry point for Uber-and-handle guidance."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {
                'type': 'error',
                'text': 'No camera image available.',
            },
            'text': 'No camera image available.',
        }

    config = input_data if isinstance(input_data, dict) else {}
    confidence = float(config.get('confidence', DEFAULT_CONFIDENCE))
    confidence = max(0.05, min(confidence, 0.95))
    is_streaming = bool(config.get('is_streaming', False))
    passenger_side = str(config.get('passenger_side', 'right')).strip().lower()
    if passenger_side not in ('left', 'right'):
        passenger_side = 'right'

    target = parse_target_details(config)

    vehicles = detect_vehicles(image, confidence)
    if not vehicles:
        text = 'No vehicle found yet. Pan slowly to locate your Uber.'
        return _trim_for_streaming(text) if is_streaming else text

    frame_height, frame_width = image.shape[:2]
    vehicle, vehicle_matched = select_target_vehicle(vehicles, target)
    if vehicle is None:
        text = 'No vehicle found yet. Pan slowly to locate your Uber.'
        return _trim_for_streaming(text) if is_streaming else text

    handle = estimate_passenger_handle(vehicle['bbox'], frame_width, frame_height, passenger_side)
    return build_guidance(vehicle, vehicle_matched, handle, target, frame_width, frame_height, is_streaming)


__all__ = [
    'main',
    'detect_vehicles',
    'parse_target_details',
    'score_vehicle_match',
    'select_target_vehicle',
    'estimate_passenger_handle',
    'get_clock_position',
    'build_guidance',
]
