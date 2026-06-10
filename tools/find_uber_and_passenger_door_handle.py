"""
Find Uber and Passenger Door Handle tool.

Runs on the backend server with camera frames from the app and returns
audio-friendly guidance for TTS playback.

Task Pipeline (minimal stages):
1) object_detection: detect vehicles in the scene.
2) visual_reasoning: pick the most likely Uber candidate using user clues and geometry.
3) object_localization: localize passenger-side door handle on the selected vehicle.

Earlier-stage outputs are reused by later stages:
- Stage 2 reuses Stage 1 vehicle detections.
- Stage 3 reuses Stage 2 selected vehicle bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


STREAMING_WORD_LIMIT = 15  # Keep streaming speech brief and responsive for repeated frame updates.
HANDLE_CONFIDENCE_DELTA = 0.05  # Slightly lower threshold helps find small handle regions.
MIN_HANDLE_CONFIDENCE = 0.2  # Lower bound to avoid dropping localization confidence too far.
VEHICLE_DETECTION_LABELS = ["car", "truck", "bus", "vehicle", "taxi"]
VEHICLE_LABEL_HINTS = set(VEHICLE_DETECTION_LABELS) | {"van", "suv"}
HANDLE_LABELS = ["car door handle", "vehicle door handle", "door handle"]

_stream_state_by_key: Dict[str, Tuple[bool, Optional[int], Optional[int]]] = {}
_stream_state_lock = threading.Lock()
_model_cache_lock = threading.Lock()
_shared_model_cache: Dict[str, Dict[str, Any]] = {}


@dataclass
class ToolConfig:
    confidence: float = 0.35
    is_streaming: bool = False
    uber_clues: str = ""
    stream_key: str = "default"
    common_model_path: Optional[str] = None
    localization_model_path: Optional[str] = None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _word_limit(text: str, max_words: int = STREAMING_WORD_LIMIT) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _color_name_from_crop(crop: np.ndarray) -> str:
    if crop.size == 0:
        return ""

    avg_bgr = crop.reshape(-1, 3).mean(axis=0)
    b, g, r = [float(x) for x in avg_bgr]

    brightness = (r + g + b) / 3.0
    if brightness < 45:
        return "black"
    if brightness > 210:
        return "white"

    if abs(r - g) < 20 and abs(g - b) < 20:
        return "gray"
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    if b >= r and b >= g:
        return "blue"
    return "dark"


def get_clock_position(center_x: int, center_y: int, frame_width: int, frame_height: int) -> int:
    """Map an object center to supported camera-visible clock positions.

    Returns only 1-3 and 9-12 because 4-8 would be behind the camera perspective.
    """
    if frame_width <= 0 or frame_height <= 0:
        return 12

    norm_x = center_x / frame_width
    norm_y = center_y / frame_height

    if norm_x < 0.33:
        h_region = "left"
    elif norm_x < 0.67:
        h_region = "center"
    else:
        h_region = "right"

    if norm_y < 0.33:
        v_region = "top"
    elif norm_y < 0.67:
        v_region = "middle"
    else:
        v_region = "bottom"

    if h_region == "center" and v_region == "top":
        return 12

    if h_region == "right":
        if v_region == "top":
            return 1
        if v_region == "middle":
            return 2
        return 3

    if h_region == "left":
        if v_region == "top":
            return 11
        if v_region == "middle":
            return 10
        return 9

    return 12


def _extract_config(input_data: Any) -> ToolConfig:
    if not isinstance(input_data, dict):
        return ToolConfig()

    return ToolConfig(
        confidence=float(input_data.get("confidence", 0.35)),
        is_streaming=bool(input_data.get("is_streaming", False)),
        uber_clues=str(input_data.get("uber_clues", "") or ""),
        stream_key=str(input_data.get("stream_key") or input_data.get("session_id") or "default"),
        common_model_path=input_data.get("common_model_path"),
        localization_model_path=input_data.get("localization_model_path"),
    )


def _vehicle_area(det: Dict[str, Any]) -> int:
    _, _, w, h = det.get("bbox", [0, 0, 0, 0])
    return max(0, _safe_int(w)) * max(0, _safe_int(h))


def _det_center(det: Dict[str, Any]) -> Tuple[int, int]:
    center = det.get("center", [0, 0])
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return _safe_int(center[0]), _safe_int(center[1])
    x, y, w, h = det.get("bbox", [0, 0, 0, 0])
    return _safe_int(x + w // 2), _safe_int(y + h // 2)


def _filter_vehicle_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for det in detections:
        class_name = str(det.get("class_name", "")).lower()
        if any(hint in class_name for hint in VEHICLE_LABEL_HINTS):
            filtered.append(det)
    return filtered


def _direction_bonus(clues: str, center_x: int, frame_width: int) -> float:
    clues_lower = clues.lower()
    third = frame_width / 3 if frame_width else 1

    if "left" in clues_lower and center_x < third:
        return 0.5
    if "right" in clues_lower and center_x > (2 * third):
        return 0.5
    if "center" in clues_lower and third <= center_x <= (2 * third):
        return 0.5
    return 0.0


def select_uber_candidate(
    vehicle_detections: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    uber_clues: str = "",
) -> Optional[Dict[str, Any]]:
    """Pick the most likely Uber candidate from vehicle detections."""
    if not vehicle_detections:
        return None

    best_det = None
    best_score = -1.0

    frame_area = max(1, frame_width * frame_height)

    for det in vehicle_detections:
        center_x, _ = _det_center(det)
        area_ratio = _vehicle_area(det) / frame_area
        center_bias = 1.0 - abs((center_x / max(1, frame_width)) - 0.5)
        score = area_ratio + center_bias * 0.25 + _direction_bonus(uber_clues, center_x, frame_width)

        if score > best_score:
            best_score = score
            best_det = det

    return best_det


def _clamp_bbox(bbox: List[int], frame_width: int, frame_height: int) -> List[int]:
    x, y, w, h = [_safe_int(v) for v in bbox]
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(1, min(w, frame_width - x))
    h = max(1, min(h, frame_height - y))
    return [x, y, w, h]


def _discover_model_path(prefer_localization: bool = False) -> Optional[str]:
    """Find an available local model file in tools/ and optionally prefer localization models."""
    current_dir = Path(__file__).resolve().parent
    candidate_dirs = [current_dir]

    all_models: List[Path] = []
    for candidate_dir in candidate_dirs:
        if candidate_dir.exists():
            all_models.extend(sorted(candidate_dir.glob("*.pt")))

    if not all_models:
        return None

    if prefer_localization:
        for model_path in all_models:
            if "world" in model_path.name.lower():
                return str(model_path)

    for model_path in all_models:
        if "world" not in model_path.name.lower():
            return str(model_path)

    return str(all_models[0])


class ModelRouter:
    """Capability router for model-backed detection/localization with cached model instances."""

    def __init__(self, cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._cache = cache if cache is not None else _shared_model_cache

    def run(
        self,
        capability: str,
        image: np.ndarray,
        *,
        labels: Optional[List[str]] = None,
        confidence: float = 0.35,
        model_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if capability not in {"object_detection", "object_localization"}:
            return []
        return self._run_detector(image, labels=labels or [], confidence=confidence, model_path=model_path)

    def _run_detector(
        self,
        image: np.ndarray,
        *,
        labels: List[str],
        confidence: float,
        model_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []

        try:
            from ultralytics import YOLO
        except Exception:
            return []

        if not model_path:
            prefer_localization = any("handle" in label.lower() for label in labels)
            model_path = _discover_model_path(prefer_localization=prefer_localization)
        if not model_path:
            return []

        cache_key = f"find_uber::{model_path}"
        with _model_cache_lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = {"model": YOLO(model_path), "labels": None}
            model_info = self._cache[cache_key]
        model = model_info["model"]

        label_key = tuple(labels)
        if labels and hasattr(model, "set_classes"):
            with _model_cache_lock:
                if model_info.get("labels") != label_key:
                    model.set_classes(labels)
                    model_info["labels"] = label_key

        try:
            results = model(image, conf=confidence, verbose=False)
        except Exception:
            return []

        detections: List[Dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, class_id))

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(box.conf[0]),
                        "bbox": [x, y, w, h],
                        "center": [x + w // 2, y + h // 2],
                    }
                )

        return detections


def locate_passenger_handle(
    image: np.ndarray,
    vehicle_bbox: List[int],
    router: ModelRouter,
    confidence: float,
    model_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Localize a likely passenger-side door handle in the selected vehicle."""
    frame_h, frame_w = image.shape[:2]
    vx, vy, vw, vh = _clamp_bbox(vehicle_bbox, frame_w, frame_h)
    vehicle_crop = image[vy : vy + vh, vx : vx + vw]

    detections = router.run(
        "object_localization",
        vehicle_crop,
        labels=HANDLE_LABELS,
        confidence=confidence,
        model_path=model_path,
    )
    if not detections:
        return None

    # Passenger-side heuristic: use right-most handle in crop.
    # This can be inaccurate with opposing camera angles or left-hand-drive contexts.
    best = max(detections, key=lambda d: _det_center(d)[0])
    cx, cy = _det_center(best)
    full_cx = vx + cx
    full_cy = vy + cy

    best = dict(best)
    best["center"] = [full_cx, full_cy]
    best["bbox"] = [
        vx + _safe_int(best["bbox"][0]),
        vy + _safe_int(best["bbox"][1]),
        _safe_int(best["bbox"][2]),
        _safe_int(best["bbox"][3]),
    ]
    return best


def _stream_state(found_vehicle: bool, vehicle_clock: Optional[int], handle_clock: Optional[int]) -> Tuple[bool, Optional[int], Optional[int]]:
    return (found_vehicle, vehicle_clock, handle_clock)


def _get_stream_state(stream_key: str) -> Optional[Tuple[bool, Optional[int], Optional[int]]]:
    with _stream_state_lock:
        return _stream_state_by_key.get(stream_key)


def _set_stream_state(stream_key: str, state: Tuple[bool, Optional[int], Optional[int]]) -> None:
    with _stream_state_lock:
        _stream_state_by_key[stream_key] = state


def reset_stream_state(stream_key: Optional[str] = None) -> None:
    """Reset streaming state for one key or all keys.

    Use this when streaming sessions end, change tools, or need deterministic tests.
    """
    with _stream_state_lock:
        if stream_key:
            _stream_state_by_key.pop(stream_key, None)
        else:
            _stream_state_by_key.clear()


def _build_output(
    image: np.ndarray,
    selected_vehicle: Dict[str, Any],
    handle_detection: Optional[Dict[str, Any]],
    is_streaming: bool,
) -> str:
    frame_h, frame_w = image.shape[:2]

    vehicle_center = _det_center(selected_vehicle)
    vehicle_clock = get_clock_position(vehicle_center[0], vehicle_center[1], frame_w, frame_h)

    vx, vy, vw, vh = _clamp_bbox(selected_vehicle.get("bbox", [0, 0, 1, 1]), frame_w, frame_h)
    vehicle_crop = image[vy : vy + vh, vx : vx + vw]
    color = _color_name_from_crop(vehicle_crop)

    car_phrase = "car"
    if color:
        car_phrase = f"{color} car"

    if not handle_detection:
        msg = f"Uber candidate {car_phrase} at {vehicle_clock} o'clock. Passenger handle not visible yet."
        return _word_limit(msg) if is_streaming else msg

    handle_center = _det_center(handle_detection)
    handle_clock = get_clock_position(handle_center[0], handle_center[1], frame_w, frame_h)

    if is_streaming:
        msg = f"Uber candidate at {vehicle_clock} o'clock. Handle at {handle_clock} o'clock."
        return _word_limit(msg)

    return f"Your Uber is likely the {car_phrase} at {vehicle_clock} o'clock. Passenger handle at {handle_clock} o'clock."


def main(image: np.ndarray, input_data: Any = None) -> Any:
    """Entry point for finding Uber vehicle and passenger door handle."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {"type": "error", "text": "No camera image available"},
            "text": "No camera image available",
        }

    config = _extract_config(input_data)
    frame_h, frame_w = image.shape[:2]

    # Optional test hooks for deterministic unit tests.
    mock_vehicle_detections = input_data.get("mock_vehicle_detections") if isinstance(input_data, dict) else None
    mock_handle_detections = input_data.get("mock_handle_detections") if isinstance(input_data, dict) else None

    router = ModelRouter()

    vehicle_detections = mock_vehicle_detections if isinstance(mock_vehicle_detections, list) else router.run(
        "object_detection",
        image,
        labels=VEHICLE_DETECTION_LABELS,
        confidence=config.confidence,
        model_path=config.common_model_path,
    )
    vehicle_detections = _filter_vehicle_detections(vehicle_detections)

    if not vehicle_detections:
        current_state = _stream_state(False, None, None)
        if config.is_streaming and _get_stream_state(config.stream_key) == current_state:
            return ""
        _set_stream_state(config.stream_key, current_state)
        msg = "No likely Uber vehicle visible. Pan slowly left and right."
        return _word_limit(msg) if config.is_streaming else msg

    selected_vehicle = select_uber_candidate(
        vehicle_detections,
        frame_w,
        frame_h,
        uber_clues=config.uber_clues,
    )
    if not selected_vehicle:
        msg = "Vehicle detected, but Uber match is unclear. Try getting closer."
        return _word_limit(msg) if config.is_streaming else msg

    if isinstance(mock_handle_detections, list):
        handle_detection = None
        if mock_handle_detections:
            best = max(mock_handle_detections, key=lambda d: _det_center(d)[0])
            handle_detection = dict(best)
    else:
        handle_detection = locate_passenger_handle(
            image,
            selected_vehicle.get("bbox", [0, 0, frame_w, frame_h]),
            router,
            confidence=max(MIN_HANDLE_CONFIDENCE, config.confidence - HANDLE_CONFIDENCE_DELTA),
            model_path=config.localization_model_path,
        )

    vehicle_center = _det_center(selected_vehicle)
    vehicle_clock = get_clock_position(vehicle_center[0], vehicle_center[1], frame_w, frame_h)
    handle_clock = None
    if handle_detection:
        hc = _det_center(handle_detection)
        handle_clock = get_clock_position(hc[0], hc[1], frame_w, frame_h)

    current_state = _stream_state(True, vehicle_clock, handle_clock)
    if config.is_streaming and _get_stream_state(config.stream_key) == current_state:
        return ""

    _set_stream_state(config.stream_key, current_state)

    message = _build_output(image, selected_vehicle, handle_detection, config.is_streaming)

    if handle_detection:
        return {
            "audio": {"type": "success", "text": message, "interrupt": True},
            "text": message,
        }

    return message


__all__ = [
    "main",
    "get_clock_position",
    "select_uber_candidate",
    "locate_passenger_handle",
    "reset_stream_state",
    "ModelRouter",
]
