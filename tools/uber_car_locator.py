"""
Uber / Rideshare Car Locator Tool

Helps blind and low-vision users find their rideshare vehicle in a crowded
pickup area.  Given vehicle details (color, make, model, license plate), the
tool:
  1. Identifies the matching car in the camera frame.
  2. Locates the passenger-side door.
  3. Gives spoken navigation guidance so the user can walk straight to it.

A single visual-reasoning model call handles all three stages so the spoken
result is coherent and concise.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from litellm_utils import extract_text, pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "uber_car_locator"
TASK_CATEGORY = "visual_reasoning"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_if_needed(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _to_data_uri(image: np.ndarray) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return pil_image_to_data_uri(pil_img)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(color: str, make: str, model: str, plate: str) -> str:
    vehicle_desc = " ".join(part for part in [color, make, model] if part)
    plate_clause = f" with license plate {plate}" if plate else ""

    return (
        f"You are a navigation assistant helping a blind person find their rideshare vehicle "
        f"in a crowded pickup area.\n\n"
        f"Target vehicle: {vehicle_desc}{plate_clause}.\n\n"
        f"Please do the following in a single, spoken-aloud response:\n"
        f"1. State whether the target vehicle is visible. If it is not visible, say so clearly.\n"
        f"2. If visible, describe where it is relative to the camera center "
        f"(e.g., slightly left, straight ahead, far right).\n"
        f"3. Identify where the passenger-side door is on the vehicle "
        f"(e.g., front-right door, rear-right door).\n"
        f"4. Give one or two short walking directions to reach that door "
        f"(e.g., 'Walk forward and slightly right about ten steps').\n\n"
        f"Rules:\n"
        f"- Speak naturally, as if guiding someone in real time.\n"
        f"- Be concise — the response will be read aloud.\n"
        f"- If multiple cars match the description, guide toward the closest one.\n"
        f"- Do not describe any other vehicles in detail."
    )


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def locate_vehicle(
    image: np.ndarray,
    color: str = "",
    make: str = "",
    model: str = "",
    plate: str = "",
) -> str:
    """
    Send the camera frame plus vehicle description to the visual-reasoning
    backend and return a spoken navigation string.
    """
    processed = _resize_if_needed(image)
    data_uri = _to_data_uri(processed)
    prompt = _build_prompt(color, make, model, plate)

    response = copilot_llm_call(
        task_category=TASK_CATEGORY,
        messages=[
            {
                "role": "system",
                "content": (
                    "You help blind users find their rideshare car. "
                    "Keep responses natural, concise, and audio-friendly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        images=[data_uri],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": (
                "identify a specific rideshare vehicle by color/make/model/plate, "
                "locate its passenger-side door, and give walking directions"
            ),
        },
    )

    return extract_text(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(image: Optional[np.ndarray], input_data: Optional[Dict[str, Any]] = None) -> Any:
    """
    Locate the user's rideshare vehicle and guide them to the passenger door.

    input_data keys (all optional):
        color  – vehicle color, e.g. "white"
        make   – manufacturer, e.g. "Toyota"
        model  – model name, e.g. "Camry"
        plate  – license-plate text, e.g. "ABC123"

    Returns an audio-friendly string or error dict.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {
                "type": "error",
                "text": "No camera image available. Please point the camera at the pickup area.",
                "rate": 1.0,
                "interrupt": False,
            },
            "text": "No camera image available.",
        }

    if input_data is None:
        input_data = {}

    color = input_data.get("color", "")
    make = input_data.get("make", "")
    model = input_data.get("model", "")
    plate = input_data.get("plate", "")

    if not any([color, make, model, plate]):
        return (
            "Please provide vehicle details such as color, make, model, or license plate "
            "so I can identify your car."
        )

    try:
        guidance = locate_vehicle(image, color=color, make=make, model=model, plate=plate)
        if not guidance:
            return {
                "audio": {
                    "type": "warning",
                    "text": "Could not determine vehicle location. Try pointing the camera at the pickup area.",
                    "rate": 1.0,
                    "interrupt": False,
                },
                "text": "Could not determine vehicle location.",
            }
        return guidance
    except Exception as exc:
        return {
            "audio": {
                "type": "error",
                "text": f"Vehicle search failed: {exc}",
                "rate": 1.0,
                "interrupt": False,
            },
            "text": f"Vehicle search failed: {exc}",
        }


__all__ = ["main", "locate_vehicle"]
