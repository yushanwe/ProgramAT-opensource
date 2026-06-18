"""
Ride-Share Vehicle Locator Tool

Helps blind and low-vision users find their ride-share vehicle (Uber, Lyft, etc.)
in crowded pickup areas. Given a vehicle description (color, make/model, license
plate), the tool scans the camera frame, identifies the correct car, and guides
the user toward the passenger-side door.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from litellm_utils import extract_text, pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "ride_share_locator"
TASK_CATEGORY = "visual_reasoning"


def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    max_width, max_height = max_size
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def build_locator_prompt(vehicle_description: str) -> str:
    return (
        "You are assisting a blind user who needs to find their ride-share vehicle and "
        "get to the passenger-side door. Analyze the camera image carefully.\n\n"
        f"Their vehicle: {vehicle_description}\n\n"
        "Instructions:\n"
        "1. Scan the scene for the described vehicle (match color, make/model, and "
        "license plate if visible).\n"
        "2. If the correct vehicle is found, identify the passenger-side door position "
        "(passenger side is typically the right side when facing the front of the car).\n"
        "3. Give clear, concise directional guidance so the user can walk to the "
        "passenger door — use plain directions like 'slightly left', 'straight ahead', "
        "'to your right', and approximate distance if estimable.\n"
        "4. If the vehicle is NOT visible, say so briefly and suggest the user pan "
        "the camera slowly.\n\n"
        "Keep your entire response under 40 words. Speak directly to the user. "
        "Do not describe what you see beyond what is necessary to guide them."
    )


def locate_vehicle(
    image: np.ndarray,
    vehicle_description: str,
) -> Dict[str, Any]:
    try:
        processed = resize_image_if_needed(image, max_size=(1024, 1024))
        pil_image = convert_cv2_to_pil(processed)
        image_data_uri = pil_image_to_data_uri(pil_image)

        prompt = build_locator_prompt(vehicle_description)

        response = copilot_llm_call(
            task_category=TASK_CATEGORY,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a navigation assistant for a blind user. "
                        "Always respond in plain, spoken English. Be direct and brief."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            images=[image_data_uri],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "identify a specific ride-share vehicle by color, make, model, and "
                    "license plate, then guide the user to the passenger-side door"
                ),
            },
        )

        guidance = extract_text(response)
        return {"success": True, "guidance": guidance}

    except Exception as exc:
        return {"success": False, "guidance": f"Vehicle search failed: {exc}"}


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Any:
    """
    Locate a described ride-share vehicle and guide the user to the passenger door.

    Args:
        image: Camera frame as numpy array (BGR format from OpenCV).
        input_data: Optional dict with any of:
            - 'vehicle_description': Free-text description, e.g.
              "white Toyota Camry, license plate ABC123"
            - 'color':   Vehicle color, e.g. "white"
            - 'make':    Make, e.g. "Toyota"
            - 'model':   Model, e.g. "Camry"
            - 'license_plate': Plate text, e.g. "ABC123"

    Returns:
        Audio-friendly string with navigation guidance spoken via TTS.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "audio": {
                "type": "error",
                "text": "No camera image available. Please point your camera toward the pickup area.",
                "rate": 1.0,
                "interrupt": True,
            },
            "text": "No camera image available.",
        }

    if input_data is None:
        input_data = {}

    # Build a vehicle description from whatever fields are provided
    vehicle_description = input_data.get("vehicle_description", "").strip()

    if not vehicle_description:
        parts = []
        color = input_data.get("color", "").strip()
        make = input_data.get("make", "").strip()
        model = input_data.get("model", "").strip()
        plate = input_data.get("license_plate", "").strip()

        if color:
            parts.append(color)
        if make:
            parts.append(make)
        if model:
            parts.append(model)
        if plate:
            parts.append(f"license plate {plate}")

        vehicle_description = " ".join(parts).strip()

    if not vehicle_description:
        vehicle_description = "my ride-share vehicle (no description provided)"

    result = locate_vehicle(image, vehicle_description)

    if result["success"]:
        return result["guidance"]

    return {
        "audio": {
            "type": "error",
            "text": result["guidance"],
            "rate": 1.0,
            "interrupt": True,
        },
        "text": result["guidance"],
    }
