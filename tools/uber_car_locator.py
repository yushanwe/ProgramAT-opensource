"""
Uber / Rideshare Car Locator Tool

Helps blind and low-vision users locate their rideshare vehicle in crowded pickup
areas and navigate to the passenger-side door.

Given vehicle details via input_data, the tool:
  1. Identifies the matching vehicle in the camera frame.
  2. Locates the passenger-side door.
  3. Provides concise directional guidance toward that door.

Configuration (input_data dict):
  color         – Vehicle color, e.g. "white"
  make          – Make / brand, e.g. "Toyota"
  model         – Model name, e.g. "Camry"
  license_plate – Plate number, e.g. "ABC123"

Example return value:
  "Your white Toyota Camry is ahead and slightly right.
   Passenger door is on the right side, about 10 feet away."
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from litellm_utils import extract_text, pil_image_to_data_uri
from model_router_client import copilot_llm_call

TOOL_NAME = "uber_car_locator"
TASK_CATEGORY = "visual_reasoning"


def _bgr_to_pil(image: np.ndarray):
    """Convert a BGR OpenCV array to a PIL Image."""
    from PIL import Image as PILImage
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """Locate a rideshare vehicle and guide the user to the passenger door."""
    if image is None:
        return "No camera image available. Please point your camera toward the pickup area."

    if input_data is None:
        input_data = {}

    color = input_data.get("color", "").strip()
    make = input_data.get("make", "").strip()
    model = input_data.get("model", "").strip()
    license_plate = input_data.get("license_plate", "").strip()

    # Build a human-readable vehicle description for the prompt.
    parts = [p for p in [color, make, model] if p]
    vehicle_desc = " ".join(parts) if parts else "rideshare vehicle"
    plate_note = f" with license plate {license_plate}" if license_plate else ""

    prompt = (
        f"You are helping a blind person board their rideshare vehicle safely.\n\n"
        f"Their vehicle: {vehicle_desc}{plate_note}.\n\n"
        f"Step 1 – Identify: Is a {vehicle_desc}{plate_note} visible in the image? "
        f"If multiple vehicles are present, focus on the one that best matches the description.\n"
        f"Step 2 – Locate door: If found, identify the passenger-side door "
        f"(right side of the vehicle when standing outside facing the front).\n"
        f"Step 3 – Navigate: Give concise walking directions toward that door using "
        f"plain language such as 'ahead', 'slightly left', 'to your right', and an "
        f"approximate distance (e.g. 'a few steps', 'about 10 feet').\n\n"
        f"If the vehicle is not visible, say so clearly and suggest the user scan around.\n"
        f"Respond in 40 words or fewer, in natural spoken language suitable for a screen reader."
    )

    try:
        pil_image = _bgr_to_pil(image)
        image_uri = pil_image_to_data_uri(pil_image)

        response = copilot_llm_call(
            task_category=TASK_CATEGORY,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You assist blind users in locating vehicles and reaching them safely. "
                        "Be brief, concrete, and audio-friendly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            images=[image_uri],
            metadata={
                "tool_name": TOOL_NAME,
                "route_text": (
                    "identify a specific rideshare vehicle by color, make, model, and license plate, "
                    "locate its passenger-side door, and give walking directions to it"
                ),
            },
        )

        guidance = extract_text(response)
        if not guidance:
            return "Could not analyze the image. Please try again."
        return guidance

    except Exception as exc:
        return {
            "audio": {"type": "error", "text": "Unable to locate vehicle. Please try again."},
            "text": f"Error: {exc}",
        }
