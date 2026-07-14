"""Available Seating Locator — finds empty chairs and guides the user toward one."""

from __future__ import annotations

from model_router_client import copilot_llm_call

TOOL_NAME = "available_seating_locator"


def main(image, input_data=None):
    if image is None:
        return "No camera image available. Please point the camera at the room."

    detection = copilot_llm_call(
        capability="object_detection_localization",
        goal="Detect all chairs, benches, and couches in the scene as well as any people sitting in them.",
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "target_labels": ["chair", "bench", "couch", "person"],
            "route_text": "detect seating and people to identify occupied vs empty seats",
        },
    )

    spatial = copilot_llm_call(
        capability="spatial_reasoning",
        goal=(
            "Using the detected chairs, benches, couches, and people, "
            "identify which seats are empty (no person sitting in or directly beside them). "
            "Report how many empty seats there are and describe the position of the nearest one "
            "relative to the camera (e.g., left, right, center, close, far)."
        ),
        metadata={
            "tool_name": TOOL_NAME,
            "previous_stage_artifact": detection["artifact"],
            "route_text": "determine which detected seats are unoccupied and locate the nearest one",
        },
    )

    guidance = copilot_llm_call(
        capability="navigation",
        goal=(
            "Guide a blind user to the nearest empty seat identified in the previous step. "
            "Give concise walking directions: which way to turn and how far to walk. "
            "If no empty seats were found, say so clearly."
        ),
        metadata={
            "tool_name": TOOL_NAME,
            "previous_stage_artifact": spatial["artifact"],
            "route_text": "provide walking directions to the nearest empty seat for a blind user",
        },
    )

    return guidance["response"]
