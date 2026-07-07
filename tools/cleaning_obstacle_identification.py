"""Cleaning Obstacle Identification Tool

Monitors the floor area for obstacles or clutter during vacuuming and
provides audio feedback about what may need attention.
"""

from __future__ import annotations

from typing import Any

from model_router_client import copilot_llm_call

TOOL_NAME = "cleaning_obstacle_identification"
TASK_CATEGORY = "object_detection_localization"

FLOOR_OBSTACLE_LABELS = [
    "shoe", "bag", "backpack", "toy", "bottle", "book", "remote",
    "suitcase", "sports ball", "potted plant", "dog", "cat", "person",
]


def main(image: Any, input_data: Any = None) -> Any:
    if image is None:
        return {
            "audio": {"type": "error", "text": "No camera image available."},
            "text": "No camera image available.",
        }

    # Stage 1: Detect possible obstacles or clutter on the floor
    detection = copilot_llm_call(
        capability=TASK_CATEGORY,
        goal="Detect possible obstacles or clutter on the floor.",
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "detect obstacles and clutter on the floor during vacuuming",
            "target_labels": FLOOR_OBSTACLE_LABELS,
        },
    )

    artifact = detection.get("artifact") or {}
    detections = artifact.get("detections") or []

    if not detections:
        return "Floor looks clear. No obstacles or clutter detected."

    # Artifact field name may be 'label' or 'class_name' depending on backend implementation
    labels = [d.get("label") or d.get("class_name") or "object" for d in detections]
    unique_labels = list(dict.fromkeys(labels))  # preserve order, deduplicate

    if len(unique_labels) == 1:
        item_text = unique_labels[0]
    elif len(unique_labels) == 2:
        item_text = f"{unique_labels[0]} and {unique_labels[1]}"
    else:
        item_text = ", ".join(unique_labels[:-1]) + f", and {unique_labels[-1]}"

    count = len(detections)
    item_word = "item" if count == 1 else "items"

    return (
        f"Obstacle detected: {count} {item_word} on the floor including {item_text}. "
        "Please clear before continuing to vacuum."
    )
