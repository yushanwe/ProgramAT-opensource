"""
Test script for find_uber_and_passenger_door_handle.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from find_uber_and_passenger_door_handle import (
    get_clock_position,
    select_uber_candidate,
    reset_stream_state,
    main,
)


def _create_test_image() -> np.ndarray:
    image = np.ones((480, 640, 3), dtype=np.uint8) * 180
    # bright car-like region
    image[120:360, 380:620] = (220, 220, 220)
    return image


def test_clock_positions() -> None:
    print("Testing clock position mapping...")
    cases = [
        ((320, 80), 12),
        ((560, 120), 1),
        ((560, 240), 2),
        ((560, 420), 3),
        ((80, 120), 11),
        ((80, 240), 10),
        ((80, 420), 9),
    ]

    for (x, y), expected in cases:
        result = get_clock_position(x, y, 640, 480)
        status = "✓" if result == expected else "✗"
        print(f"  {status} ({x}, {y}) -> {result}, expected {expected}")
    print()


def test_candidate_selection() -> None:
    print("Testing Uber candidate selection...")
    detections = [
        {"class_name": "car", "bbox": [40, 160, 140, 180], "center": [110, 250]},
        {"class_name": "car", "bbox": [360, 120, 220, 240], "center": [470, 240]},
    ]

    selected = select_uber_candidate(detections, 640, 480, uber_clues="on the right")
    status = "✓" if selected and selected["center"][0] == 470 else "✗"
    print(f"  {status} selected center: {selected['center'] if selected else None}")
    print()


def test_main_behaviors() -> None:
    print("Testing main() behavior...")

    # reset streaming state for deterministic tests
    reset_stream_state()

    no_image = main(None, {})
    print(f"  None image response type: {'dict' if isinstance(no_image, dict) else 'str'}")

    image = _create_test_image()

    no_vehicle = main(image, {"mock_vehicle_detections": []})
    print(f"  No vehicle: {no_vehicle}")

    vehicle = {
        "class_name": "car",
        "bbox": [360, 120, 220, 240],
        "center": [470, 240],
        "confidence": 0.91,
    }
    handle = {
        "class_name": "car door handle",
        "bbox": [520, 250, 30, 20],
        "center": [535, 260],
        "confidence": 0.88,
    }

    one_shot = main(
        image,
        {
            "mock_vehicle_detections": [vehicle],
            "mock_handle_detections": [handle],
            "is_streaming": False,
        },
    )
    print(f"  One-shot with handle response type: {'dict' if isinstance(one_shot, dict) else 'str'}")
    print(f"  One-shot text: {one_shot.get('text') if isinstance(one_shot, dict) else one_shot}")

    reset_stream_state()
    stream_first = main(
        image,
        {
            "mock_vehicle_detections": [vehicle],
            "mock_handle_detections": [handle],
            "is_streaming": True,
            "stream_key": "test-stream",
        },
    )
    stream_second = main(
        image,
        {
            "mock_vehicle_detections": [vehicle],
            "mock_handle_detections": [handle],
            "is_streaming": True,
            "stream_key": "test-stream",
        },
    )

    first_text = stream_first.get("text", "") if isinstance(stream_first, dict) else str(stream_first)
    print(f"  Streaming first: {first_text}")
    print(f"  Streaming first word count: {len(first_text.split())}")
    print(f"  Streaming second (should be empty): '{stream_second}'")
    print()


def run_all_tests() -> None:
    print("=" * 70)
    print("Find Uber and Passenger Door Handle Tool Tests")
    print("=" * 70)
    test_clock_positions()
    test_candidate_selection()
    test_main_behaviors()
    print("=" * 70)
    print("Tests completed")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
