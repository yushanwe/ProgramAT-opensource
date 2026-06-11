"""Focused tests for uber_passenger_door_handle.py."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import uber_passenger_door_handle as tool


def test_build_target_description():
    print("Testing target description parsing...")

    parsed = tool.build_target_description({'uber_description': 'white Toyota Camry'})
    assert parsed == 'white Toyota Camry'

    parsed = tool.build_target_description({'color': 'white', 'make': 'Toyota', 'model': 'Camry'})
    assert parsed == 'white Toyota Camry'

    parsed = tool.build_target_description({})
    assert parsed == ''

    print("  ✓ target description parsing")


def test_streaming_word_limit():
    print("Testing streaming word limiting...")

    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    limited = tool.limit_streaming_words(text, True)
    assert len(limited.split()) == 15

    untouched = tool.limit_streaming_words("short phrase", True)
    assert untouched == "short phrase"

    print("  ✓ streaming word limiting")


def test_clock_positions_allowed_range():
    print("Testing clock position limits...")

    width = 700
    positions = [
        tool.get_clock_position(0, width),
        tool.get_clock_position(100, width),
        tool.get_clock_position(200, width),
        tool.get_clock_position(300, width),
        tool.get_clock_position(400, width),
        tool.get_clock_position(500, width),
        tool.get_clock_position(699, width),
    ]
    allowed = {1, 2, 3, 9, 10, 11, 12}
    assert all(p in allowed for p in positions)

    print("  ✓ clock positions constrained")


def test_main_with_stubbed_reasoning():
    print("Testing main with stubbed detections/reasoning...")

    original_detect = tool.detect_vehicle_candidates
    original_reason = tool.reason_about_vehicle_and_handle
    observed = {'called': False, 'confidence': None, 'image_shape': None}

    try:
        def _stub_detect(image, confidence):
            observed['called'] = True
            observed['confidence'] = confidence
            observed['image_shape'] = image.shape
            return [
                {
                    'bbox': [100, 120, 260, 180],
                    'center': [230, 210],
                    'confidence': 0.9,
                    'frame_size': [640, 480],
                }
            ]

        tool.detect_vehicle_candidates = _stub_detect

        tool.reason_about_vehicle_and_handle = lambda image, candidates, target, model, api_key: {
            'selected_index': 0,
            'handle_x': 360,
            'handle_y': 220,
            'vehicle_summary': 'the white Toyota',
            'confidence': 0.9,
        }

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tool.main(image, {'uber_description': 'white Toyota', 'is_streaming': False})

        text = result.get('text') if isinstance(result, dict) else result
        assert "Your Uber is" in text
        assert "Passenger handle" in text
        assert observed['called'] is True
        assert observed['image_shape'] == (480, 640, 3)
        assert observed['confidence'] == tool.DEFAULT_CONFIDENCE

        print(f"  ✓ main output: {text}")
    finally:
        tool.detect_vehicle_candidates = original_detect
        tool.reason_about_vehicle_and_handle = original_reason


def test_main_requests_more_details_when_multiple_vehicles():
    print("Testing multiple-vehicle prompt...")

    original_detect = tool.detect_vehicle_candidates
    original_reason = tool.reason_about_vehicle_and_handle
    observed = {'reason_called': False}
    try:
        tool.detect_vehicle_candidates = lambda image, confidence: [
            {'bbox': [10, 10, 100, 100], 'center': [60, 60], 'confidence': 0.7, 'frame_size': [640, 480]},
            {'bbox': [200, 20, 120, 110], 'center': [260, 75], 'confidence': 0.72, 'frame_size': [640, 480]},
        ]
        def _stub_reason(*_args, **_kwargs):
            observed['reason_called'] = True
            return {'selected_index': 0, 'handle_x': 60, 'handle_y': 60}
        tool.reason_about_vehicle_and_handle = _stub_reason

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tool.main(image, {'is_streaming': False})
        text = result.get('text') if isinstance(result, dict) else result
        assert "multiple vehicles" in text.lower()
        assert observed['reason_called'] is False

        print("  ✓ multiple-vehicle clarification prompt")
    finally:
        tool.detect_vehicle_candidates = original_detect
        tool.reason_about_vehicle_and_handle = original_reason


if __name__ == '__main__':
    print("=" * 60)
    print("Uber + Passenger Door Handle Tool Test Suite")
    print("=" * 60)
    print()

    test_build_target_description()
    test_streaming_word_limit()
    test_clock_positions_allowed_range()
    test_main_with_stubbed_reasoning()
    test_main_requests_more_details_when_multiple_vehicles()

    print()
    print("=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
