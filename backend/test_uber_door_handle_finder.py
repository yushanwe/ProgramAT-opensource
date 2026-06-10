"""Focused tests for uber_door_handle_finder.py building blocks."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import uber_door_handle_finder as tool


def test_clock_positions():
    assert tool.get_clock_position(320, 60, 640, 480) == 12
    assert tool.get_clock_position(560, 120, 640, 480) == 1
    assert tool.get_clock_position(560, 260, 640, 480) == 2
    assert tool.get_clock_position(560, 420, 640, 480) == 3
    assert tool.get_clock_position(80, 120, 640, 480) == 11
    assert tool.get_clock_position(80, 260, 640, 480) == 10
    assert tool.get_clock_position(80, 420, 640, 480) == 9


def test_streaming_limit():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    limited = tool.limit_streaming_words(text)
    assert len(limited.split()) == 15


def test_infer_handle_location():
    bbox = [100, 100, 200, 120]
    right = tool.infer_passenger_handle(bbox, 'right')
    left = tool.infer_passenger_handle(bbox, 'left')
    assert right['center'][0] > left['center'][0]


def test_main_with_stubbed_detectors():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    original_detect_vehicles = tool.detect_vehicles
    original_detect_handles = tool.detect_passenger_door_handles

    try:
        def fake_detect_vehicles(_image, confidence_threshold):
            return [{
                'class_name': 'car',
                'confidence': 0.9,
                'bbox': [180, 150, 280, 180],
                'center': [320, 240],
                'estimated_color': 'white',
            }]

        def fake_detect_handles(_image, _bbox):
            return [{
                'class_name': 'car door handle',
                'confidence': 0.8,
                'bbox': [370, 240, 30, 12],
                'center': [385, 246],
            }]

        tool.detect_vehicles = fake_detect_vehicles
        tool.detect_passenger_door_handles = fake_detect_handles

        result = tool.main(image, {'is_streaming': False, 'uber_color': 'white'})
        assert isinstance(result, str)
        assert 'Your Uber is' in result
        assert 'Passenger door handle' in result
    finally:
        tool.detect_vehicles = original_detect_vehicles
        tool.detect_passenger_door_handles = original_detect_handles


def test_main_invalid_image():
    result = tool.main(None, {})
    assert isinstance(result, dict)
    assert result['audio']['type'] == 'error'


if __name__ == '__main__':
    test_clock_positions()
    test_streaming_limit()
    test_infer_handle_location()
    test_main_with_stubbed_detectors()
    test_main_invalid_image()
    print('All uber door handle finder tests passed')
