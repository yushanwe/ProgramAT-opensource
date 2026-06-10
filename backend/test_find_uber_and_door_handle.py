"""Test script for find_uber_and_door_handle.py."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import find_uber_and_door_handle as tool


def test_parse_target_details():
    print('Testing parse_target_details()...')
    details = tool.parse_target_details({'vehicle_color': 'White', 'make': 'Toyota'})
    assert details['color'] == 'white'
    assert details['make'] == 'toyota'
    print('  ✓ parsed target details')


def test_select_target_vehicle_prefers_color_match():
    print('Testing select_target_vehicle() color match...')
    vehicles = [
        {'class_name': 'car', 'area': 40000, 'color': 'black'},
        {'class_name': 'car', 'area': 35000, 'color': 'white'},
    ]
    selected, matched = tool.select_target_vehicle(vehicles, {'color': 'white', 'make': '', 'model': '', 'plate': ''})
    assert selected['color'] == 'white'
    assert matched is True
    print('  ✓ selected matching color vehicle')


def test_estimate_passenger_handle():
    print('Testing estimate_passenger_handle()...')
    handle = tool.estimate_passenger_handle([100, 80, 300, 200], 640, 480, side='right')
    assert handle['clock'] in [12, 1, 2, 3, 9, 10, 11]
    assert handle['distance'] in ['very close', 'close', 'near', 'far']
    print(f"  ✓ handle clock={handle['clock']} distance={handle['distance']}")


def test_main_with_stubbed_detection():
    print('Testing main() with stubbed detections...')
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220

    original_detect = tool.detect_vehicles
    try:
        tool.detect_vehicles = lambda _img, _conf=0.35: [
            {
                'class_name': 'car',
                'confidence': 0.8,
                'bbox': [180, 120, 260, 220],
                'center': [310, 230],
                'area': 260 * 220,
                'color': 'white',
            }
        ]

        result = tool.main(image, {'vehicle_color': 'white'})
        text = result['text'] if isinstance(result, dict) else result

        assert 'uber' in text.lower()
        assert 'passenger handle' in text.lower()
        print(f"  ✓ response: {text}")
    finally:
        tool.detect_vehicles = original_detect


def test_streaming_word_limit():
    print('Testing streaming response length...')
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220

    original_detect = tool.detect_vehicles
    try:
        tool.detect_vehicles = lambda _img, _conf=0.35: [
            {
                'class_name': 'car',
                'confidence': 0.8,
                'bbox': [180, 120, 260, 220],
                'center': [310, 230],
                'area': 260 * 220,
                'color': 'white',
            }
        ]
        text = tool.main(image, {'is_streaming': True, 'vehicle_color': 'white'})
        assert isinstance(text, str)
        assert len(text.split()) <= 15
        print(f"  ✓ streaming response ({len(text.split())} words): {text}")
    finally:
        tool.detect_vehicles = original_detect


def test_invalid_image():
    print('Testing invalid image handling...')
    result = tool.main(None, {})
    assert isinstance(result, dict)
    assert result['audio']['type'] == 'error'
    print('  ✓ invalid image returns error audio')


if __name__ == '__main__':
    print('=' * 60)
    print('Find Uber and Door Handle Tool Tests')
    print('=' * 60)
    print()

    test_parse_target_details()
    test_select_target_vehicle_prefers_color_match()
    test_estimate_passenger_handle()
    test_main_with_stubbed_detection()
    test_streaming_word_limit()
    test_invalid_image()

    print()
    print('=' * 60)
    print('All tests completed!')
    print('=' * 60)
