"""Test script for uber_passenger_door_handle.py"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from uber_passenger_door_handle import (
    build_target_description,
    build_analysis_prompt,
    parse_analysis_json,
    enforce_word_limit,
    create_audio_response,
    main,
)


def test_target_description():
    print('Testing build_target_description()...')
    data = {'vehicle_color': 'white', 'vehicle_make': 'Toyota', 'vehicle_model': 'Camry', 'plate': 'ABC123'}
    result = build_target_description(data)
    print(f"  Result: '{result}'")
    assert 'white Toyota Camry' in result
    assert 'ABC123' in result


def test_prompt_streaming_limit_rule():
    print('Testing build_analysis_prompt()...')
    prompt_stream = build_analysis_prompt('white Toyota Camry', True)
    prompt_single = build_analysis_prompt('white Toyota Camry', False)
    assert '15 words max' in prompt_stream
    assert '15 words max' not in prompt_single
    print('  ✓ Prompt includes streaming-specific word limit rule')


def test_parse_analysis_json():
    print('Testing parse_analysis_json()...')
    raw = '{"target_found": true, "vehicle_description": "white Toyota", "vehicle_clock": "2 o\'clock", "handle_visible": true, "handle_direction": "slightly right of center", "navigation_text": "Your Uber is at 2 o\'clock. Handle slightly right of center."}'
    parsed = parse_analysis_json(raw)
    assert parsed['target_found'] is True
    assert parsed['handle_visible'] is True
    print(f"  Parsed vehicle clock: {parsed['vehicle_clock']}")


def test_word_limit():
    print('Testing enforce_word_limit()...')
    long_text = 'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen'
    limited = enforce_word_limit(long_text)
    count = len(limited.split())
    print(f'  Word count after limit: {count}')
    assert count == 15


def test_response_creation():
    print('Testing create_audio_response()...')

    missing_target = {
        'target_found': False,
        'vehicle_description': '',
        'vehicle_clock': '',
        'handle_visible': False,
        'handle_direction': '',
        'navigation_text': '',
    }
    response = create_audio_response(missing_target, 'white Toyota Camry', False)
    assert response['audio']['type'] == 'speech'
    print(f"  Missing target response: '{response['text']}'")

    found_handle = {
        'target_found': True,
        'vehicle_description': 'white Toyota Camry',
        'vehicle_clock': "2 o'clock",
        'handle_visible': True,
        'handle_direction': 'slightly right of center',
        'navigation_text': "Your Uber is at 2 o'clock. Handle slightly right of center.",
    }
    response2 = create_audio_response(found_handle, 'white Toyota Camry', True)
    assert response2['audio']['type'] == 'success'
    assert len(response2['text'].split()) <= 15
    print(f"  Found handle response: '{response2['text']}'")


def test_main_invalid_image():
    print('Testing main() invalid image handling...')
    result = main(None, {})
    assert result['audio']['type'] == 'error'
    print(f"  Result: '{result['text']}'")


def run_all_tests():
    print('=' * 60)
    print('Uber Passenger Door Handle Tool - Test Suite')
    print('=' * 60)

    test_target_description()
    test_prompt_streaming_limit_rule()
    test_parse_analysis_json()
    test_word_limit()
    test_response_creation()
    test_main_invalid_image()

    print('=' * 60)
    print('All tests completed!')
    print('=' * 60)


if __name__ == '__main__':
    run_all_tests()
