"""
Test script for weather_detection.py
"""

import os
import sys

import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from weather_detection import (
    WEATHER_LABELS,
    build_weather_prompt,
    estimate_weather_from_pixels,
    format_weather_for_audio,
    main,
)


def create_bright_test_image():
    """Create a bright outdoor-like synthetic image."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220
    cv2.rectangle(image, (0, 0), (640, 200), (255, 180, 120), -1)
    return image


def test_prompt_generation():
    prompt_stream = build_weather_prompt(streaming=True)
    prompt_single = build_weather_prompt(streaming=False)
    assert 'one word' in prompt_stream.lower()
    assert 'umbrella' in prompt_single.lower()


def test_pixel_fallback_label():
    image = create_bright_test_image()
    label = estimate_weather_from_pixels(image)
    assert label in WEATHER_LABELS


def test_streaming_audio_length():
    text = format_weather_for_audio('rainy', streaming=True)
    assert len(text.split()) <= 15
    assert 'umbrella' in text.lower()


def test_main_none_image_error_dict():
    result = main(None, {})
    assert isinstance(result, dict)
    assert result.get('audio', {}).get('type') == 'error'


def test_main_without_api_key_uses_fallback():
    image = create_bright_test_image()
    old_key = os.environ.pop('GEMINI_API_KEY', None)
    try:
        result = main(image, {'stream_mode': True})
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        assert len(result.split()) <= 15
    finally:
        if old_key:
            os.environ['GEMINI_API_KEY'] = old_key


def run_all_tests():
    test_prompt_generation()
    test_pixel_fallback_label()
    test_streaming_audio_length()
    test_main_none_image_error_dict()
    test_main_without_api_key_uses_fallback()
    print('All weather_detection tests passed')


if __name__ == '__main__':
    run_all_tests()
