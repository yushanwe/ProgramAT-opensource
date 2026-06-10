"""
Test script for clothing_match_checker.py
"""

import os
import sys

import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import clothing_match_checker
from clothing_match_checker import (
    build_match_prompt,
    convert_cv2_to_pil,
    format_for_audio,
    main,
    resize_image_if_needed,
)


def create_test_image():
    """Create synthetic top/bottom clothing image."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (150, 40), (500, 230), (210, 130, 70), -1)  # top
    cv2.rectangle(image, (170, 240), (490, 470), (160, 90, 50), -1)  # pants
    return image


def test_resize_image():
    print("Testing resize_image_if_needed()...")
    image = np.ones((1800, 2200, 3), dtype=np.uint8)
    resized = resize_image_if_needed(image, max_size=(1024, 1024))
    assert resized.shape[0] <= 1024 and resized.shape[1] <= 1024
    print(f"  Resized to: {resized.shape}")


def test_convert_cv2_to_pil():
    print("Testing convert_cv2_to_pil()...")
    image = create_test_image()
    pil_image = convert_cv2_to_pil(image)
    assert pil_image.size == (640, 480)
    print(f"  PIL size: {pil_image.size}")


def test_build_prompt():
    print("Testing build_match_prompt()...")
    stream_prompt = build_match_prompt(is_streaming=True)
    one_shot_prompt = build_match_prompt(is_streaming=False)
    assert "15 words" in stream_prompt
    assert "one short sentence" in one_shot_prompt
    print("  Prompt variants generated")


def test_format_for_audio():
    print("Testing format_for_audio()...")
    text = "Your bright red patterned shirt and neon green striped pants do not match very well today"
    limited = format_for_audio(text, max_words=15)
    assert len(limited.split()) <= 15
    full = format_for_audio(text)
    assert len(full.split()) == len(text.split())
    print(f"  Streaming text: '{limited}'")


def test_main_no_image():
    print("Testing main() with no image...")
    result = main(None)
    assert isinstance(result, dict)
    assert result.get('audio', {}).get('type') == 'error'
    print(f"  Result: {result.get('text')}")


def test_main_streaming_behavior():
    print("Testing main() streaming word limit and dedup...")
    image = create_test_image()

    original = clothing_match_checker.analyze_clothing_match

    def fake_analysis(*args, **kwargs):
        return {
            'success': True,
            'message': 'Your bright red patterned shirt and neon green striped pants do not match very well today',
        }

    clothing_match_checker.analyze_clothing_match = fake_analysis
    clothing_match_checker._last_stream_message = None

    try:
        first = main(image, {'is_streaming': True})
        assert isinstance(first, dict)
        assert len(first.get('text', '').split()) <= 15

        second = main(image, {'is_streaming': True})
        assert second == ""

        one_shot = main(image, {'is_streaming': False})
        assert isinstance(one_shot, dict)
        assert len(one_shot.get('text', '').split()) > 15
        print(f"  First streaming: {first.get('text')}")
        print("  Repeated streaming frame correctly silenced")
    finally:
        clothing_match_checker.analyze_clothing_match = original
        clothing_match_checker._last_stream_message = None


def run_all_tests():
    print("=" * 60)
    print("CLOTHING MATCH CHECKER TOOL - TEST SUITE")
    print("=" * 60)

    test_resize_image()
    test_convert_cv2_to_pil()
    test_build_prompt()
    test_format_for_audio()
    test_main_no_image()
    test_main_streaming_behavior()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
