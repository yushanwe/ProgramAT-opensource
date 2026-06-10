"""Test script for tools/clothing_matching_checker.py."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from clothing_matching_checker import _limit_words, evaluate_match, main


def create_test_outfit(top_bgr: tuple[int, int, int], bottom_bgr: tuple[int, int, int]) -> np.ndarray:
    image = np.ones((480, 360, 3), dtype=np.uint8) * 240
    # Rectangles align with extraction bands: upper 18%-52%, lower 56%-92%.
    cv2.rectangle(image, (90, 90), (270, 250), top_bgr, -1)
    cv2.rectangle(image, (90, 280), (270, 450), bottom_bgr, -1)
    return image


def test_evaluate_match_rules():
    print('Testing evaluate_match() rules...')

    text, level = evaluate_match('blue', 'black')
    assert level == 'success'
    assert 'match' in text.lower() or 'safe' in text.lower()

    text, level = evaluate_match('red', 'green')
    assert level == 'success'

    text, level = evaluate_match('red', 'cyan')
    assert level == 'warning'

    print('  ✓ evaluate_match rules pass')


def test_main_matching_case():
    print('Testing main() with matching outfit...')
    image = create_test_outfit((255, 0, 0), (0, 0, 0))  # BGR: blue top, black pants
    result = main(image, {'is_streaming': False})
    assert isinstance(result, dict)
    assert 'audio' in result
    assert result['audio']['type'] in {'success', 'speech'}
    print(f"  Output: {result['text']}")


def test_main_streaming_word_limit():
    print('Testing streaming word limit...')
    image = create_test_outfit((0, 255, 255), (255, 255, 255))  # yellow + white
    result = main(image, {'is_streaming': True})
    assert isinstance(result, dict)
    assert len(result.get('text', '').split()) <= 15
    print(f"  Streaming output: {result['text']}")


def test_limit_words_truncates():
    print('Testing _limit_words() truncation...')
    long_text = 'This is a long sentence that should definitely be trimmed for streaming output right now please.'
    trimmed = _limit_words(long_text, max_words=15)
    assert trimmed != long_text
    assert len(trimmed.split()) <= 15
    print(f'  Trimmed output: {trimmed}')


def test_main_no_image():
    print('Testing no-image error handling...')
    result = main(None)
    assert isinstance(result, dict)
    assert result['audio']['type'] == 'error'
    print('  ✓ no-image handling pass')


def run_all_tests():
    print('=' * 60)
    print('CLOTHING MATCHING CHECKER - TEST SUITE')
    print('=' * 60)
    test_evaluate_match_rules()
    test_main_matching_case()
    test_main_streaming_word_limit()
    test_limit_words_truncates()
    test_main_no_image()
    print('=' * 60)
    print('ALL TESTS COMPLETED')
    print('=' * 60)


if __name__ == '__main__':
    run_all_tests()
