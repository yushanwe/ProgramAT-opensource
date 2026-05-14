"""
Test script for currency_bill_identifier.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import currency_bill_identifier as tool


def create_test_image():
    return np.ones((200, 400, 3), dtype=np.uint8) * 255


def test_identify_usd_denomination():
    print("Testing identify_usd_denomination()...")
    cases = [
        ("UNITED STATES OF AMERICA TWENTY DOLLARS", 20),
        ("FEDERAL RESERVE NOTE FIVE DOLLARS", 5),
        ("ONE DOLLAR WASHINGTON", 1),
        ("hello world", None),
    ]
    for text, expected in cases:
        result = tool.identify_usd_denomination(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result} (expected {expected})")
    print()


def test_streaming_limit():
    print("Testing streaming word limit...")
    long_text = "I couldn't determine the bill value. Try better lighting or a closer view."
    limited = tool.limit_words(long_text, max_words=7)
    print(f"  Limited text: '{limited}'")
    print(f"  Word count: {len(limited.split())} (expected <= 7)")
    print()


def test_main_with_mocked_ocr():
    print("Testing main() with mocked OCR...")

    original_detect = tool.detect_text_google_vision
    image = create_test_image()

    try:
        tool.detect_text_google_vision = lambda image, credentials_hint=None, language_hints=None: [
            {'text': 'UNITED STATES OF AMERICA TWENTY DOLLARS'}
        ]
        result = tool.main(image, {})
        print(f"  One-shot result: {result}")

        streaming_result = tool.main(image, {'is_streaming': True})
        print(f"  Streaming result: {streaming_result}")
    finally:
        tool.detect_text_google_vision = original_detect
    print()


def test_main_error_and_invalid_image():
    print("Testing invalid image and OCR errors...")
    invalid_result = tool.main(None, {})
    print(f"  Invalid image result: {invalid_result}")

    original_detect = tool.detect_text_google_vision
    image = create_test_image()
    try:
        tool.detect_text_google_vision = lambda image, credentials_hint=None, language_hints=None: [
            {'error': 'mock OCR failure'}
        ]
        error_result = tool.main(image, {})
        print(f"  OCR error result: {error_result}")
    finally:
        tool.detect_text_google_vision = original_detect
    print()


def run_all_tests():
    print("=" * 60)
    print("CURRENCY BILL IDENTIFIER TEST SUITE")
    print("=" * 60)
    print()
    test_identify_usd_denomination()
    test_streaming_limit()
    test_main_with_mocked_ocr()
    test_main_error_and_invalid_image()
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
