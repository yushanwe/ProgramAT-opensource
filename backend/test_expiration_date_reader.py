"""Test script for expiration_date_reader.py"""

import sys
import os
from datetime import date
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import expiration_date_reader as tool


def test_extract_expiration_date():
    print("Testing extract_expiration_date()...")

    text = """
    ORGANIC MILK
    BEST BY MAY 22, 2026
    KEEP REFRIGERATED
    """
    result = tool.extract_expiration_date(text)
    print(f"  Keyword date: {result}")
    assert result == date(2026, 5, 22)

    text2 = "RX 10431\nEXP 05/22/26\nLOT A11"
    result2 = tool.extract_expiration_date(text2)
    print(f"  Numeric date: {result2}")
    assert result2 == date(2026, 5, 22)

    text3 = "HELLO WORLD\nNO DATE HERE"
    result3 = tool.extract_expiration_date(text3)
    print(f"  No date: {result3}")
    assert result3 is None


def test_main_with_mocked_ocr():
    print("Testing main() with mocked OCR...")

    image = np.ones((200, 200, 3), dtype=np.uint8) * 255

    original = tool.detect_text_google_vision
    try:
        tool.detect_text_google_vision = lambda _img, api_key=None, language_hints=None: [
            {'text': 'MEDICATION\\nEXP 05/22/2026'}
        ]
        result = tool.main(image, {'mode': 'stream'})
        print(f"  Found date: {result}")
        assert isinstance(result, str)
        assert "Expires" in result

        tool.detect_text_google_vision = lambda _img, api_key=None, language_hints=None: [
            {'text': 'NO DATE PRESENT'}
        ]
        result2 = tool.main(image, {})
        print(f"  Missing date: {result2}")
        assert result2 == "No expiration date found."

        tool.detect_text_google_vision = lambda _img, api_key=None, language_hints=None: [
            {'error': 'OCR failed'}
        ]
        result3 = tool.main(image, {})
        print(f"  OCR error: {result3}")
        assert isinstance(result3, dict)
        assert result3.get('audio', {}).get('type') == 'error'
    finally:
        tool.detect_text_google_vision = original


def run_all_tests():
    print("=" * 60)
    print("EXPIRATION DATE READER TEST SUITE")
    print("=" * 60)
    test_extract_expiration_date()
    test_main_with_mocked_ocr()
    print("All expiration date reader tests passed.")


if __name__ == '__main__':
    run_all_tests()
