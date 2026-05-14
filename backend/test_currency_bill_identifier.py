"""
Test script for currency_bill_identifier.py
"""

import sys
import os
import numpy as np
from unittest.mock import patch

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import currency_bill_identifier as tool


def create_test_image():
    """Create simple synthetic image for main() tests."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


def test_normalization():
    print("Testing normalize_ocr_text()...")
    text = "  Twenty\nDollars!!  "
    normalized = tool.normalize_ocr_text(text)
    print(f"  Input: {text!r}")
    print(f"  Output: {normalized!r}")
    assert normalized == "TWENTY DOLLARS"
    print("  ✓ normalization works")
    print()


def test_denomination_matching():
    print("Testing identify_us_bill_denomination()...")

    cases = [
        (["TWENTY DOLLARS", "FEDERAL RESERVE NOTE"], 20),
        (["ONE HUNDRED DOLLARS", "THE UNITED STATES OF AMERICA"], 100),
        (["FIVE DOLLARS"], 5),
        (["HELLO WORLD"], None),
    ]

    for texts, expected in cases:
        value, label, score = tool.identify_us_bill_denomination(texts)
        print(f"  OCR: {texts}")
        print(f"  Detected: value={value}, label={label}, score={score}")
        assert value == expected, f"Expected {expected}, got {value}"

    print("  ✓ denomination matching works")
    print()


def test_main_with_mocked_ocr():
    print("Testing main() with mocked OCR...")
    image = create_test_image()

    with patch.object(tool, 'VISION_API_AVAILABLE', True):
        with patch.object(tool, 'detect_text_google_vision', return_value=["TEN DOLLARS", "FEDERAL RESERVE NOTE"]):
            result = tool.main(image, {})
            print(f"  Result: {result}")
            assert isinstance(result, str)
            assert "10 dollars" in result.lower()

        with patch.object(tool, 'detect_text_google_vision', return_value=["RANDOM TEXT ONLY"]):
            result_unknown = tool.main(image, {})
            print(f"  Unknown result: {result_unknown}")
            assert "could not identify" in result_unknown.lower()

        with patch.object(tool, 'detect_text_google_vision', return_value=[]):
            result_no_text = tool.main(image, {})
            print(f"  No-text result: {result_no_text}")
            assert "no bill text" in result_no_text.lower()

    print("  ✓ main() behavior works with mocked OCR")
    print()


def test_main_no_image():
    print("Testing main() no-image handling...")
    result = tool.main(None, {})
    print(f"  Result: {result}")
    assert isinstance(result, dict)
    assert result.get('audio', {}).get('type') == 'error'
    print("  ✓ no-image handling works")
    print()


def run_all_tests():
    print("=" * 60)
    print("CURRENCY BILL IDENTIFIER TOOL - TEST SUITE")
    print("=" * 60)
    print()

    test_normalization()
    test_denomination_matching()
    test_main_with_mocked_ocr()
    test_main_no_image()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
