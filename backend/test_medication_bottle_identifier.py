"""
Test script for medication_bottle_identifier.py
"""

import os
import sys

import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

import medication_bottle_identifier as tool
from medication_bottle_identifier import (
    extract_medication_name,
    is_expected_medication,
    main,
    reset_tracking,
)


def create_test_image() -> np.ndarray:
    """Create a simple synthetic image for non-vision logic tests."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 200


def test_extract_medication_name() -> None:
    print("Testing extract_medication_name()...")

    ocr_text = """RX 12345
Metformin 500mg
Take one tablet daily
Refill 2"""
    result = extract_medication_name(ocr_text)
    print(f"  Result: {result}")
    assert "metformin" in result.lower(), "Should extract medication name line"


def test_is_expected_medication() -> None:
    print("Testing is_expected_medication()...")

    assert is_expected_medication("Metformin 500mg", "metformin")
    assert is_expected_medication("Lisinopril 10 mg", "lisinopril")
    assert not is_expected_medication("Ibuprofen 200mg", "metformin")

    print("  ✓ Matching logic works for positive and negative cases")


def test_main_match_and_mismatch() -> None:
    print("Testing main() match and mismatch responses...")

    image = create_test_image()

    original_detect = tool.detect_bottle_presence
    original_ocr = tool.extract_text_google_vision

    try:
        tool.detect_bottle_presence = lambda *args, **kwargs: [{"class_name": "bottle", "confidence": 0.9}]

        # Matching medication
        tool.extract_text_google_vision = lambda *args, **kwargs: ("Metformin 500mg", None)
        reset_tracking()
        result_match = main(image, {"expected_medication": "Metformin", "track_mode": False})
        print(f"  Match result: {result_match}")
        assert isinstance(result_match, str)
        assert "yes" in result_match.lower()

        # Mismatched medication
        tool.extract_text_google_vision = lambda *args, **kwargs: ("Ibuprofen 200mg", None)
        result_mismatch = main(image, {"expected_medication": "Metformin", "track_mode": False})
        print(f"  Mismatch result: {result_mismatch}")
        assert isinstance(result_mismatch, dict)
        assert result_mismatch.get("audio", {}).get("type") == "warning"

    finally:
        tool.detect_bottle_presence = original_detect
        tool.extract_text_google_vision = original_ocr


def test_streaming_limit() -> None:
    print("Testing streaming 15-word limit...")

    image = create_test_image()

    original_detect = tool.detect_bottle_presence
    original_ocr = tool.extract_text_google_vision

    try:
        tool.detect_bottle_presence = lambda *args, **kwargs: [{"class_name": "bottle", "confidence": 0.9}]
        tool.extract_text_google_vision = lambda *args, **kwargs: (
            "Acetaminophen Extended Release Extra Strength 650mg Caplets", None
        )

        reset_tracking()
        result = main(
            image,
            {
                "expected_medication": "Metformin",
                "is_streaming": True,
                "track_mode": False,
            },
        )

        if isinstance(result, dict):
            spoken = result.get("audio", {}).get("text", "")
        else:
            spoken = result

        word_count = len(spoken.split())
        print(f"  Streaming text: {spoken}")
        print(f"  Word count: {word_count}")

        assert word_count <= 15, "Streaming response should be capped at 15 words"

    finally:
        tool.detect_bottle_presence = original_detect
        tool.extract_text_google_vision = original_ocr


def run_all_tests() -> None:
    print("=" * 60)
    print("MEDICATION BOTTLE IDENTIFIER TEST SUITE")
    print("=" * 60)

    test_extract_medication_name()
    test_is_expected_medication()
    test_main_match_and_mismatch()
    test_streaming_limit()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
