"""
Test script for mail_categorizer.py.
Run from the repository root:
    python3 backend/test_mail_categorizer.py
"""

import os
import sys

import cv2
import numpy as np

# Make tools/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from mail_categorizer import main


def create_test_image() -> np.ndarray:
    """Create a minimal test frame that looks like a mail photo."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 240  # light-grey background

    # Draw a white rectangle to simulate an envelope
    cv2.rectangle(image, (80, 100), (560, 380), (255, 255, 255), -1)
    cv2.rectangle(image, (80, 100), (560, 380), (180, 180, 180), 2)

    # Simulate printed text lines on the envelope
    cv2.putText(image, "ACME Bank", (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, "Statement of Account", (100, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(image, "Jane Doe", (100, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(image, "123 Main St, Springfield", (100, 315),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    return image


def test_no_image():
    """main(None) should return a friendly error string."""
    print("Testing main(None)...")
    result = main(None)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Result should not be empty"
    print(f"  Result: {result}")
    print("  PASS\n")


def test_with_synthetic_image():
    """main() with a synthetic image should return a non-empty string or dict."""
    print("Testing main() with synthetic mail image...")
    image = create_test_image()
    result = main(image)
    assert result is not None, "Result should not be None"
    assert result != "", "Result should not be empty"
    if isinstance(result, dict):
        assert "audio" in result or "text" in result, (
            "Dict result must contain 'audio' or 'text' key"
        )
        spoken = result.get("text") or result.get("audio", {}).get("text", "")
    else:
        spoken = result
    assert isinstance(spoken, str), f"Spoken output must be a string, got {type(spoken)}"
    spoken_lower = spoken.lower()
    # The tool should produce categorization language
    categorization_keywords = {"important", "junk", "bank", "statement", "mail", "letter", "envelope"}
    matched = [kw for kw in categorization_keywords if kw in spoken_lower]
    assert matched, (
        f"Expected at least one categorization keyword in output, got: {spoken[:200]}"
    )
    print(f"  Result type: {type(result).__name__}")
    print(f"  Matched keywords: {matched}")
    print(f"  Spoken output: {spoken[:300]}")
    print("  PASS\n")


def test_with_input_data():
    """main() should accept and ignore unknown input_data keys gracefully."""
    print("Testing main() with extra input_data keys...")
    image = create_test_image()
    result = main(image, {"unknown_param": True, "another": 42})
    assert result is not None
    assert result != ""
    print(f"  Result type: {type(result).__name__}")
    print("  PASS\n")


def run_all_tests():
    print("=" * 60)
    print("MAIL CATEGORIZER TOOL — TEST SUITE")
    print("=" * 60)
    print()

    test_no_image()
    test_with_synthetic_image()
    test_with_input_data()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
