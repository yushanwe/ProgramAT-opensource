"""
Test script for color_detection.py
Tests center color detection behavior with synthetic images.
"""

import os
import sys

import cv2
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from color_detection import detect_center_color, main


def create_test_image(center_bgr, width: int = 640, height: int = 480):
    """Create image with a colored center object on white background."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    x1, y1 = width // 3, height // 3
    x2, y2 = 2 * width // 3, 2 * height // 3
    cv2.rectangle(image, (x1, y1), (x2, y2), center_bgr, -1)

    return image


def test_center_color_detection():
    """Verify center color detection for common colors."""
    print("Testing detect_center_color()...")

    cases = [
        ((255, 0, 0), "blue"),
        ((0, 255, 0), "green"),
        ((0, 0, 255), "red"),
        ((0, 255, 255), "yellow"),
    ]

    for bgr, expected in cases:
        image = create_test_image(bgr)
        result = detect_center_color(image)
        print(f"  BGR {bgr} -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print()


def test_main_function():
    """Verify tool main returns audio-friendly short color output."""
    print("Testing main()...")

    blue_image = create_test_image((255, 0, 0))
    result = main(blue_image, {})
    print(f"  main() result: {result}")
    assert result == "blue", f"Expected 'blue', got '{result}'"

    none_result = main(None, {})
    print(f"  None image result: {none_result}")
    assert none_result == "No camera image available"

    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Color Detection Tool Test Suite")
    print("=" * 60)
    print()

    test_center_color_detection()
    test_main_function()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
