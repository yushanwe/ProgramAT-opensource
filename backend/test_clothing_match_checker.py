"""
Test script for clothing_match_checker.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import clothing_match_checker as cmc


def create_test_image():
    """Create a simple person-like clothing layout."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (220, 80), (420, 240), (180, 130, 60), -1)   # shirt-like block
    cv2.rectangle(image, (230, 240), (410, 430), (70, 60, 50), -1)    # pants-like block
    return image


def test_limit_words():
    text = "This is a very long sentence that should definitely be cut for streaming mode output."
    shortened = cmc.limit_words(text, max_words=7)
    assert len(shortened.split()) <= 7
    print("✓ limit_words enforces limits")


def test_main_no_image():
    result = cmc.main(None, {})
    assert isinstance(result, dict)
    assert result.get("audio", {}).get("type") == "error"
    print("✓ main handles missing image")


def test_streaming_dedup():
    original = cmc.analyze_clothing_match
    cmc._last_stream_feedback = None
    try:
        cmc.analyze_clothing_match = lambda **kwargs: {
            "success": True,
            "match": "good",
            "shirt": "navy shirt",
            "pants": "black pants",
            "reason": "colors coordinate well",
            "spoken_feedback": "Your shirt and pants match well.",
        }

        image = create_test_image()
        first = cmc.main(image, {"is_streaming": True})
        second = cmc.main(image, {"is_streaming": True})

        assert isinstance(first, dict)
        assert len(first.get("text", "").split()) <= 15
        assert second == ""
        print("✓ streaming mode deduplicates repeated output")
    finally:
        cmc.analyze_clothing_match = original
        cmc._last_stream_feedback = None


def test_audio_type_for_poor_match():
    original = cmc.analyze_clothing_match
    try:
        cmc.analyze_clothing_match = lambda **kwargs: {
            "success": True,
            "match": "poor",
            "shirt": "bright striped shirt",
            "pants": "plaid pants",
            "reason": "patterns clash",
            "spoken_feedback": "Your shirt and pants clash. Consider simpler colors.",
        }

        image = create_test_image()
        result = cmc.main(image, {})
        assert result.get("audio", {}).get("type") == "warning"
        print("✓ poor match uses warning audio")
    finally:
        cmc.analyze_clothing_match = original


def run_all_tests():
    print("=" * 60)
    print("CLOTHING MATCH CHECKER - TEST SUITE")
    print("=" * 60)

    test_limit_words()
    test_main_no_image()
    test_streaming_dedup()
    test_audio_type_for_poor_match()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

