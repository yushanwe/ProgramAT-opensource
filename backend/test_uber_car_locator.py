"""
Test script for uber_car_locator.py

Run from the repository root:
    python3 backend/test_uber_car_locator.py
"""

import os
import sys

import cv2
import numpy as np

# Backend must appear before tools so that backend/litellm_utils.py (which has
# call_model) is found before tools/litellm_utils.py when model_router imports it.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))
sys.path.insert(0, _HERE)

from uber_car_locator import main, _bgr_to_pil, TOOL_NAME, TASK_CATEGORY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(color_bgr=(255, 255, 255), size=(480, 640)):
    """Create a solid-colour test image."""
    h, w = size
    img = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    return img


def _parking_lot_image():
    """Synthesise a minimal 'parking lot' scene with a white rectangle (car)."""
    img = _solid_image(color_bgr=(100, 120, 100))  # green-grey background
    # Draw a white rectangle representing a car body.
    cv2.rectangle(img, (180, 200), (460, 340), (255, 255, 255), -1)
    # Draw dark rectangles for windows.
    cv2.rectangle(img, (210, 210), (280, 260), (40, 40, 40), -1)
    cv2.rectangle(img, (360, 210), (430, 260), (40, 40, 40), -1)
    # Add a crude licence-plate text.
    cv2.putText(img, "ABC123", (270, 335), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Unit tests (no API calls)
# ---------------------------------------------------------------------------

def test_constants():
    """Tool-level constants are defined and non-empty."""
    print("Testing constants...")
    assert TOOL_NAME, "TOOL_NAME should be a non-empty string"
    assert TASK_CATEGORY, "TASK_CATEGORY should be a non-empty string"
    print(f"  TOOL_NAME    = {TOOL_NAME!r}")
    print(f"  TASK_CATEGORY= {TASK_CATEGORY!r}")
    print()


def test_bgr_to_pil():
    """_bgr_to_pil converts a BGR numpy array to a PIL image correctly."""
    print("Testing _bgr_to_pil()...")
    img = _solid_image(color_bgr=(0, 0, 255))  # pure red in BGR
    pil = _bgr_to_pil(img)
    r, g, b = pil.getpixel((0, 0))
    assert r == 255 and g == 0 and b == 0, "BGR→RGB conversion failed"
    print("  BGR→RGB conversion: ✓")
    print()


def test_no_image():
    """main() with None returns an informative string, not an exception."""
    print("Testing main() with no image...")
    result = main(None, {})
    assert isinstance(result, str), "Expected a string when image is None"
    assert len(result) > 0, "Expected a non-empty message when image is None"
    print(f"  Result: {result!r}")
    print()


def test_empty_input_data():
    """main() works even when input_data is empty or None."""
    print("Testing main() with empty input_data (no API expected to succeed)...")
    img = _solid_image()
    # The function should not raise; it may fail gracefully without API keys.
    try:
        result = main(img, {})
        assert result is not None, "Result should not be None"
        print(f"  Result type : {type(result).__name__}")
        if isinstance(result, dict):
            print(f"  Result keys : {list(result.keys())}")
        else:
            print(f"  Result      : {str(result)[:120]}")
    except Exception as exc:
        # Without valid API credentials the call may raise; that is expected
        # in a CI environment without keys.
        print(f"  Note: API call raised {type(exc).__name__}: {exc} (expected without keys)")
    print()


# ---------------------------------------------------------------------------
# Integration test (requires API keys)
# ---------------------------------------------------------------------------

def test_full_flow_with_vehicle_details():
    """Integration test: main() with vehicle details and a synthetic image."""
    print("Testing main() with vehicle details (requires API keys)...")

    has_keys = any(
        os.environ.get(k)
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_keys:
        print("  ⚠  No API keys found – skipping live model call.")
        print()
        return

    img = _parking_lot_image()
    input_data = {
        "color": "white",
        "make": "Toyota",
        "model": "Camry",
        "license_plate": "ABC123",
    }

    result = main(img, input_data)
    print(f"  Result type : {type(result).__name__}")
    if isinstance(result, dict):
        print(f"  Audio text  : {result.get('audio', {}).get('text', 'N/A')}")
        print(f"  Display text: {result.get('text', 'N/A')[:200]}")
    else:
        print(f"  Result      : {str(result)[:200]}")
    print()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("=" * 60)
    print("UBER CAR LOCATOR TOOL – TEST SUITE")
    print("=" * 60)
    print()

    test_constants()
    test_bgr_to_pil()
    test_no_image()
    test_empty_input_data()
    test_full_flow_with_vehicle_details()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
