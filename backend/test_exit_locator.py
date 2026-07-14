"""
Test script for exit_locator.py
Tests the exit locator tool with sample data
"""

import sys
import os
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from exit_locator import main, TOOL_PROMPT


def create_test_image():
    """Create a simple synthetic indoor test image (corridor / hallway style)."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # light grey background

    # Draw a door-like rectangle at 12 o'clock (center top area)
    image[100:380, 270:370] = [80, 60, 40]  # dark brown door

    # Draw a green exit sign above the door
    image[80:100, 265:375] = [0, 200, 0]  # green bar (exit sign color)

    return image


def test_tool_prompt_defined():
    """TOOL_PROMPT must be a non-empty string."""
    print("Testing TOOL_PROMPT...")
    assert isinstance(TOOL_PROMPT, str), "TOOL_PROMPT should be a string"
    assert len(TOOL_PROMPT) > 0, "TOOL_PROMPT should not be empty"
    print(f"  TOOL_PROMPT length: {len(TOOL_PROMPT)} chars")
    print(f"  First 100 chars: {TOOL_PROMPT[:100]}...")
    print()


def test_none_image_returns_fallback():
    """main(None) must return a non-empty, audio-friendly fallback string."""
    print("Testing main() with None image...")
    result = main(None)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Fallback message should not be empty"
    print(f"  Result: '{result}'")
    print()


def test_main_returns_string_type():
    """main() with a valid image must return a string (or dict with 'audio'/'text' keys)."""
    print("Testing main() return type with synthetic image...")
    image = create_test_image()

    # Patch call_take_photo_baseline_vlm on the exit_locator module so the
    # test doesn't hit the network.  The function is bound directly in the
    # exit_locator namespace via "from litellm_utils import ...", so we must
    # patch it there.
    import exit_locator as _module

    original_fn = _module.call_take_photo_baseline_vlm

    def _mock_vlm(image, prompt):
        return "Exit door is at 12 o'clock. Walk straight ahead about 10 steps to reach the exit."

    _module.call_take_photo_baseline_vlm = _mock_vlm
    try:
        result = main(image)
    finally:
        _module.call_take_photo_baseline_vlm = original_fn

    assert isinstance(result, (str, dict)), f"Expected str or dict, got {type(result)}"
    if isinstance(result, dict):
        assert 'audio' in result or 'text' in result, "dict result must have 'audio' or 'text' key"
    else:
        assert len(result) > 0, "String result should not be empty"
    print(f"  Result type: {type(result).__name__}")
    print(f"  Result: '{result}'")
    print()


def test_live_api(api_key_name='GEMINI_API_KEY'):
    """Integration test — skipped when no API key is present."""
    print("Testing main() with live VLM API...")
    env_configured = bool(os.environ.get(api_key_name))
    if not env_configured:
        print(f"  ⚠️  {api_key_name} not set — skipping live API test")
        print("  Set the environment variable to run an end-to-end test.")
        print()
        return

    image = create_test_image()
    result = main(image)
    assert isinstance(result, (str, dict)), f"Expected str or dict, got {type(result)}"
    print(f"  Live result: '{result}'")
    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("EXIT LOCATOR TOOL - TEST SUITE")
    print("=" * 60)
    print()

    test_tool_prompt_defined()
    test_none_image_returns_fallback()
    test_main_returns_string_type()
    test_live_api()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    try:
        run_all_tests()
    except Exception as exc:
        print(f"\n❌ Test failed with error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
