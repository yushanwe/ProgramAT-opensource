"""
Test script for hand_gesture_mapping.py.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

stub_router_client = types.ModuleType("model_router_client")
stub_router_client.copilot_llm_call = lambda **kwargs: {
    "response": "",
    "artifact": None,
    "capability": "general_reasoning",
}
sys.modules["model_router_client"] = stub_router_client

import hand_gesture_mapping as tool


def create_test_image():
    """Create a simple synthetic frame with a hand-like shape."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    image[180:340, 260:380] = (160, 190, 220)
    image[100:220, 305:335] = (160, 190, 220)
    return image


def test_extract_conversation_context():
    print("Testing extract_conversation_context()...")
    context = tool.extract_conversation_context({"spoken_text": "That sounds great"})
    print(f"  Extracted: {context}")
    assert context == "That sounds great"
    print()


def test_build_gesture_prompt():
    print("Testing build_gesture_prompt()...")
    prompt = tool.build_gesture_prompt("This is exciting")
    print(f"  Prompt length: {len(prompt)}")
    assert "best guess" in prompt
    assert "This is exciting" in prompt
    print()


def test_format_response():
    print("Testing format_response()...")
    formatted = tool.format_response("  Likely thumbs up  ")
    print(f"  Formatted: {formatted}")
    assert formatted == "Likely thumbs up."
    fallback = tool.format_response("")
    print(f"  Fallback: {fallback}")
    assert "could not confidently identify" in fallback
    print()


def test_main_no_image():
    print("Testing main() with no image...")
    result = tool.main(None, {})
    print(f"  Result: {result}")
    assert isinstance(result, dict)
    assert result["audio"]["type"] == "error"
    print()


def test_main_with_mocked_router():
    print("Testing main() with mocked router...")
    image = create_test_image()
    captured = {}
    original_call = tool.copilot_llm_call

    def fake_call(**kwargs):
        captured.update(kwargs)
        return {
            "response": "Likely thumbs up. The tone seems positive and encouraging.",
            "artifact": {"gesture": "thumbs up"},
            "capability": "general_reasoning",
        }

    tool.copilot_llm_call = fake_call
    try:
        result = tool.main(image, {"spoken_text": "I really like this"})
    finally:
        tool.copilot_llm_call = original_call

    print(f"  Result: {result}")
    assert result == "Likely thumbs up. The tone seems positive and encouraging."
    assert captured["capability"] == "general_reasoning"
    assert "I really like this" in captured["messages"][1]["content"]
    assert captured["images"][0].startswith("data:image/jpeg;base64,")
    print()


def run_all_tests():
    print("=" * 60)
    print("HAND GESTURE MAPPING TOOL - TEST SUITE")
    print("=" * 60)
    print()

    test_extract_conversation_context()
    test_build_gesture_prompt()
    test_format_response()
    test_main_no_image()
    test_main_with_mocked_router()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
