"""
Test script for clothing_matching_assistant.py
"""

import os
import sys

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import clothing_matching_assistant


def create_test_image(width=10, height=10):
    return [[[0, 0, 0] for _ in range(width)] for _ in range(height)]


def test_build_prompt():
    print("Testing build_matching_prompt()...")
    prompt = clothing_matching_assistant.build_matching_prompt(
        occasion='formal',
        include_suggestion=True,
    )
    assert 'formal outfit' in prompt
    assert 'If they do not match' in prompt
    assert 'not visible enough' in prompt
    print("  ✓ Prompt includes matching, visibility, and suggestion guidance")
    print()


def test_clean_response():
    print("Testing clean_response()...")
    cleaned = clothing_matching_assistant.clean_response(
        "  Your shirt and pants match well! Both are dark neutrals. Extra detail here.  "
    )
    assert cleaned == "Your shirt and pants match well. Both are dark neutrals."
    print(f"  ✓ Cleaned response: {cleaned}")
    print()


def test_main_no_image():
    print("Testing main() with no image...")
    result = clothing_matching_assistant.main(None)
    assert result['audio']['type'] == 'error'
    assert 'need a photo' in result['text']
    print(f"  ✓ Error response: {result['text']}")
    print()


def test_main_success():
    print("Testing main() success path...")
    calls = []
    fake_image = create_test_image()

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return "Your shirt and pants match well. Both are dark neutrals."

    original = clothing_matching_assistant.llm_call
    clothing_matching_assistant.llm_call = fake_llm_call
    try:
        result = clothing_matching_assistant.main(fake_image, {'occasion': 'work'})
    finally:
        clothing_matching_assistant.llm_call = original

    assert result == "Your shirt and pants match well. Both are dark neutrals."
    assert calls[0]['task_category'] == 'visual_reasoning'
    assert 'work outfit' in calls[0]['prompt']
    print(f"  ✓ Success response: {result}")
    print()


def test_main_failure():
    print("Testing main() failure path...")

    def fake_llm_call(**kwargs):
        raise RuntimeError('boom')

    original = clothing_matching_assistant.llm_call
    clothing_matching_assistant.llm_call = fake_llm_call
    try:
        result = clothing_matching_assistant.main(object())
    finally:
        clothing_matching_assistant.llm_call = original

    assert result['audio']['type'] == 'error'
    assert 'could not check' in result['text']
    assert result['debug'] == 'boom'
    print(f"  ✓ Failure response: {result['text']}")
    print()


def run_all_tests():
    print("=" * 60)
    print("CLOTHING MATCHING ASSISTANT - TEST SUITE")
    print("=" * 60)
    print()

    test_build_prompt()
    test_clean_response()
    test_main_no_image()
    test_main_success()
    test_main_failure()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
