"""Focused standalone tests for clothing_match_checker.py."""

import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

import clothing_match_checker


def test_prompt_includes_optional_context():
    prompt = clothing_match_checker.build_clothing_match_prompt(
        {'occasion': 'an interview', 'style_goal': 'professional'}
    )
    assert 'an interview' in prompt
    assert 'professional' in prompt
    print('✓ Prompt includes optional context')


def test_no_image_returns_error():
    result = clothing_match_checker.main(None, {})
    assert result['audio']['type'] == 'error'
    assert 'No camera image available' in result['text']
    print('✓ No-image path returns audio-friendly error')


def test_successful_match_result():
    captured = {}

    def fake_route(image, prompt, **kwargs):
        captured['image'] = image
        captured['prompt'] = prompt
        captured['kwargs'] = kwargs
        return 'Your shirt and pants match well'

    original = clothing_match_checker.load_model_router_client
    clothing_match_checker.load_model_router_client = lambda: (RuntimeError, fake_route)
    try:
        result = clothing_match_checker.main(object(), {'occasion': 'work'})
    finally:
        clothing_match_checker.load_model_router_client = original

    assert captured['image'] is not None
    assert 'work' in captured['prompt']
    assert captured['kwargs']['system_instruction'] == clothing_match_checker.DEFAULT_SYSTEM_INSTRUCTION
    assert result['audio']['type'] == 'speech'
    assert result['text'] == 'Your shirt and pants match well.'
    print('✓ Success path returns spoken match feedback')


def test_visibility_warning_result():
    def fake_route(image, prompt, **kwargs):
        return "I can't see enough of your outfit yet. Please show more of your pants"

    original = clothing_match_checker.load_model_router_client
    clothing_match_checker.load_model_router_client = lambda: (RuntimeError, fake_route)
    try:
        result = clothing_match_checker.main(object(), {})
    finally:
        clothing_match_checker.load_model_router_client = original

    assert result['audio']['type'] == 'warning'
    assert result['text'].endswith('.')
    print('✓ Limited-visibility feedback is marked as warning')


def run_all_tests():
    print('=' * 60)
    print('CLOTHING MATCH CHECKER TEST SUITE')
    print('=' * 60)
    test_prompt_includes_optional_context()
    test_no_image_returns_error()
    test_successful_match_result()
    test_visibility_warning_result()
    print('=' * 60)
    print('ALL TESTS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    run_all_tests()
