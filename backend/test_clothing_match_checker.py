"""Standalone tests for tools/clothing_match_checker.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import clothing_match_checker  # noqa: E402


class ClothingMatchCheckerTests(unittest.TestCase):
    def test_no_image_returns_error_audio(self):
        result = clothing_match_checker.main(None, {})

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('audio', {}).get('type'), 'error')
        self.assertIn('No camera image available', result.get('text', ''))

    def test_successful_match_response(self):
        original = clothing_match_checker.run_visual_understanding

        def fake_router(image, prompt, input_data):
            self.assertIsNotNone(image)
            self.assertIn('match', prompt.lower())
            return {'success': True, 'text': 'Your shirt and pants match well with balanced colors.'}

        clothing_match_checker.run_visual_understanding = fake_router
        try:
            result = clothing_match_checker.main(image=object(), input_data={})
        finally:
            clothing_match_checker.run_visual_understanding = original

        self.assertEqual(result.get('audio', {}).get('type'), 'speech')
        self.assertIn('match well', result.get('text', '').lower())

    def test_router_failure_returns_error_audio(self):
        original = clothing_match_checker.run_visual_understanding

        def fake_router(image, prompt, input_data):
            return {'success': False, 'text': 'Model services are unavailable right now.', 'error': 'timeout'}

        clothing_match_checker.run_visual_understanding = fake_router
        try:
            result = clothing_match_checker.main(image=object(), input_data={})
        finally:
            clothing_match_checker.run_visual_understanding = original

        self.assertEqual(result.get('audio', {}).get('type'), 'error')
        self.assertEqual(result.get('text'), 'timeout')


if __name__ == '__main__':
    unittest.main()
