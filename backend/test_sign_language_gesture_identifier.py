"""
Test script for sign_language_gesture_identifier.py
Tests the sign language gesture identifier tool with sample data
"""

import sys
import os
import unittest
from unittest.mock import patch
import numpy as np

# Add backend first so litellm_utils resolves to the backend module, then tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import sign_language_gesture_identifier as tool


class TestSignLanguageGestureIdentifier(unittest.TestCase):

    def _make_image(self):
        """Return a small valid BGR image."""
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def test_constants(self):
        self.assertEqual(tool.TOOL_NAME, "sign_language_gesture_identifier")
        self.assertEqual(tool.EXECUTION_MODE, "take_photo")
        self.assertIsInstance(tool.TOOL_PROMPT, str)
        self.assertGreater(len(tool.TOOL_PROMPT), 0)

    def test_none_image_returns_fallback(self):
        result = tool.main(None, {})
        self.assertIn("No camera image", result)

    def test_main_calls_vlm_once(self):
        image = self._make_image()
        with patch(
            "sign_language_gesture_identifier.call_take_photo_vlm",
            return_value="Thumbs up",
        ) as mock_vlm:
            result = tool.main(image, {})

        mock_vlm.assert_called_once_with(
            image=image,
            prompt=tool.TOOL_PROMPT,
            tool_name=tool.TOOL_NAME,
        )
        self.assertEqual(result, "Thumbs up")

    def test_main_returns_vlm_response(self):
        image = self._make_image()
        expected = "Letter A in ASL"
        with patch(
            "sign_language_gesture_identifier.call_take_photo_vlm",
            return_value=expected,
        ):
            result = tool.main(image, {})
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
