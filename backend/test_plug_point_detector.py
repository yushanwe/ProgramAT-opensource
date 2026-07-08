"""
Test script for plug_point_detector.py
Tests the plug point detection tool with synthetic images.
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Insert backend dir first so model_router's litellm_utils resolves correctly,
# then tools dir so the tool module itself can be imported.
_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'backend'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

# Stub out model_router_client before the tool module is imported so that no
# live model or API key is needed during unit tests.
_mock_router = MagicMock()
sys.modules.setdefault('model_router_client', _mock_router)

from plug_point_detector import main, _truncate_to_words, _image_to_data_uri  # noqa: E402


def create_blank_wall_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a plain grey image representing a blank wall (no outlets)."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    return image


def create_outlet_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with a white rectangle simulating an outlet."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 180
    # Draw an outlet-like rectangle slightly left of centre
    cv2_x, cv2_y = 260, 200
    cv2_w, cv2_h = 60, 90
    image[cv2_y:cv2_y + cv2_h, cv2_x:cv2_x + cv2_w] = [255, 255, 255]
    return image


def create_two_outlet_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with two white rectangles simulating two outlets."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 180
    # Left outlet
    image[200:290, 100:160] = [255, 255, 255]
    # Right outlet (slightly larger → nearer)
    image[190:300, 480:560] = [240, 240, 240]
    return image


class TestTruncateToWords(unittest.TestCase):
    def test_short_text_unchanged(self):
        text = "One outlet at 12 o'clock."
        self.assertEqual(_truncate_to_words(text, 15), text)

    def test_long_text_truncated(self):
        text = " ".join(["word"] * 20)
        result = _truncate_to_words(text, 15)
        self.assertEqual(len(result.split()), 15)

    def test_exactly_limit(self):
        text = " ".join(["word"] * 15)
        self.assertEqual(_truncate_to_words(text, 15), text)


class TestImageToDataUri(unittest.TestCase):
    def test_returns_data_uri(self):
        image = create_blank_wall_image()
        uri = _image_to_data_uri(image)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(uri), 50)


class TestMainNoImage(unittest.TestCase):
    def test_none_image_returns_string(self):
        result = main(None, {})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_empty_image_returns_string(self):
        empty = np.array([])
        result = main(empty, {})
        self.assertIsInstance(result, str)


class TestMainWithMockedRouter(unittest.TestCase):
    """Test main() with mocked copilot_llm_call to avoid needing real API keys."""

    def _mock_call(self, capability, **kwargs):
        """Return canned responses based on capability."""
        if capability == "object_detection_localization":
            return {
                "response": "One electrical outlet detected near the centre-left.",
                "artifact": {"count": 1, "labels": ["electrical outlet"]},
                "implementation": "mock",
                "capability": capability,
            }
        if capability == "spatial_reasoning":
            return {
                "response": "I see one outlet at 11 o'clock.",
                "artifact": {},
                "implementation": "mock",
                "capability": capability,
            }
        return {"response": "", "artifact": {}, "implementation": "mock", "capability": capability}

    def _mock_call_no_outlet(self, capability, **kwargs):
        if capability == "object_detection_localization":
            return {
                "response": "No outlets detected.",
                "artifact": {"count": 0, "labels": []},
                "implementation": "mock",
                "capability": capability,
            }
        if capability == "spatial_reasoning":
            return {
                "response": "No electrical outlets are visible in the current view.",
                "artifact": {},
                "implementation": "mock",
                "capability": capability,
            }
        return {"response": "", "artifact": {}, "implementation": "mock", "capability": capability}

    def _mock_call_two_outlets(self, capability, **kwargs):
        if capability == "object_detection_localization":
            return {
                "response": "Two electrical outlets detected.",
                "artifact": {"count": 2, "labels": ["electrical outlet", "electrical outlet"]},
                "implementation": "mock",
                "capability": capability,
            }
        if capability == "spatial_reasoning":
            return {
                "response": "I see two outlets. The closest is at 12 o'clock, straight ahead.",
                "artifact": {},
                "implementation": "mock",
                "capability": capability,
            }
        return {"response": "", "artifact": {}, "implementation": "mock", "capability": capability}

    def test_single_outlet_response(self):
        image = create_outlet_image()
        with patch("plug_point_detector.copilot_llm_call", side_effect=self._mock_call):
            result = main(image, {})
        self.assertIsInstance(result, str)
        self.assertIn("11", result)

    def test_no_outlet_response(self):
        image = create_blank_wall_image()
        with patch("plug_point_detector.copilot_llm_call", side_effect=self._mock_call_no_outlet):
            result = main(image, {})
        self.assertIsInstance(result, str)
        self.assertIn("No", result)

    def test_two_outlets_response(self):
        image = create_two_outlet_image()
        with patch("plug_point_detector.copilot_llm_call", side_effect=self._mock_call_two_outlets):
            result = main(image, {})
        self.assertIsInstance(result, str)
        self.assertIn("two", result.lower())

    def test_streaming_mode_word_limit(self):
        image = create_outlet_image()
        long_response = " ".join(["word"] * 25)

        def mock_long(capability, **kwargs):
            return {"response": long_response, "artifact": {}, "implementation": "mock", "capability": capability}

        with patch("plug_point_detector.copilot_llm_call", side_effect=mock_long):
            result = main(image, {"streaming": True})
        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result.split()), 15)

    def test_non_streaming_no_word_limit(self):
        image = create_outlet_image()
        long_response = " ".join(["word"] * 25)

        def mock_long(capability, **kwargs):
            return {"response": long_response, "artifact": {}, "implementation": "mock", "capability": capability}

        with patch("plug_point_detector.copilot_llm_call", side_effect=mock_long):
            result = main(image, {"streaming": False})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.split()), 15)


if __name__ == "__main__":
    print("Running plug_point_detector tests...\n")
    unittest.main(verbosity=2)
