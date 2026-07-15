"""
Test script for find_nearest_exit.py
Tests the Find Nearest Exit tool with synthetic test data and mocked LLM calls.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Stub model_router_client so the tool can be imported without backend deps.
_mock_client = MagicMock()
sys.modules.setdefault("model_router_client", _mock_client)

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import find_nearest_exit
from find_nearest_exit import main


def create_test_image():
    """Create a simple test image."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 200


class TestFindNearestExit(unittest.TestCase):

    def test_main_returns_navigation_response(self):
        """main() returns the Stage 2 navigation response as a string."""
        image = create_test_image()
        stage1_artifact = {"detections": [{"label": "exit", "position": "right"}]}
        stage1_result = {"response": "Exit door detected to the right.", "artifact": stage1_artifact}
        stage2_result = {"response": "Turn right and walk forward ten steps to reach the exit.", "artifact": {}}

        with patch.object(find_nearest_exit, "copilot_llm_call", side_effect=[stage1_result, stage2_result]) as mock_call:
            result = main(image, {})

        self.assertEqual(result, "Turn right and walk forward ten steps to reach the exit.")
        self.assertEqual(mock_call.call_count, 2)

    def test_stage1_uses_object_detection_localization(self):
        """Stage 1 copilot_llm_call uses capability=object_detection_localization."""
        image = create_test_image()
        stage1_result = {"response": "Exit door ahead.", "artifact": {"detections": []}}
        stage2_result = {"response": "Walk straight ahead.", "artifact": {}}

        with patch.object(find_nearest_exit, "copilot_llm_call", side_effect=[stage1_result, stage2_result]) as mock_call:
            main(image, {})

        first_call_kwargs = mock_call.call_args_list[0]
        self.assertEqual(first_call_kwargs.kwargs.get("capability"), "object_detection_localization")

    def test_stage2_uses_navigation(self):
        """Stage 2 copilot_llm_call uses capability=navigation."""
        image = create_test_image()
        stage1_result = {"response": "Exit door ahead.", "artifact": {"detections": []}}
        stage2_result = {"response": "Walk straight ahead.", "artifact": {}}

        with patch.object(find_nearest_exit, "copilot_llm_call", side_effect=[stage1_result, stage2_result]) as mock_call:
            main(image, {})

        second_call_kwargs = mock_call.call_args_list[1]
        self.assertEqual(second_call_kwargs.kwargs.get("capability"), "navigation")

    def test_stage2_receives_stage1_artifact(self):
        """Stage 2 receives the Stage 1 artifact in metadata."""
        image = create_test_image()
        stage1_artifact = {"detections": [{"label": "exit", "position": "left"}]}
        stage1_result = {"response": "Exit on the left.", "artifact": stage1_artifact}
        stage2_result = {"response": "Turn left.", "artifact": {}}

        with patch.object(find_nearest_exit, "copilot_llm_call", side_effect=[stage1_result, stage2_result]) as mock_call:
            main(image, {})

        second_call_kwargs = mock_call.call_args_list[1]
        metadata = second_call_kwargs.kwargs.get("metadata", {})
        self.assertEqual(metadata.get("previous_stage_artifact"), stage1_artifact)

    def test_both_stages_receive_image(self):
        """Both stages pass the image to copilot_llm_call."""
        image = create_test_image()
        stage1_result = {"response": "Exit detected.", "artifact": {}}
        stage2_result = {"response": "Go forward.", "artifact": {}}

        with patch.object(find_nearest_exit, "copilot_llm_call", side_effect=[stage1_result, stage2_result]) as mock_call:
            main(image, {})

        for call in mock_call.call_args_list:
            images = call.kwargs.get("images", [])
            self.assertTrue(len(images) > 0, "Both stages must pass images")

    def test_missing_image_returns_error_string(self):
        """main() returns an audio-friendly error when image is None."""
        result = main(None, {})
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_empty_image_returns_error_string(self):
        """main() returns an audio-friendly error when image is empty."""
        result = main(np.array([]), {})
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_tool_name_constant(self):
        """TOOL_NAME constant matches expected value."""
        self.assertEqual(find_nearest_exit.TOOL_NAME, "find_nearest_exit")

    def test_llm_call_failure_returns_audio_friendly_error(self):
        """main() returns an audio-friendly error string when copilot_llm_call raises."""
        image = create_test_image()
        with patch.object(find_nearest_exit, "copilot_llm_call", side_effect=RuntimeError("network error")):
            result = main(image, {})
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


def run_all_tests():
    """Run all tests and print a summary."""
    print("=" * 60)
    print("Find Nearest Exit Tool Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFindNearestExit)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"{len(result.failures)} failure(s), {len(result.errors)} error(s).")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_all_tests())
