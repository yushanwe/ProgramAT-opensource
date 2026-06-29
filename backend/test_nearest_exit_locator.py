import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
BACKEND_DIR = Path(__file__).resolve().parent

backend_litellm_spec = importlib.util.spec_from_file_location("litellm_utils", BACKEND_DIR / "litellm_utils.py")
backend_litellm_utils = importlib.util.module_from_spec(backend_litellm_spec)
assert backend_litellm_spec.loader is not None
backend_litellm_spec.loader.exec_module(backend_litellm_utils)
sys.modules["litellm_utils"] = backend_litellm_utils

sys.path.insert(0, str(TOOLS_DIR))
import nearest_exit_locator


def fake_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class TestNearestExitLocator(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_main_returns_error_when_image_missing(self):
        result = nearest_exit_locator.main(None, {})

        self.assertEqual(result["audio"]["type"], "error")
        self.assertIn("No camera image available", result["text"])

    @patch("nearest_exit_locator.copilot_llm_call")
    def test_main_runs_detection_then_navigation(self, mock_copilot_llm_call):
        mock_copilot_llm_call.side_effect = [
            fake_response("Exit door ahead, slightly left."),
            fake_response("Walk forward and veer left toward the exit door."),
        ]

        result = nearest_exit_locator.main(self.image, {})

        self.assertEqual(result, "Walk forward and veer left toward the exit door.")
        self.assertEqual(mock_copilot_llm_call.call_count, 2)

        first_call = mock_copilot_llm_call.call_args_list[0].kwargs
        second_call = mock_copilot_llm_call.call_args_list[1].kwargs

        self.assertEqual(first_call["task_category"], "object_detection")
        self.assertEqual(second_call["task_category"], "navigation")
        self.assertEqual(
            second_call["metadata"]["previous_stage_output"],
            "Exit door ahead, slightly left.",
        )
        self.assertTrue(first_call["images"][0].startswith("data:image/jpeg;base64,"))

    @patch("nearest_exit_locator.copilot_llm_call")
    def test_main_falls_back_when_navigation_text_is_empty(self, mock_copilot_llm_call):
        mock_copilot_llm_call.side_effect = [
            fake_response("No exit door visible."),
            fake_response(""),
        ]

        result = nearest_exit_locator.main(self.image, {})

        self.assertEqual(result, "I could not guide you to an exit. Please scan again.")


if __name__ == "__main__":
    unittest.main()
