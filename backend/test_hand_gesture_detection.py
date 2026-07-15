import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import litellm_utils
import hand_gesture_detection


class TestHandGestureDetection(unittest.TestCase):
    def _make_response(self, text):
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": text})()
            })()]
        })()

    def test_returns_vlm_answer(self):
        expected = "Someone is giving a thumbs up, which typically signals approval or agreement."
        with patch.object(litellm_utils, "call_model", return_value=self._make_response(expected)):
            result = hand_gesture_detection.main(b"image", {})
        self.assertEqual(result, expected)

    def test_passes_tool_prompt_verbatim(self):
        with patch.object(litellm_utils, "call_model", return_value=self._make_response("ok")) as mock_call:
            hand_gesture_detection.main(b"image", {})
        _, kwargs = mock_call.call_args
        messages = kwargs.get("messages") or mock_call.call_args[0][1]
        prompt_sent = mock_call.call_args[1]["messages"][0]["content"]
        self.assertEqual(prompt_sent, hand_gesture_detection.TOOL_PROMPT)

    def test_no_image_returns_error(self):
        result = hand_gesture_detection.main(None, {})
        self.assertEqual(result, "No camera image is available.")

    def test_makes_exactly_one_model_call(self):
        with patch.object(litellm_utils, "call_model", return_value=self._make_response("waving")) as mock_call:
            hand_gesture_detection.main(b"image", {})
        self.assertEqual(mock_call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
