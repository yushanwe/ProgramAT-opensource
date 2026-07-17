"""
Test script for play_card_identifier.py
Tests the play card identifier tool with sample data.
"""

import sys
import os
import ast
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

# Add backend directory first so litellm_utils resolves to backend/litellm_utils.py,
# which contains call_take_photo_baseline_vlm.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# Then add tools directory for the tool module itself.
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "tools"))

import litellm_utils
import play_card_identifier
from play_card_identifier import main, TOOL_NAME, TOOL_PROMPT


def create_test_image():
    """Create a simple blank test image."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 200


class TestPlayCardIdentifier(unittest.TestCase):
    def test_tool_name_constant(self):
        self.assertEqual(TOOL_NAME, "play_card_identifier")

    def test_tool_prompt_is_non_empty(self):
        self.assertTrue(TOOL_PROMPT.strip())

    def test_tool_source_passes_baseline_validation(self):
        from validate_generated_tools import validate_take_photo_baseline
        tool_path = Path(__file__).resolve().parent.parent / "tools" / "play_card_identifier.py"
        tool_text = tool_path.read_text(encoding="utf-8")
        issue_text = "## Mode\n\ntake-photo\n"
        failures = validate_take_photo_baseline(tool_text, issue_text, Path("tools/play_card_identifier.py"))
        self.assertEqual(failures, [], msg="\n".join(failures))

    def test_main_returns_string_from_vlm(self):
        image = create_test_image()
        with patch.object(
            play_card_identifier,
            "call_take_photo_baseline_vlm",
            return_value="Seven of Hearts",
        ) as mock_vlm:
            result = main(image, {})

        self.assertEqual(result, "Seven of Hearts")
        mock_vlm.assert_called_once_with(
            image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
        )

    def test_main_handles_none_image(self):
        result = main(None, {})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_main_passes_input_data(self):
        image = create_test_image()
        with patch.object(
            play_card_identifier,
            "call_take_photo_baseline_vlm",
            return_value="Ace of Spades",
        ):
            result = main(image, {"some_key": "some_value"})

        self.assertEqual(result, "Ace of Spades")


if __name__ == "__main__":
    unittest.main()
