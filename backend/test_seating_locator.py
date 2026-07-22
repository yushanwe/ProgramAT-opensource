import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parent.parent / "tools"))
import seating_locator
from validate_generated_tools import validate_take_photo_tool


class SeatingLocatorTests(unittest.TestCase):
    def test_declares_take_photo_contract(self):
        self.assertEqual(seating_locator.TOOL_NAME, "seating_locator")
        self.assertEqual(seating_locator.EXECUTION_MODE, "take_photo")
        self.assertEqual(
            seating_locator.TOOL_POLICY,
            {"strategy": "single", "models": ["gemini-3.1-flash-lite"]},
        )

    def test_main_returns_clear_message_when_image_missing(self):
        self.assertEqual(seating_locator.main(None, {}), "No camera image is available.")

    def test_main_executes_declared_policy_once(self):
        with patch.object(
            seating_locator, "execute_tool_policy", return_value="Turn slightly right."
        ) as execute:
            result = seating_locator.main(object(), {})

        self.assertEqual(result, "Turn slightly right.")
        execute.assert_called_once_with(
            image=unittest.mock.ANY,
            prompt=seating_locator.TOOL_PROMPT,
            policy=seating_locator.TOOL_POLICY,
            tool_name=seating_locator.TOOL_NAME,
        )

    def test_validates_against_take_photo_rules(self):
        tool_path = Path(__file__).resolve().parent.parent / "tools" / "seating_locator.py"
        source = tool_path.read_text(encoding="utf-8")
        issue_text = """## Task
locate available seating

## Expected output
Find an empty chair nearby

## Constraints / examples
guide me toward it, crowded spaces

## Mode
take_photo
"""
        failures = validate_take_photo_tool(source, issue_text, Path("tools/seating_locator.py"))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
