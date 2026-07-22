"""
Tests for locate_empty_chairs.py

Validates the tool constants, contract compliance, and main() behavior.
Uses AST-based constant checks and sys.modules mocking to avoid runtime
dependencies (PIL, GPU libraries) that are not available in CI.
"""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make backend helpers importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_generated_tools

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "locate_empty_chairs.py"
TOOL_SOURCE = TOOL_PATH.read_text(encoding="utf-8")

ISSUE_TEXT = """
## Tool name
Locate Empty Chairs

## Task
Locating available seating in crowded spaces

## Expected output
Guide me toward an empty chair nearby

## Constraints / examples
Crowded spaces, locating empty chairs

## Mode
take_photo
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_string_constants(source: str) -> dict:
    """Return top-level string-literal assignments from source."""
    tree = ast.parse(source)
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _load_tool_with_mocked_vlm(mock_return: str):
    """
    Import locate_empty_chairs with litellm_utils mocked so PIL / network
    dependencies are not required.  Returns (module, mock_vlm_callable).
    """
    mock_vlm = MagicMock(return_value=mock_return)
    fake_litellm = types.ModuleType("litellm_utils")
    fake_litellm.call_take_photo_vlm = mock_vlm

    # Inject the fake module before importing the tool.
    with patch.dict(sys.modules, {"litellm_utils": fake_litellm}):
        # Remove any previously cached version so the mock is used.
        sys.modules.pop("locate_empty_chairs", None)
        tools_dir = str(TOOL_PATH.parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import importlib
        mod = importlib.import_module("locate_empty_chairs")

    return mod, mock_vlm


# ---------------------------------------------------------------------------
# Tests: module-level constants (AST-based, no import needed)
# ---------------------------------------------------------------------------

class TestLocateEmptyChairsConstants(unittest.TestCase):
    """Verify module-level constants via AST — no runtime imports needed."""

    def setUp(self):
        self.constants = _parse_string_constants(TOOL_SOURCE)

    def test_tool_name(self):
        self.assertEqual(self.constants.get("TOOL_NAME"), "locate_empty_chairs")

    def test_execution_mode(self):
        self.assertEqual(self.constants.get("EXECUTION_MODE"), "take_photo")

    def test_tool_prompt_is_non_empty(self):
        self.assertTrue(self.constants.get("TOOL_PROMPT", "").strip())

    def test_no_video_config(self):
        tree = ast.parse(TOOL_SOURCE)
        assigned_names = {
            node.targets[0].id
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertNotIn("VIDEO_CONFIG", assigned_names)


# ---------------------------------------------------------------------------
# Tests: contract validation
# ---------------------------------------------------------------------------

class TestLocateEmptyChairsContractValidation(unittest.TestCase):
    """Validate the tool passes the generated-tool take-photo validator."""

    def test_passes_take_photo_validator(self):
        failures = validate_generated_tools.validate_take_photo_tool(
            TOOL_SOURCE, ISSUE_TEXT, Path("tools/locate_empty_chairs.py")
        )
        self.assertEqual(failures, [], failures)


# ---------------------------------------------------------------------------
# Tests: main() behaviour (litellm_utils mocked)
# ---------------------------------------------------------------------------

class TestLocateEmptyChairsMain(unittest.TestCase):
    """Unit tests for main() with VLM dependency mocked."""

    def test_returns_string_when_image_is_none(self):
        mod, _ = _load_tool_with_mocked_vlm("unused")
        result = mod.main(None, {})
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_returns_vlm_answer_for_valid_image(self):
        expected = "There is an empty chair to your left, about five steps away."
        mod, mock_vlm = _load_tool_with_mocked_vlm(expected)

        # Use a sentinel object; the VLM is mocked so actual image format is irrelevant.
        fake_image = object()
        result = mod.main(fake_image, {})

        self.assertEqual(result, expected)
        mock_vlm.assert_called_once_with(
            image=fake_image,
            prompt=mod.TOOL_PROMPT,
            tool_name=mod.TOOL_NAME,
        )

    def test_fallback_response_when_no_seats_visible(self):
        fallback = "No empty seats are visible in the current view."
        mod, _ = _load_tool_with_mocked_vlm(fallback)

        fake_image = object()
        result = mod.main(fake_image, {})

        self.assertEqual(result, fallback)


if __name__ == "__main__":
    unittest.main()
