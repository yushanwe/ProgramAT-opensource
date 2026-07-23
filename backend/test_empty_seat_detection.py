import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
TOOL_PATH = TOOLS_DIR / "empty_seat_detection.py"


class EmptySeatDetectionTests(unittest.TestCase):
    def test_declares_required_constants(self):
        tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"TOOL_NAME", "EXECUTION_MODE", "TOOL_POLICY"}
        }
        self.assertEqual(constants["TOOL_NAME"], "empty_seat_detection")
        self.assertEqual(constants["EXECUTION_MODE"], "take_photo")
        self.assertEqual(
            constants["TOOL_POLICY"],
            {"strategy": "single", "models": ["gemini-3.1-flash-lite"]},
        )

    def test_main_handles_none_image(self):
        module = self._load_tool_module()
        self.assertEqual(module.main(None, {}), "No camera image is available.")

    def test_main_calls_execute_tool_policy_once(self):
        module = self._load_tool_module()
        fake_image = object()
        with patch.object(module, "execute_tool_policy", return_value="Go left two steps.") as call:
            result = module.main(fake_image, {})
        self.assertEqual(result, "Go left two steps.")
        call.assert_called_once_with(
            image=fake_image,
            prompt=module.TOOL_PROMPT,
            policy=module.TOOL_POLICY,
            tool_name=module.TOOL_NAME,
        )

    def _load_tool_module(self):
        spec = importlib.util.spec_from_file_location("empty_seat_detection", TOOL_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
