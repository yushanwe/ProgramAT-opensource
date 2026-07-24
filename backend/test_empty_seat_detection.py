import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
TOOL_PATH = TOOLS_DIR / "empty_seat_detection.py"
HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("cv2", "numpy", "PIL", "litellm_utils")
)


class EmptySeatDetectionTests(unittest.IsolatedAsyncioTestCase):
    def test_declares_required_constants(self):
        tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
        constants = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id
                in {
                    "TOOL_NAME",
                    "EXECUTION_MODE",
                    "TOOL_PROMPT",
                    "PRECISE_TOOL_PROMPT",
                }
            ):
                name = node.targets[0].id
                if isinstance(node.value, ast.Constant):
                    constants[name] = node.value.value
                else:
                    constants[name] = ast.literal_eval(node.value)

        self.assertEqual(constants["TOOL_NAME"], "empty_seat_detection")
        self.assertEqual(constants["EXECUTION_MODE"], "take_photo")
        self.assertIn("blind or low-vision user", constants["TOOL_PROMPT"])
        self.assertIn("clock-face direction", constants["TOOL_PROMPT"])
        self.assertIn("more precise", constants["PRECISE_TOOL_PROMPT"])

    @unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime deps unavailable")
    async def test_take_photo_handles_none_image(self):
        module = self._load_tool_module()
        self.assertEqual(
            await module.on_take_photo(None, None, {}),
            "No camera image is available.",
        )

    @unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime deps unavailable")
    async def test_take_photo_calls_progressive_emitter(self):
        module = self._load_tool_module()
        fake_image = object()
        runtime = object()
        with patch.object(
            module,
            "_emit_progressive_results",
            new=AsyncMock(return_value="result"),
        ) as emit:
            result = await module.on_take_photo(runtime, fake_image, {})
        self.assertIsNone(result)
        emit.assert_awaited_once()

    def _load_tool_module(self):
        spec = importlib.util.spec_from_file_location("empty_seat_detection", TOOL_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
