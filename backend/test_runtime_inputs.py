import ast
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"


def _literal_assignment(tool_path: Path, name: str):
    tree = ast.parse(tool_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    return None


class TestLocalToolRuntimeInputMetadata(unittest.TestCase):
    def test_object_recognition_declares_runtime_input_metadata(self):
        tool_path = TOOLS_DIR / "object_recognition.py"
        runtime_input = _literal_assignment(tool_path, "TOOL_RUNTIME_INPUT")

        self.assertEqual(
            runtime_input,
            {
                "key": "target_object",
                "label": "Object to find",
                "placeholder": "Enter an object, such as a water cup",
                "prompt_instruction": (
                    "The user is specifically looking for: {value}. Focus only on locating "
                    "that target and report whether and where it is visible."
                ),
            },
        )

    def test_scene_description_does_not_declare_runtime_input_metadata(self):
        tool_path = TOOLS_DIR / "scene_description.py"
        runtime_input = _literal_assignment(tool_path, "TOOL_RUNTIME_INPUT")

        self.assertIsNone(runtime_input)


if __name__ == "__main__":
    unittest.main()
