import ast
import unittest
from pathlib import Path
from typing import Any, Dict, Optional


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
STREAM_SERVER_PATH = Path(__file__).resolve().parent / "stream_server.py"


def _literal_assignment(tool_path: Path, name: str):
    tree = ast.parse(tool_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    return None


def _load_stream_server_runtime_input_helpers():
    tree = ast.parse(STREAM_SERVER_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_literal_tool_metadata",
        "_sanitize_runtime_input_definition",
        "_tool_runtime_input_definition",
    }
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "Optional": Optional,
        "ast": ast,
    }
    exec(compile(module, str(STREAM_SERVER_PATH), "exec"), namespace)
    return namespace


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

    def test_generated_tool_alias_runtime_input_is_parsed(self):
        helpers = _load_stream_server_runtime_input_helpers()
        tool_code = '''
runtime_input = {
    "key": "target_object",
    "label": "Object to find",
    "placeholder": "Enter an object",
    "prompt_instruction": "Focus on {value}.",
}
'''
        parsed = helpers["_tool_runtime_input_definition"](tool_code)

        self.assertEqual(
            parsed,
            {
                "key": "target_object",
                "label": "Object to find",
                "placeholder": "Enter an object",
                "prompt_instruction": "Focus on {value}.",
            },
        )


if __name__ == "__main__":
    unittest.main()
