import unittest

import stream_server


TOOL_WITH_RUNTIME_INPUT = '''
TOOL_NAME = "object_recognition"
TOOL_PROMPT = "Describe useful visible objects."
TOOL_RUNTIME_INPUT = {
    "key": "target_object",
    "label": "Object to find",
    "placeholder": "Enter an object",
    "prompt_instruction": "The user is specifically looking for: {value}. Focus only on that target.",
}
'''


class TestRuntimeInputs(unittest.TestCase):
    def test_effective_prompt_defaults_to_tool_prompt_without_runtime_input(self):
        prompt = stream_server._effective_tool_prompt(
            "object_recognition",
            TOOL_WITH_RUNTIME_INPUT,
            {},
        )
        self.assertEqual(prompt, "Describe useful visible objects.")

    def test_effective_prompt_appends_sanitized_runtime_instruction(self):
        prompt = stream_server._effective_tool_prompt(
            "object_recognition",
            TOOL_WITH_RUNTIME_INPUT,
            {"target_object": "  water   cup  "},
        )
        self.assertEqual(
            prompt,
            "Describe useful visible objects.\n\n"
            "The user is specifically looking for: water cup. Focus only on that target.",
        )

    def test_runtime_inputs_ignore_malformed_or_unknown_values(self):
        self.assertEqual(
            stream_server._parse_runtime_inputs_from_request(
                {"runtime_inputs": {"wrong_key": "cup"}},
                TOOL_WITH_RUNTIME_INPUT,
            ),
            {},
        )
        self.assertEqual(
            stream_server._parse_runtime_inputs_from_request(
                {"runtime_inputs": {"target_object": "x" * 201}},
                TOOL_WITH_RUNTIME_INPUT,
            ),
            {},
        )

    def test_runtime_input_metadata_is_exposed_for_valid_tools(self):
        metadata = stream_server._tool_runtime_input_metadata(TOOL_WITH_RUNTIME_INPUT)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["key"], "target_object")
        self.assertEqual(metadata["label"], "Object to find")


if __name__ == "__main__":
    unittest.main()
