import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litellm_utils
import stream_server
from validate_generated_tools import validate_take_photo_baseline


class TestTakePhotoBaseline(unittest.TestCase):
    def test_no_planner_creation_persists_raw_prompt_without_llama(self):
        request = (
            "Read the medication label. Say only the drug name and dosage; "
            "if either is unreadable, say unreadable."
        )
        with patch.object(stream_server, "TAKE_PHOTO_PLANNING_MODE", "no_planner"), \
             patch.object(stream_server, "system_llm_call") as planner:
            parsed = stream_server.parse_transcript_with_ai(request)

        planner.assert_not_called()
        self.assertEqual(parsed["runtime_prompt"], request)
        self.assertEqual(parsed["description"], request)
        self.assertEqual(parsed["stages"] if "stages" in parsed else [], [])
        self.assertNotIn("fused_vlm_prompt", parsed)

    def test_fused_prompt_creation_calls_planner_once_and_persists_fused_prompt(self):
        response = type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": '''{
                    "type": "visual AT", "title": "Read medication",
                    "description": "Read the medication label.",
                    "example_usage": "Drug name and dosage only.",
                    "additional": "Say unreadable when needed.",
                    "execution_mode": "take-photo", "missing_fields": [],
                    "fused_vlm_prompt": "Read the label; return only drug name and dosage."
                }'''})()
            })()]
        })()
        with patch.object(stream_server, "TAKE_PHOTO_PLANNING_MODE", "fused_prompt"), \
             patch.object(stream_server, "system_llm_call", return_value=response) as planner:
            parsed = stream_server.parse_transcript_with_ai("Read my medication label in take-photo mode.")

        planner.assert_called_once()
        self.assertEqual(parsed["runtime_prompt"], "Read the label; return only drug name and dosage.")
        self.assertEqual(parsed["stages"], [])
        self.assertNotIn("fused_vlm_prompt", parsed)

    def test_p1_runtime_uses_persisted_prompt_and_one_gemini_call(self):
        tool_code = 'TOOL_PROMPT = "Return only the color; say unknown if uncertain."\n'
        with patch.object(stream_server, "call_take_photo_baseline_vlm", return_value="Blue") as gemini, \
             patch.object(stream_server, "tool_copilot_llm_call") as router, \
             patch.object(stream_server, "system_llm_call") as planner:
            result = stream_server._run_take_photo_single_vlm(
                "color_tool", "rewritten description", tool_code, b"image"
            )

        self.assertEqual(result, "Blue")
        gemini.assert_called_once_with(
            image=b"image", prompt="Return only the color; say unknown if uncertain."
        )
        router.assert_not_called()
        planner.assert_not_called()

    def test_helper_makes_one_fixed_model_call_without_retries(self):
        response = type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": "The label says aspirin."})()
            })()]
        })()
        with patch.object(litellm_utils, "call_model", return_value=response) as call:
            answer = litellm_utils.call_take_photo_baseline_vlm(
                image=b"image", prompt="Read the label."
            )

        self.assertEqual(answer, "The label says aspirin.")
        call.assert_called_once_with(
            model_name=litellm_utils.TAKE_PHOTO_BASELINE_MODEL,
            messages=[{"role": "user", "content": "Read the label."}],
            images=[b"image"],
            metadata={"num_retries": 0},
        )

    def test_take_photo_validator_accepts_exactly_one_baseline_call(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
from litellm_utils import call_take_photo_baseline_vlm
TOOL_PROMPT = "Read the label."
def main(image, input_data):
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
'''
        self.assertEqual(validate_take_photo_baseline(tool, issue, Path("tools/read_label.py")), [])

    def test_take_photo_validator_rejects_router_and_multiple_calls(self):
        issue = "## Mode\n\ntake-photo\n"
        tool = '''
TOOL_PROMPT = "Read the label."
def main(image, input_data):
    copilot_llm_call(capability="ocr", images=[image])
    call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
'''
        failures = validate_take_photo_baseline(tool, issue, Path("tools/read_label.py"))
        self.assertTrue(any("exactly one" in failure for failure in failures))
        self.assertTrue(any("must not call copilot_llm_call" in failure for failure in failures))

    def test_take_photo_validator_rejects_copilot_prompt_rewrite(self):
        issue = """## Mode
take-photo

## Runtime prompt
Read the label. Return only the drug name and dosage; say unreadable if unclear.
"""
        tool = '''
from litellm_utils import call_take_photo_baseline_vlm
TOOL_PROMPT = "Please provide a helpful summary of this medication label."
def main(image, input_data):
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
'''
        failures = validate_take_photo_baseline(tool, issue, Path("tools/read_label.py"))
        self.assertTrue(any("exactly match" in failure for failure in failures))

    def test_runtime_bypass_is_only_in_run_tool_branch(self):
        source = (Path(__file__).resolve().parent / "stream_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        references = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "_run_take_photo_single_vlm"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(references), 1)
        self.assertIn("if data.get('type') == 'run_tool':", source)


if __name__ == "__main__":
    unittest.main()
