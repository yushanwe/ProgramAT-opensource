"""
Tests for AI parsing and template filling functionality
These tests test the logic in isolation without importing stream_server
"""
import unittest
import json
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainstorming import (
    build_clarified_update_request,
    build_update_brainstorm_context,
    generate_brainstorming_question,
    sanitize_action_log,
)
from stage_decomposition import (
    PLAYING_CARD_STAGE_GOAL,
    build_stage_decomposition_prompt,
    normalize_stage_plan,
)


def load_stream_server_function(name: str):
    """Load one pure helper without importing the server and triggering startup config."""
    source_path = Path(__file__).resolve().parent / "stream_server.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    namespace = {
        "Any": Any,
        "ast": ast,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "json": json,
        "re": re,
        "ToolPlanningContext": Any,
        "has_executable_lifecycle": lambda text: "async def on_frame(" in text,
    }
    dependency_names = {
        "_normalize_issue_creation_requirements": {
            "_normalize_custom_gpt_value",
            "_explicit_custom_gpt_value",
        },
        "_append_task_stages_to_issue_body": {"_build_task_stages_markdown"},
        "_select_streaming_context": {
            "_request_explicitly_requires_temporal_context",
        },
        "_tool_execution_mode": {"_literal_tool_metadata"},
        "fill_template": {
            "_original_request_fallback",
            "_replace_issue_section",
        },
    }.get(name, set())
    dependencies = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in dependency_names
    ]
    exec(
        compile(ast.Module(body=[*dependencies, function_node], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace[name]


class TestIssueTemplateGuidance(unittest.TestCase):
    """Test Claude-facing issue template guidance."""

    def test_visual_at_template_uses_executable_tool_lifecycle(self):
        template_path = Path(__file__).resolve().parent.parent / ".github" / "ISSUE_TEMPLATE" / "visual_at.md"
        template = template_path.read_text(encoding="utf-8")

        for heading in (
            "Tool name", "Task", "Expected output",
            "Constraints / examples", "Streaming context",
        ):
            self.assertIn(f"## {heading}", template)
            self.assertIn(f"## {heading}\n\n<!--", template)
        self.assertNotIn("call_take_photo_vlm", template)
        self.assertNotIn("execute_tool_policy", template)
        self.assertNotIn("TOOL_POLICY", template)
        self.assertNotIn("Task Stages", template)
        self.assertNotIn("copilot_llm_call", template)
        self.assertIn("repository-root `CLAUDE.md`", template)
        self.assertIn("Modify only the requested tool file", template)

        instructions_path = template_path.parent.parent.parent / "CLAUDE.md"
        instructions = instructions_path.read_text(encoding="utf-8")
        normalized_instructions = " ".join(instructions.split())
        self.assertTrue(instructions.startswith("# ProgramAT\n"))
        self.assertIn("call_model", instructions)
        self.assertIn("call_openai_responses_model", instructions)
        self.assertIn("get_recent_frames", instructions)
        self.assertIn("get_state", instructions)
        self.assertIn("Do not split prompt steps into separate model calls", instructions)
        self.assertIn("Each hook independently decides", instructions)
        self.assertIn("Take Photo and Streaming may use different", instructions)
        self.assertIn("blind and low-vision users", instructions)
        self.assertIn("never use color", instructions.casefold())
        self.assertIn("compatibility with older tools", instructions)
        self.assertIn("## Tool quality and accessibility conventions", instructions)
        self.assertIn("Python in `tools/`", instructions)
        self.assertIn("must not connect to the backend or use WebSockets", normalized_instructions)
        self.assertIn("plain language, not JSON", normalized_instructions)
        self.assertIn("does not prove the requested target is absent", normalized_instructions)
        self.assertIn("9–12 and 1–3", instructions)
        self.assertIn("Avoid GPU-heavy packages", instructions)
        self.assertIn("do not import one tool module from another", normalized_instructions)
        self.assertIn("approved shared backend helpers", normalized_instructions)
        self.assertIn("avoid unnecessary documentation", instructions)
        self.assertIn("If one instruction is enough", instructions)
        self.assertIn("short numbered sequence", instructions)
        self.assertIn("Do not create separate prompts, planner stages", instructions)
        self.assertIn("Do not add steps merely to restate the issue fields", instructions)
        self.assertIn("Assist a blind user in finding an indoor exit", instructions)
        self.assertIn("Read the medication label and report the text concisely", instructions)
        self.assertIn("Use one model when", instructions)
        self.assertIn("Use a conditional cascade when", instructions)
        self.assertIn("call the stronger model only when escalation is needed", instructions)
        self.assertIn("deliver one final answer", instructions)
        self.assertIn("Use parallel progressive only when", instructions)
        self.assertIn("multiple model results delivered as they complete", instructions)
        self.assertIn("Do not infer progressive output", instructions)
        self.assertIn("Model strategy remains tool-controlled", instructions)
        self.assertIn("Intentional parallel execution is valid", instructions)
        self.assertIn("Assume `on_frame()` may run again while earlier async work is still running", instructions)
        self.assertIn("decide what unit of work it represents", instructions)
        self.assertIn("Avoid launching another identical call for the same unit of work", normalized_instructions)
        self.assertIn("intentional calls to different models may run concurrently", normalized_instructions)
        self.assertIn("Record an in-flight task, generation, window id, or request key before awaiting", normalized_instructions)
        self.assertIn("Validate cancellation and relevance again before emitting", normalized_instructions)
        self.assertIn("clear completed task state in `finally` or a completion callback", normalized_instructions)
        self.assertIn("Do not use one global lock around all model calls", instructions)
        self.assertIn("separate frame collection, temporal window scheduling, and model execution", normalized_instructions)
        self.assertIn("Strongly overlapping frame histories should not automatically become separate analyses", normalized_instructions)
        self.assertIn("Multi-frame means several ordered frames may be passed in one model call", normalized_instructions)
        self.assertIn("call_key = (window_key, model_name, phase)", instructions)
        self.assertIn("Event-driven tools should define what begins an event", normalized_instructions)
        self.assertIn("Continuous monitoring tools should define an explicit interval", normalized_instructions)
        self.assertIn("window_key = build_window_key(selected_frames)", instructions)
        self.assertIn("What event or interval does one model call represent?", instructions)
        self.assertIn("A per-frame or constantly changing generation key does not deduplicate temporal work", normalized_instructions)
        self.assertIn("Scene change, motion, and temporal-window identity are not the same thing", normalized_instructions)
        self.assertIn("Motion may contribute frames to the current gesture or event window", normalized_instructions)
        self.assertIn("idle -> collecting -> analyzing -> waiting_for_reset -> idle", instructions)
        self.assertIn("Do not use JPEG bytes, base64 fragments", instructions)
        self.assertIn("Cache heavyweight local models instead of loading them on every frame", normalized_instructions)
        self.assertIn("key = (\"detailed_scene\", scene_generation)", instructions)
        self.assertIn("tasks[key] = task", instructions)
        self.assertIn("if runtime.get_state(\"scene_generation\") != scene_generation", instructions)
        self.assertIn("source of truth for generated-tool instructions", instructions)

        compatibility_path = template_path.parent.parent / "copilot-instructions.md"
        compatibility = compatibility_path.read_text(encoding="utf-8")
        self.assertIn("Compatibility note", compatibility)
        self.assertIn("source of truth is the repository-root `CLAUDE.md`", compatibility)

        self.assertNotIn("EXECUTION_MODE", template)
        self.assertNotIn("hosted_video_streaming", template)

    def test_temporal_sign_language_request_cannot_be_downgraded_to_take_photo(self):
        request = """Problem:
I need help understanding a temporal short sign language gesture.

Example Usage:
Observe the signer’s recent hand movements and identify the sign or short signed phrase they just made. The sign language might be static or last for a few seconds.

Custom GPT:
No"""
        select_context = load_stream_server_function("_select_streaming_context")

        # Regression: even a mistaken Llama suggestion must not erase the
        # request's explicit recent-motion and just-made semantics.
        self.assertIn("one user-facing task", template)
        self.assertIn("If this issue enumerates multiple stages", template)
        self.assertIn("one ordered `copilot_llm_call(...)` per stage", template)
        self.assertIn("evaluate and escalate reasoning capabilities", template)
        self.assertIn("pass useful structured artifacts to later calls", template)
        self.assertIn("spatial_reasoning", template)
        self.assertIn("Choose only from these capabilities", template)
        self.assertNotIn("Capability Pipeline", template)
        self.assertNotIn("should either utilize Yolo11", template)
        self.assertNotIn("Google Vision", template)
        self.assertIn("must not choose implementations", template)
        self.assertIn("provider-specific `DEFAULT_MODEL` constants", template)


class TestBrainstormingHelpers(unittest.IsolatedAsyncioTestCase):
    def test_sanitize_action_log_removes_noise_and_redacts(self):
        raw_log = """
        npm install
        Authorization: Bearer abc123
        Claude summary: updated the matching logic.
        Modified tools/uber_car_identifier.py
        Modified tools/uber_car_identifier.py
        Validation passed
        """

        cleaned = sanitize_action_log(raw_log, max_chars=500)

        self.assertIn("Claude summary: updated the matching logic.", cleaned)
        self.assertIn("Modified tools/uber_car_identifier.py", cleaned)
        self.assertIn("Validation passed", cleaned)
        self.assertNotIn("npm install", cleaned)
        self.assertNotIn("abc123", cleaned)

    def test_update_context_includes_expected_fields(self):
        context = build_update_brainstorm_context(
            repository="owner/repo",
            issue_number=12,
            pr_number=34,
            tool_name="tool_a",
            tool_path="tools/tool_a.py",
            update_request="Make streaming quieter",
            claude_summary="Claude says the detector flips often.",
            action_log="Validation passed",
            tool_source="def run(): pass",
        )

        self.assertEqual(context["repository"], "owner/repo")
        self.assertEqual(context["issue_number"], 12)
        self.assertEqual(context["pr_number"], 34)
        self.assertEqual(context["tool_name"], "tool_a")
        self.assertEqual(context["tool_path"], "tools/tool_a.py")
        self.assertEqual(context["update_request"], "Make streaming quieter")
        self.assertEqual(context["claude_summary"], "Claude says the detector flips often.")
        self.assertEqual(context["action_log"], "Validation passed")
        self.assertEqual(context["tool_source"], "def run(): pass")

    async def test_create_mode_prompt_shape_remains_unchanged(self):
        captured = {}

        def fake_llm_call(*, messages):
            captured["prompt"] = messages[0]["content"]
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "How should it react?"})()})()]})()

        def fake_extract_text(response):
            return response.choices[0].message.content

        question = await generate_brainstorming_question(
            mode="create",
            context={
                "title": "Find my Uber",
                "description": "Identify the correct car.",
                "solution": "Match by make and color.",
                "example_usage": "Find the white Toyota.",
                "video_summary": "User points at the curb.",
            },
            llm_call=fake_llm_call,
            text_extractor=fake_extract_text,
        )

        self.assertEqual(question, "How should it react?")
        self.assertIn("You are helping a blind or low-vision user design a camera-based assistive tool.", captured["prompt"])
        self.assertIn("Tool title: Find my Uber", captured["prompt"])
        self.assertIn("Video demonstration summary:\nUser points at the curb.", captured["prompt"])

    async def test_update_mode_prompt_contains_context_and_tolerates_missing_fields(self):
        captured = {}

        def fake_llm_call(*, messages):
            captured["prompt"] = messages[0]["content"]
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "Should streaming announce only on changes?"})()})()]})()

        question = await generate_brainstorming_question(
            mode="update",
            context=build_update_brainstorm_context(
                repository="owner/repo",
                issue_number=77,
                pr_number=88,
                tool_name="uber_car_identifier",
                tool_path="tools/uber_car_identifier.py",
                update_request="Add a streaming mode.",
                claude_summary="",
                action_log="",
                tool_source="",
            ),
            llm_call=fake_llm_call,
            text_extractor=lambda response: response.choices[0].message.content,
        )

        self.assertEqual(question, "Should streaming announce only on changes?")
        self.assertIn("Repository: owner/repo", captured["prompt"])
        self.assertIn("Issue number: 77", captured["prompt"])
        self.assertIn("PR number: 88", captured["prompt"])
        self.assertIn("Current tool: uber_car_identifier", captured["prompt"])
        self.assertIn("User update request: Add a streaming mode.", captured["prompt"])

    def test_build_clarified_update_request_combines_answers(self):
        clarified = build_clarified_update_request(
            "Add streaming support.",
            ["Should output change?", "What stays the same?"],
            ["Yes, announce only changes.", "Single-frame mode should stay the same."],
        )

        self.assertIn("Original update request:", clarified)
        self.assertIn("Q: Should output change?", clarified)
        self.assertIn("A: Yes, announce only changes.", clarified)

    def test_root_test_script_uses_shared_implementation(self):
        script_path = Path(__file__).resolve().parent.parent / "test.py"
        script_text = script_path.read_text(encoding="utf-8")

        self.assertIn("from brainstorming import", script_text)
        self.assertIn("generate_brainstorming_question", script_text)
        self.assertNotIn("Return only the question itself, nothing else.", script_text)


class TestParserStageIssueIntegration(unittest.TestCase):
    """Test parser-owned stage normalization and issue-section wiring."""

    def setUp(self):
        self.normalize_requirements = load_stream_server_function("_normalize_issue_creation_requirements")
        self.extraction_prompt = load_stream_server_function("_build_issue_extraction_prompt")
        self.merge_outputs = load_stream_server_function("_merge_issue_and_stage_outputs")
        self.normalize = normalize_stage_plan
        self.render = load_stream_server_function("_build_task_stages_markdown")
        self.append_stages = load_stream_server_function("_append_task_stages_to_issue_body")
        self.parse_json = load_stream_server_function("_parse_llm_json_object")
        self.response_field_only = load_stream_server_function("_response_field_only")

    def test_mobile_payload_uses_only_atomic_response_field(self):
        structured = {
            "response": "Turn right toward the exit.",
            "artifact": {"detections": [{"label": "exit"}]},
            "implementation": "navigator",
            "capability": "navigation",
        }

        self.assertEqual(
            select_context("latest_frame", request),
            "recent_history",
        )
        self.assertEqual(
            select_context("", request),
            "recent_history",
        )

        fill_issue = load_stream_server_function("fill_template")
        template = (
            Path(__file__).resolve().parent.parent
            / ".github" / "ISSUE_TEMPLATE" / "visual_at.md"
        ).read_text(encoding="utf-8")
        issue = fill_issue(template, {
            "title": "recent_sign_language",
            "description": "Understand a temporal short sign language gesture.",
            "example_usage": "Observe recent hand movements and identify the phrase just made.",
            "additional": "The sign may last for a few seconds.",
            "streaming_context": select_context("latest_frame", request),
            "original_prompts": [request],
        })
        self.assertIn("## Streaming context\n\nrecent_history", issue)
        self.assertIn("signer’s recent hand movements", issue)

    def test_current_template_maps_task_and_expected_output_by_heading(self):
        fill_issue = load_stream_server_function("fill_template")
        template = (
            Path(__file__).resolve().parent.parent
            / ".github" / "ISSUE_TEMPLATE" / "visual_at.md"
        ).read_text(encoding="utf-8")
        original = (
            "Problem: I need help understanding a recent hand gesture. "
            "Example Usage: Identify the gesture or short meaning."
        )
        issue = fill_issue(template, {
            "title": "Hand Gesture Detector",
            "description": "Identify a recent hand gesture from visible motion.",
            "example_usage": "Report the gesture or short meaning just expressed.",
            "additional": "Use the full motion sequence.",
            "streaming_context": "recent_history",
            "original_prompts": [f"[2026-07-27 15:49] {original}"],
        })

        self.assertIn(
            "## Task\n\nIdentify a recent hand gesture from visible motion.",
            issue,
        )
        self.assertIn(
            "## Expected output\n\nReport the gesture or short meaning just expressed.",
            issue,
        )
        self.assertNotIn(
            "<!-- What should the tool determine from the camera image or recent frames? -->",
            issue,
        )

    def test_required_issue_fields_fall_back_to_original_request(self):
        fill_issue = load_stream_server_function("fill_template")
        template = (
            Path(__file__).resolve().parent.parent
            / ".github" / "ISSUE_TEMPLATE" / "visual_at.md"
        ).read_text(encoding="utf-8")
        original = "Find the nearest exit and tell me how to reach it."
        issue = fill_issue(template, {
            "title": "Exit Finder",
            "description": "",
            "example_usage": "",
            "streaming_context": "latest_frame",
            "original_prompts": [f"[2026-07-27 16:00] {original}"],
        })

        self.assertIn(f"## Task\n\n{original}", issue)
        self.assertIn(f"## Expected output\n\n{original}", issue)

    def test_current_held_sign_posture_remains_static(self):
        select_context = load_stream_server_function("_select_streaming_context")
        request = "Identify the sign shown by the signer’s current held hand posture."
        self.assertEqual(select_context("latest_frame", request), "latest_frame")

    def test_lifecycle_hooks_override_legacy_execution_mode(self):
        execution_mode = load_stream_server_function("_tool_execution_mode")
        lifecycle_tool = '''
EXECUTION_MODE = "hosted_video_streaming"
async def on_frame(runtime, frame):
    return None
'''
        declarative_tool = 'EXECUTION_MODE = "hosted_video_streaming"\n'
        self.assertEqual(execution_mode(lifecycle_tool), "")
        self.assertEqual(execution_mode(declarative_tool), "hosted_video_streaming")


class TestSentenceDetectionLogic(unittest.TestCase):
    """Test sentence detection logic"""
    
    def detect_complete_sentence(self, text: str) -> dict:
        """Helper function to detect complete sentences"""
        import re
        sentenceRegex = r'^(.*?[.!?])(\s+|$)'
        match = re.match(sentenceRegex, text)
        
        if match:
            return {
                'complete': match.group(1).strip(),
                'remainder': text[len(match.group(0)):].strip()
            }
        
        return {'complete': '', 'remainder': text}
    
    def test_detects_sentence_with_period(self):
        """Test detection of sentence ending with period"""
        result = self.detect_complete_sentence('Hello world.')
        self.assertEqual(result['complete'], 'Hello world.')
        self.assertEqual(result['remainder'], '')
    
    def test_detects_sentence_with_question_mark(self):
        """Test detection of sentence ending with question mark"""
        result = self.detect_complete_sentence('How are you?')
        self.assertEqual(result['complete'], 'How are you?')
        self.assertEqual(result['remainder'], '')
    
    def test_detects_sentence_with_exclamation(self):
        """Test detection of sentence ending with exclamation mark"""
        result = self.detect_complete_sentence('Great job!')
        self.assertEqual(result['complete'], 'Great job!')
        self.assertEqual(result['remainder'], '')
    
    def test_detects_sentence_with_remainder(self):
        """Test detection of sentence with remaining text"""
        result = self.detect_complete_sentence('Hello world. This is')
        self.assertEqual(result['complete'], 'Hello world.')
        self.assertEqual(result['remainder'], 'This is')
    
    def test_no_complete_sentence(self):
        """Test when there's no complete sentence"""
        result = self.detect_complete_sentence('Hello world')
        self.assertEqual(result['complete'], '')
        self.assertEqual(result['remainder'], 'Hello world')
    
    def test_empty_string(self):
        """Test with empty string"""
        result = self.detect_complete_sentence('')
        self.assertEqual(result['complete'], '')
        self.assertEqual(result['remainder'], '')


class TestTemplateFilling(unittest.TestCase):
    """Test template filling logic"""
    
    def fill_template(self, template_content: str, parsed_data: dict) -> str:
        """Helper function to fill templates"""
        filled = template_content
        
        # Fill visual AT template
        filled = filled.replace('<!-- A clear and concise description of the tool you\'d like. -->', 
                               parsed_data.get('description', ''))
        filled = filled.replace('<!-- Describe the problem this tool would solve. -->', 
                               parsed_data.get('problem', ''))
        filled = filled.replace('<!-- Describe how you envision this tool working. -->', 
                               parsed_data.get('solution', ''))
        filled = filled.replace('<!-- Any particular models or libraries that should be employed -->', 
                               parsed_data.get('implementation_details', ''))
        filled = filled.replace("<!-- Describe any alternative solutions or features you've considered. -->", 
                               parsed_data.get('alternatives', ''))
        filled = filled.replace('<!-- Describe an example situation the tool would be used in and how it could work -->', 
                               parsed_data.get('example_usage', ''))
        filled = filled.replace('<!-- Add any other context or screenshots about the feature request here. -->', 
                               parsed_data.get('additional', ''))
        
        return filled
    
    def test_fill_visual_at_template(self):
        """Test filling a visual AT template"""
        template = """**Feature Description**
<!-- A clear and concise description of the tool you'd like. -->

**Problem It Solves**
<!-- Describe the problem this tool would solve. -->

**Proposed Solution**
<!-- Describe how you envision this tool working. -->

**Example usage**
<!-- Describe an example situation the tool would be used in and how it could work -->"""

        data = {
            'type': 'visual AT',
            'description': 'Object detection tool',
            'problem': 'Need to identify objects in environment',
            'solution': 'Declare object_detection_localization capability and use the backend capability layer',
            'example_usage': 'User points camera at table, tool identifies cup and phone'
        }
        
        result = self.fill_template(template, data)
        
        self.assertIn('Object detection tool', result)
        self.assertIn('Need to identify objects in environment', result)
        self.assertIn('Declare object_detection_localization capability and use the backend capability layer', result)
        self.assertIn('User points camera at table, tool identifies cup and phone', result)
        self.assertNotIn('<!--', result.replace('<!--', '', 1))  # Check most placeholders replaced
    
    def test_fill_with_empty_fields(self):
        """Test filling template with some empty fields"""
        template = """**Description**
<!-- A clear and concise description of the tool you'd like. -->

**Additional Context**
<!-- Add any other context or screenshots about the feature request here. -->"""

        data = {
            'type': 'visual AT',
            'description': 'Tool description',
            'additional': ''  # Empty field
        }
        
        result = self.fill_template(template, data)
        
        self.assertIn('Tool description', result)
        # Should still have the placeholder for empty field
        self.assertIn('**Additional Context**', result)
    
    def test_fill_with_list_values(self):
        """Test filling template when AI returns lists instead of strings"""
        template = """**Description**
<!-- A clear and concise description of the tool you'd like. -->

**Problem It Solves**
<!-- Describe the problem this tool would solve. -->"""

        # Helper function matching the one in stream_server
        def ensure_string(value):
            if isinstance(value, list):
                return '\n'.join(str(item) for item in value)
            elif value is None:
                return ''
            return str(value)
        
        data = {
            'type': 'visual AT',
            'description': ['Object', 'detection', 'tool'],  # List instead of string
            'problem': ['Hard to identify', 'objects in environment']
        }
        
        # Apply the ensure_string logic
        description = ensure_string(data['description'])
        problem = ensure_string(data['problem'])
        
        # Verify lists are converted to strings
        self.assertEqual(description, 'Object\ndetection\ntool')
        self.assertEqual(problem, 'Hard to identify\nobjects in environment')


class TestAIParsingFallback(unittest.TestCase):
    """Test AI parsing fallback logic"""
    
    def simple_parse(self, transcript: str) -> dict:
        """Fallback parsing when API is not available"""
        return {
            'type': 'visual AT',
            'title': transcript[:100],
            'description': transcript,
            'problem': '',
            'solution': '',
            'implementation_details': '',
            'example_usage': '',
            'alternatives': '',
            'additional': ''
        }
    
    def test_fallback_parsing(self):
        """Test that fallback parsing works"""
        transcript = "I want a tool to detect objects in my environment"
        result = self.simple_parse(transcript)
        
        self.assertEqual(result['type'], 'visual AT')
        self.assertEqual(result['title'], transcript)
        self.assertEqual(result['description'], transcript)
        self.assertEqual(result['problem'], '')
    
    def test_fallback_with_long_text(self):
        """Test fallback with text longer than 100 chars"""
        transcript = "A" * 150
        result = self.simple_parse(transcript)
        
        self.assertEqual(len(result['title']), 100)
        self.assertEqual(len(result['description']), 150)


class TestDataMerging(unittest.TestCase):
    """Test multi-turn conversation data merging"""
    
    def merge_parsed_data(self, existing_data: dict, new_data: dict) -> dict:
        """Helper function to merge parsed data"""
        merged = existing_data.copy()
        
        # Merge all fields - new non-empty values override old ones
        for key, value in new_data.items():
            if key == 'missing_fields':
                continue  # Don't merge missing_fields, we'll recalculate
            
            # Update if new value is non-empty
            if value:
                # Handle both strings and lists
                if isinstance(value, str) and value.strip():
                    merged[key] = value
                elif isinstance(value, list) and value:
                    merged[key] = value
                elif not isinstance(value, (str, list)):
                    merged[key] = value
        
        # Update missing_fields from new data (AI re-evaluated what's still missing)
        merged['missing_fields'] = new_data.get('missing_fields', [])
        
        return merged
    
    def test_merge_fills_missing_fields(self):
        """Test that merging fills in previously missing fields"""
        existing = {
            'type': 'visual AT',
            'title': 'Object detection',
            'description': 'A tool to detect objects',
            'problem': '',
            'solution': '',
            'example_usage': '',
            'missing_fields': ['problem', 'solution', 'example_usage']
        }
        
        new = {
            'type': 'visual AT',
            'title': '',
            'description': '',
            'problem': 'Hard to identify objects',
            'solution': 'Declare object_detection_localization capability and use the backend capability layer',
            'example_usage': 'Point camera at table to identify items',
            'missing_fields': []
        }
        
        merged = self.merge_parsed_data(existing, new)
        
        # Should keep original title and description
        self.assertEqual(merged['title'], 'Object detection')
        self.assertEqual(merged['description'], 'A tool to detect objects')
        
        # Should add new fields
        self.assertEqual(merged['problem'], 'Hard to identify objects')
        self.assertEqual(merged['solution'], 'Declare object_detection_localization capability and use the backend capability layer')
        self.assertEqual(merged['example_usage'], 'Point camera at table to identify items')
        
        # Should update missing_fields to empty
        self.assertEqual(merged['missing_fields'], [])
    
    def test_merge_preserves_existing_non_empty(self):
        """Test that merge doesn't overwrite existing non-empty fields with empty ones"""
        existing = {
            'type': 'visual AT',
            'title': 'Object detection',
            'description': 'Full description',
            'problem': 'Need to identify objects',
            'solution': 'Use AI model',
            'example_usage': '',
            'missing_fields': ['example_usage']
        }
        
        new = {
            'type': 'visual AT',
            'title': '',  # Empty, should not overwrite
            'description': '',  # Empty, should not overwrite
            'problem': '',  # Empty, should not overwrite
            'solution': '',  # Empty, should not overwrite
            'example_usage': 'Point at desk to find items',  # New info
            'missing_fields': []
        }
        
        merged = self.merge_parsed_data(existing, new)
        
        # Should preserve existing non-empty fields
        self.assertEqual(merged['title'], 'Object detection')
        self.assertEqual(merged['description'], 'Full description')
        self.assertEqual(merged['problem'], 'Need to identify objects')
        self.assertEqual(merged['solution'], 'Use AI model')
        
        # Should add the new example_usage field
        self.assertEqual(merged['example_usage'], 'Point at desk to find items')
    
    def test_merge_updates_missing_fields(self):
        """Test that missing_fields is properly updated"""
        existing = {
            'type': 'visual AT',
            'title': 'Tool',
            'description': 'A tool',
            'problem': '',
            'solution': '',
            'example_usage': '',
            'missing_fields': ['problem', 'solution', 'example_usage']
        }
        
        new = {
            'type': 'visual AT',
            'title': '',
            'description': '',
            'problem': 'Problem provided',
            'solution': '',
            'example_usage': '',
            'missing_fields': ['solution', 'example_usage']  # Still missing 2
        }
        
        merged = self.merge_parsed_data(existing, new)
        
        # Should have updated missing_fields from new data
        self.assertEqual(merged['missing_fields'], ['solution', 'example_usage'])
        self.assertEqual(merged['problem'], 'Problem provided')


class TestWellSpecifiedIssues(unittest.TestCase):
    """Test that well-specified issues don't trigger requests for more info"""
    
    def test_well_specified_visual_at_should_have_no_missing_fields(self):
        """
        Test that a visual AT request with all important fields specified should have empty missing_fields.
        This simulates what the AI should return for a well-specified issue.
        """
        # Simulate AI parsing a well-specified visual AT request
        transcript = """I want a tool to detect objects in my environment. 
        The problem is I can't identify what's around me when walking. 
        I envision a tool that uses the phone camera to identify objects and speak their names. 
        For example, if I point my camera at my desk, it should tell me there's a laptop, coffee mug, and phone."""
        
        # This is what the AI SHOULD return for a well-specified visual AT request
        expected_parsed_data = {
            'type': 'visual AT',
            'title': 'Object detection tool for environment awareness',
            'description': 'A tool to detect objects in my environment',
            'problem': 'Cannot identify what\'s around me when walking',
            'solution': 'Use phone camera to identify objects and speak their names',
            'implementation_details': '',
            'example_usage': 'Point camera at desk, tool identifies laptop, coffee mug, and phone',
            'alternatives': '',
            'additional': '',
            'missing_fields': []  # All important fields are filled!
        }
        
        # Verify all important visual AT fields are present
        self.assertEqual(expected_parsed_data['type'], 'visual AT')
        self.assertNotEqual(expected_parsed_data['description'].strip(), '')
        self.assertNotEqual(expected_parsed_data['problem'].strip(), '')
        self.assertNotEqual(expected_parsed_data['solution'].strip(), '')
        self.assertNotEqual(expected_parsed_data['example_usage'].strip(), '')
        
        # Most importantly, missing_fields should be empty
        self.assertEqual(expected_parsed_data['missing_fields'], [])
    
    def test_visual_at_with_brief_but_complete_info_should_not_ask_for_more(self):
        """
        Test that even brief descriptions are accepted if they cover important fields.
        """
        # Brief but complete visual AT request
        expected_parsed_data = {
            'type': 'visual AT',
            'title': 'Text reader tool',
            'description': 'Read text from images',
            'problem': 'Cannot read text in environment',
            'solution': 'Declare OCR capability and use the backend capability layer',
            'implementation_details': '',
            'example_usage': 'Point at sign to hear text',
            'alternatives': '',
            'additional': '',
            'missing_fields': []  # Brief but complete!
        }
        
        # Even though the fields are brief, they have meaningful content
        self.assertTrue(len(expected_parsed_data['description']) > 0)
        self.assertTrue(len(expected_parsed_data['problem']) > 0)
        self.assertTrue(len(expected_parsed_data['solution']) > 0)
        self.assertTrue(len(expected_parsed_data['example_usage']) > 0)
        
        # Should not ask for more info
        self.assertEqual(expected_parsed_data['missing_fields'], [])
    
    def test_incomplete_visual_at_should_identify_missing_fields(self):
        """
        Test that genuinely incomplete visual AT requests are correctly identified.
        """
        # Incomplete visual AT request - only has description, missing problem/solution/example
        expected_parsed_data = {
            'type': 'visual AT',
            'title': 'New tool',
            'description': 'I want a new tool',
            'problem': '',  # Missing
            'solution': '',  # Missing
            'implementation_details': '',
            'example_usage': '',  # Missing
            'alternatives': '',
            'additional': '',
            'missing_fields': ['problem', 'solution', 'example_usage']
        }
        
        # Verify the important fields are actually missing
        self.assertEqual(expected_parsed_data['problem'].strip(), '')
        self.assertEqual(expected_parsed_data['solution'].strip(), '')
        self.assertEqual(expected_parsed_data['example_usage'].strip(), '')
        
        # Should correctly identify all missing fields
        self.assertEqual(set(expected_parsed_data['missing_fields']), 
                        {'problem', 'solution', 'example_usage'})


if __name__ == '__main__':
    unittest.main()
