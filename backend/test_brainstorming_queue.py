import ast
import asyncio
import json
import logging
import re
import unittest
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def load_stream_server_function(name: str):
    source_path = Path(__file__).resolve().parent / "stream_server.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "SequenceMatcher": SequenceMatcher,
        "asyncio": asyncio,
        "call_model": lambda *args, **kwargs: None,
        "extract_text": lambda response: "",
        "json": json,
        "logger": logging.getLogger("test_brainstorming_queue"),
        "re": re,
    }
    exec(
        compile(ast.Module(body=[function_node], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace[name]


class BrainstormingQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_filters_already_asked_question_and_close_paraphrase(self):
        generate_queue = load_stream_server_function("generate_ranked_question_queue")

        class FakeResponse:
            pass

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        response = FakeResponse()
        generate_queue.__globals__["asyncio"] = SimpleNamespace(to_thread=fake_to_thread)
        generate_queue.__globals__["call_model"] = lambda *args, **kwargs: response
        generate_queue.__globals__["extract_text"] = lambda _response: "\n".join([
            "1. What existing behavior must stay the same while adding this update?",
            "2. Which current behavior should remain unchanged during this update?",
            "3. Should the streaming mode keep speaking on every frame, or only when the result changes?",
        ])

        questions = await generate_queue(
            {
                "title": "Uber car identifier",
                "description": "Add streaming mode and make announcements less repetitive.",
                "solution": "",
                "example_usage": "",
            },
            "",
            brainstorm_history=[{
                "question": "What existing behavior must stay the same while adding this update?",
                "answer": "Keep single-frame behavior the same.",
            }],
            existing_queue=["Should the spoken result include color and license plate when available?"],
            mode="update",
            update_context={
                "repository": "owner/repo",
                "issue_number": 12,
                "pr_number": 34,
                "tool_name": "uber_car_identifier",
                "tool_path": "tools/uber_car_identifier.py",
                "tool_source": "def on_frame(): pass",
                "implementation_metadata": {},
                "implementation_summary": {},
                "action_log": "",
                "issue_body": "",
                "pr_body": "",
                "claude_summary": "",
                "pr_diff": "",
            },
        )

        self.assertIn("Should the spoken result include color and license plate when available?", questions)
        self.assertIn("Should the streaming mode keep speaking on every frame, or only when the result changes?", questions)
        self.assertNotIn("What existing behavior must stay the same while adding this update?", questions)
        self.assertNotIn("Which current behavior should remain unchanged during this update?", questions)


if __name__ == "__main__":
    unittest.main()
