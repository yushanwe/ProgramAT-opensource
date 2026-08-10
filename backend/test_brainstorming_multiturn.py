import ast
import asyncio
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from brainstorming import build_clarified_update_request, format_brainstorm_transcript


SOURCE_PATH = Path(__file__).resolve().parent / "stream_server.py"


def load_function(name: str, namespace: Optional[Dict[str, Any]] = None):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    globals_dict = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "asyncio": asyncio,
        "json": json,
        "logger": SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
        ),
        "re": re,
    }
    globals_dict.update(namespace or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"), globals_dict)
    return globals_dict[name]


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def json_response(payload, status=200):
    return {"payload": payload, "status": status}


class BrainstormTurnClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_classifier_handles_question_without_punctuation(self):
        classifier = load_function(
            "classify_brainstorm_turn",
            {
                "system_llm_call": lambda **_kwargs: {"text": "ignored"},
                "extract_text": lambda _response: '{"kind":"clarification_question"}',
                "_parse_llm_json_object": json.loads,
            },
        )

        result = await classifier(
            "How often should it speak?",
            "can you explain what the choices mean",
        )

        self.assertEqual("clarification_question", result)

    async def test_fallback_distinguishes_question_and_answer(self):
        def fail_model(**_kwargs):
            raise RuntimeError("offline")

        classifier = load_function(
            "classify_brainstorm_turn",
            {
                "system_llm_call": fail_model,
                "extract_text": lambda _response: "",
                "_parse_llm_json_object": json.loads,
            },
        )

        self.assertEqual(
            "clarification_question",
            await classifier("Choose a mode", "what does streaming mean"),
        )
        self.assertEqual(
            "substantive_answer",
            await classifier("Choose a mode", "Use streaming and announce only changes"),
        )
        self.assertEqual("substantive_answer", await classifier("Enable alerts?", "yes"))
        self.assertEqual("substantive_answer", await classifier("Choose a model", "Use a VLM instead"))

    async def test_unknown_model_classification_defaults_to_clarification(self):
        classifier = load_function(
            "classify_brainstorm_turn",
            {
                "system_llm_call": lambda **_kwargs: {"text": "ignored"},
                "extract_text": lambda _response: '{"kind":"uncertain"}',
                "_parse_llm_json_object": json.loads,
            },
        )

        self.assertEqual(
            "clarification_question",
            await classifier("Choose a mode", "maybe something else"),
        )


class BrainstormTurnHandlerTests(unittest.IsolatedAsyncioTestCase):
    def make_handler(self, session, classifications):
        pending = {"token": session}
        summary_calls = []
        classification_iter = iter(classifications)

        async def classify(*_args):
            return next(classification_iter)

        async def answer(*_args):
            return "Streaming checks frames continuously."

        async def follow_up(*_args):
            return "Should it speak only when the result changes?"

        async def summarize(*args, **kwargs):
            summary_calls.append((args, kwargs))
            return "Updated understanding", "Only announce changes"

        async def broadcast(_payload):
            return None

        def append_transcript(entry, kind, text):
            transcript = list(entry.get("brainstorm_transcript") or [])
            transcript.append({"kind": kind, "text": text})
            entry["brainstorm_transcript"] = transcript

        handler = load_function(
            "handle_brainstorm_ask_agent",
            {
                "web": SimpleNamespace(
                    Request=object,
                    Response=object,
                    json_response=json_response,
                ),
                "pending_ideation_http": pending,
                "classify_brainstorm_turn": classify,
                "answer_brainstorm_question": answer,
                "generate_brainstorm_clarification_followup": follow_up,
                "generate_understanding_summary": summarize,
                "_broadcast_ws": broadcast,
                "_append_brainstorm_transcript": append_transcript,
            },
        )
        return handler, pending, summary_calls

    async def test_clarifications_do_not_commit_or_update_summary(self):
        session = {
            "parsed_data": {"title": "Reader"},
            "video_summary": "",
            "brainstorm_history": [],
            "clarification_history": [],
            "last_question": "How often should it speak?",
            "last_summary": "Original understanding",
            "phase": "awaiting_answer",
        }
        handler, pending, summary_calls = self.make_handler(
            session,
            ["clarification_question", "clarification_question"],
        )

        first = await handler(FakeRequest({"token": "token", "text": "what is streaming"}))
        second = await handler(FakeRequest({"token": "token", "text": "does that use more data"}))

        self.assertEqual("clarification", first["payload"]["status"])
        self.assertEqual("clarification", second["payload"]["status"])
        self.assertEqual("Original understanding", second["payload"]["summary"])
        self.assertEqual([], pending["token"]["brainstorm_history"])
        self.assertEqual("Original understanding", pending["token"]["last_summary"])
        self.assertEqual(2, len(pending["token"]["clarification_history"]))
        self.assertEqual(6, len(pending["token"]["brainstorm_transcript"]))
        self.assertEqual([], summary_calls)

    async def test_substantive_answer_commits_once_and_moves_to_choice(self):
        session = {
            "mode": "update",
            "parsed_data": {"title": "Reader"},
            "video_summary": "",
            "brainstorm_history": [],
            "clarification_history": [{"user_question": "what is streaming"}],
            "last_question": "Should it speak only when the result changes?",
            "last_summary": "Original understanding",
            "phase": "awaiting_answer",
            "update_context": {"tool_name": "reader"},
        }
        handler, pending, summary_calls = self.make_handler(session, ["substantive_answer"])

        response = await handler(FakeRequest({
            "token": "token",
            "text": "Yes, only announce when the result changes",
        }))

        self.assertEqual("brainstorm_choice", response["payload"]["status"])
        self.assertEqual("awaiting_choice", pending["token"]["phase"])
        self.assertEqual([], pending["token"]["clarification_history"])
        self.assertEqual(1, len(pending["token"]["brainstorm_history"]))
        self.assertEqual(
            "Should it speak only when the result changes?",
            pending["token"]["brainstorm_history"][0]["question"],
        )
        self.assertEqual(1, len(summary_calls))
        self.assertEqual(
            [{"user_question": "what is streaming"}],
            summary_calls[0][1]["clarification_context"],
        )
        self.assertEqual("user_answer", pending["token"]["brainstorm_transcript"][-1]["kind"])

    async def test_requirement_outside_question_commits_and_exact_retry_is_idempotent(self):
        session = {
            "parsed_data": {"title": "Reader"},
            "video_summary": "",
            "brainstorm_history": [{"question": "Frequency?", "answer": "On changes"}],
            "clarification_history": [],
            "brainstorm_transcript": [
                {"kind": "assistant_question", "text": "Frequency?"},
                {"kind": "user_answer", "text": "On changes"},
            ],
            "last_question": "Frequency?",
            "last_summary": "Speak on changes",
            "phase": "awaiting_choice",
        }
        handler, pending, summary_calls = self.make_handler(
            session,
            ["substantive_answer", "substantive_answer"],
        )

        payload = {"token": "token", "text": "Also use a VLM instead of OCR."}
        first = await handler(FakeRequest(payload))
        second = await handler(FakeRequest(payload))

        self.assertEqual("brainstorm_choice", first["payload"]["status"])
        self.assertEqual("brainstorm_choice", second["payload"]["status"])
        self.assertEqual(2, len(pending["token"]["brainstorm_history"]))
        self.assertEqual(
            "User-provided requirement or correction",
            pending["token"]["brainstorm_history"][-1]["question"],
        )
        self.assertEqual("user_requirement", pending["token"]["brainstorm_transcript"][-1]["kind"])
        self.assertEqual(1, len(summary_calls))
        self.assertEqual("user_statement", summary_calls[0][1]["latest_qa_role"])

    async def test_normal_agent_question_in_choice_phase_has_no_follow_up(self):
        session = {
            "parsed_data": {"title": "Reader"},
            "video_summary": "",
            "brainstorm_history": [{"question": "Frequency?", "answer": "On changes"}],
            "clarification_history": [],
            "brainstorm_transcript": [
                {"kind": "assistant_question", "text": "Frequency?"},
                {"kind": "user_answer", "text": "On changes"},
            ],
            "last_question": "Frequency?",
            "last_summary": "Speak on changes",
            "phase": "awaiting_choice",
        }
        handler, pending, summary_calls = self.make_handler(session, ["clarification_question"])

        response = await handler(FakeRequest({
            "token": "token",
            "text": "What model would work well for that?",
        }))

        self.assertEqual("agent_answer", response["payload"]["status"])
        self.assertEqual("awaiting_choice", pending["token"]["phase"])
        self.assertEqual("Frequency?", pending["token"]["last_question"])
        self.assertEqual([], pending["token"]["clarification_history"])
        self.assertEqual(2, len(pending["token"]["brainstorm_transcript"]))
        self.assertEqual([], summary_calls)


class BrainstormTranscriptTests(unittest.TestCase):
    def test_full_transcript_preserves_context_for_referential_answer(self):
        transcript = [
            {"kind": "assistant_question", "text": "How should an illegible plate be handled?"},
            {"kind": "user_clarification", "text": "What options are available?"},
            {"kind": "assistant_clarification", "text": "Retry, report uncertainty, or ask for another angle."},
            {"kind": "assistant_question", "text": "Which option should be prioritized?"},
            {"kind": "user_answer", "text": "The last option would be best."},
        ]

        formatted = format_brainstorm_transcript(transcript)
        update = build_clarified_update_request(
            "Improve plate handling.",
            ["Which option should be prioritized?"],
            ["The last option would be best."],
            transcript,
        )

        self.assertIn("Assistant clarification answer: Retry, report uncertainty, or ask for another angle.", formatted)
        self.assertIn("Q: Which option should be prioritized?", update)
        self.assertIn("A: The last option would be best.", update)


if __name__ == "__main__":
    unittest.main()
