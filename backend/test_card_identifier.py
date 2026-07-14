#!/usr/bin/env python3
"""Focused tests for the card_identifier take-photo tool."""

import os
import sys
import types
import unittest


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, TOOLS_DIR)

if "litellm_utils" not in sys.modules:
    litellm_utils_module = types.ModuleType("litellm_utils")

    def _default_call_take_photo_baseline_vlm(*, image, prompt, tool_name="take_photo_tool"):
        return "No readable cards visible."

    litellm_utils_module.call_take_photo_baseline_vlm = _default_call_take_photo_baseline_vlm
    sys.modules["litellm_utils"] = litellm_utils_module

import card_identifier  # noqa: E402


class TestCardIdentifier(unittest.TestCase):
    def test_returns_no_image_message(self):
        result = card_identifier.main(None, {})
        self.assertEqual(result, "No camera image available to read cards.")

    def test_calls_baseline_vlm_with_prompt(self):
        calls = {}

        def fake_call_take_photo_baseline_vlm(*, image, prompt, tool_name):
            calls["image"] = image
            calls["prompt"] = prompt
            calls["tool_name"] = tool_name
            return "Ace of spades, ten of hearts."

        original = card_identifier.call_take_photo_baseline_vlm
        card_identifier.call_take_photo_baseline_vlm = fake_call_take_photo_baseline_vlm
        try:
            image = object()
            result = card_identifier.main(image, {})
        finally:
            card_identifier.call_take_photo_baseline_vlm = original

        self.assertEqual(result, "Ace of spades, ten of hearts.")
        self.assertIs(calls["image"], image)
        self.assertEqual(calls["prompt"], card_identifier.TOOL_PROMPT)
        self.assertEqual(calls["tool_name"], card_identifier.TOOL_NAME)

    def test_returns_fallback_message_on_exception(self):
        def fake_call_take_photo_baseline_vlm(*, image, prompt, tool_name):
            raise RuntimeError("simulated failure")

        original = card_identifier.call_take_photo_baseline_vlm
        card_identifier.call_take_photo_baseline_vlm = fake_call_take_photo_baseline_vlm
        try:
            result = card_identifier.main(object(), {})
        finally:
            card_identifier.call_take_photo_baseline_vlm = original

        self.assertEqual(
            result,
            "I couldn't read the cards from this photo. Please hold them steady and try again.",
        )

    def test_returns_fallback_message_on_type_error(self):
        def fake_call_take_photo_baseline_vlm(*, image, prompt, tool_name):
            raise TypeError("bad image")

        original = card_identifier.call_take_photo_baseline_vlm
        card_identifier.call_take_photo_baseline_vlm = fake_call_take_photo_baseline_vlm
        try:
            result = card_identifier.main(object(), {})
        finally:
            card_identifier.call_take_photo_baseline_vlm = original

        self.assertEqual(
            result,
            "I couldn't read the cards from this photo. Please hold them steady and try again.",
        )


if __name__ == "__main__":
    unittest.main()
