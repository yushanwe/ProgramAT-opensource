#!/usr/bin/env python3
"""Focused tests for hand_gesture_detection tool stage flow."""

import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

stub_router_client = types.ModuleType("model_router_client")
stub_router_client.copilot_llm_call = lambda **_: {"response": ""}
sys.modules["model_router_client"] = stub_router_client

import hand_gesture_detection


class TestHandGestureDetection(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.calls = []

    def _install_stub(self, stage1_result, stage2_response="Person is waving hello. A thumbs up usually means approval."):
        def _stub(**kwargs):
            self.calls.append(kwargs)
            if kwargs.get("capability") == "spatial_reasoning":
                return stage1_result
            return {"response": stage2_response}

        hand_gesture_detection.copilot_llm_call = _stub

    def test_two_stage_flow_passes_artifact_when_usable(self):
        self._install_stub(
            {
                "response": "A person on your left is waving and giving a thumbs up.",
                "artifact": {"gestures": ["waving", "thumbs up"], "confidence": 0.9},
                "confidence": 0.9,
            }
        )

        result = hand_gesture_detection.main(self.image, {})

        self.assertIn("waving", result.lower())
        self.assertEqual([c["capability"] for c in self.calls], ["spatial_reasoning", "general_reasoning"])
        self.assertIs(self.calls[0]["images"][0], self.image)
        self.assertIs(self.calls[1]["images"][0], self.image)
        self.assertIn("previous_stage_artifact", self.calls[1]["metadata"])

    def test_low_confidence_stage1_does_not_pass_artifact(self):
        self._install_stub(
            {
                "response": "Unclear hand shape.",
                "artifact": {"gestures": ["unknown"], "confidence": 0.2},
                "confidence": 0.2,
            }
        )

        hand_gesture_detection.main(self.image, {})

        self.assertNotIn("previous_stage_artifact", self.calls[1]["metadata"])

    def test_no_image_returns_audio_friendly_error(self):
        result = hand_gesture_detection.main(None, {})

        self.assertIsInstance(result, dict)
        self.assertEqual(result["audio"]["type"], "error")
        self.assertIn("No camera image", result["text"])


if __name__ == "__main__":
    unittest.main()
