"""Standalone tests for tools/plug_point_finder.py."""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

def mock_empty_detection_response(**kwargs):
    return {"response": "", "artifact": {"detections": []}}


mock_router_client = types.ModuleType("model_router_client")
mock_router_client.copilot_llm_call = mock_empty_detection_response
sys.modules.setdefault("model_router_client", mock_router_client)

import plug_point_finder as tool


class FakeImage:
    def __init__(self, height: int, width: int, mean_value: float, std_value: float = 0.0):
        self.shape = (height, width, 3)
        self.size = height * width * 3
        self._mean = mean_value
        self._std = std_value

    def mean(self):
        return self._mean

    def std(self):
        return self._std


class TestPlugPointFinder(unittest.TestCase):
    def setUp(self):
        tool.reset_state()
        self._original_call = tool.copilot_llm_call

    def tearDown(self):
        tool.copilot_llm_call = self._original_call
        tool.reset_state()

    def test_reports_no_plug_points(self):
        image = FakeImage(120, 160, mean_value=120, std_value=20)

        def fake_call(**kwargs):
            return {"response": "No requested object was detected.", "artifact": {"detections": []}}

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {})
        self.assertEqual(result, "No plug points found.")

    def test_reports_count_and_direction(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)
        called_capabilities = []

        def fake_call(**kwargs):
            capability = kwargs.get("capability")
            called_capabilities.append(capability)
            if capability == "general_reasoning":
                return {"response": "2"}
            return {
                "response": "detected",
                "artifact": {
                    "detections": [
                        {"label": "electrical outlet", "bbox": [65, 10, 95, 50]},
                        {"label": "electrical outlet", "bbox": [5, 10, 20, 40]},
                    ]
                },
            }

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {})
        self.assertEqual(result, "2 outlets visible, closest one at 2 o'clock.")
        self.assertIn("general_reasoning", called_capabilities)

    def test_counts_individual_sockets_not_strips(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)

        def fake_call(**kwargs):
            capability = kwargs.get("capability")
            if capability == "object_detection_localization":
                return {
                    "response": "detected",
                    "artifact": {
                        "detections": [
                            {"label": "power strip", "bbox": [40, 15, 85, 55]},
                        ]
                    },
                }
            if capability == "general_reasoning":
                return {"response": "There are 3 visible sockets.", "artifact": {"socket_count": 3}}
            raise AssertionError(f"Unexpected capability call: {capability}")

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {})
        self.assertEqual(result, "3 outlets visible, closest one at 1 o'clock.")

    def test_falls_back_to_detection_count_when_socket_counting_fails(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)

        def fake_call(**kwargs):
            capability = kwargs.get("capability")
            if capability == "object_detection_localization":
                return {
                    "response": "detected",
                    "artifact": {
                        "detections": [
                            {"label": "wall outlet", "bbox": [15, 20, 35, 55]},
                            {"label": "wall outlet", "bbox": [55, 20, 75, 55]},
                        ]
                    },
                }
            if capability == "general_reasoning":
                raise RuntimeError("temporary failure")
            raise AssertionError(f"Unexpected capability call: {capability}")

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {})
        self.assertEqual(result, "2 outlets visible, closest one at 10 o'clock.")

    def test_ignores_non_count_numbers_in_socket_response(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)

        def fake_call(**kwargs):
            capability = kwargs.get("capability")
            if capability == "object_detection_localization":
                return {
                    "response": "detected",
                    "artifact": {
                        "detections": [
                            {"label": "wall outlet", "bbox": [15, 20, 35, 55]},
                            {"label": "wall outlet", "bbox": [55, 20, 75, 55]},
                        ]
                    },
                }
            if capability == "general_reasoning":
                return {"response": "Closest appears at 2 o'clock."}
            raise AssertionError(f"Unexpected capability call: {capability}")

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {})
        self.assertEqual(result, "2 outlets visible, closest one at 10 o'clock.")

    def test_parses_socket_count_even_when_response_includes_clock_position(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)

        def fake_call(**kwargs):
            capability = kwargs.get("capability")
            if capability == "object_detection_localization":
                return {
                    "response": "detected",
                    "artifact": {
                        "detections": [
                            {"label": "wall outlet", "bbox": [40, 20, 80, 60]},
                        ]
                    },
                }
            if capability == "general_reasoning":
                return {"response": "There are 4 sockets visible, closest at 2 o'clock."}
            raise AssertionError(f"Unexpected capability call: {capability}")

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {})
        self.assertEqual(result, "4 outlets visible, closest one straight ahead.")

    def test_parses_socket_count_from_number_words(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)

        def fake_call(**kwargs):
            capability = kwargs.get("capability")
            if capability == "object_detection_localization":
                return {
                    "response": "detected",
                    "artifact": {
                        "detections": [
                            {"label": "power strip", "bbox": [35, 20, 85, 60]},
                        ]
                    },
                }
            if capability == "general_reasoning":
                return {"response": "Two outlets are visible on this strip."}
            raise AssertionError(f"Unexpected capability call: {capability}")

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {})
        self.assertEqual(result, "2 outlets visible, closest one straight ahead.")

    def test_persistent_blocked_camera_warns(self):
        image = FakeImage(80, 80, mean_value=0, std_value=0)

        first = tool.main(image, {"is_streaming": True, "blocked_frame_threshold": 3})
        second = tool.main(image, {"is_streaming": True, "blocked_frame_threshold": 3})
        third = tool.main(image, {"is_streaming": True, "blocked_frame_threshold": 3})

        self.assertEqual(first, "")
        self.assertEqual(second, "")
        self.assertIsInstance(third, dict)
        self.assertEqual(third.get("audio", {}).get("type"), "warning")

    def test_streaming_response_is_short(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)

        def fake_call(**kwargs):
            return {
                "response": "detected",
                "artifact": {
                    "detections": [
                        {"label": "electrical outlet", "bbox": [40, 20, 60, 70]},
                    ]
                },
            }

        tool.copilot_llm_call = fake_call
        result = tool.main(image, {"is_streaming": True})
        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result.split()), 15)


if __name__ == "__main__":
    unittest.main()
