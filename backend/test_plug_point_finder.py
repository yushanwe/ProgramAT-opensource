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

    def test_reports_count_and_direction_for_all_plug_points(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)

        def fake_call(**kwargs):
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
        self.assertEqual(result, "2 plug points visible: 9 o'clock and 2 o'clock.")

    def test_reports_grouped_directions_for_multiple_plug_points(self):
        image = FakeImage(100, 100, mean_value=160, std_value=20)
        detections = [
            {"label": "electrical outlet", "bbox": [26, 10, 46, 50]},
            {"label": "electrical outlet", "bbox": [28, 10, 48, 50]},
            {"label": "electrical outlet", "bbox": [65, 10, 85, 50]},
        ]

        def fake_call(**kwargs):
            return {
                "response": "detected",
                "artifact": {"detections": detections},
            }

        tool.copilot_llm_call = fake_call
        ordered = tool._sort_detections_left_to_right(detections)
        directions = [tool._clock_direction_from_bbox(detection, image.shape[1]) for detection in ordered]
        self.assertEqual(directions.count("11 o'clock"), 2)
        result = tool.main(image, {})
        self.assertEqual(result, "3 plug points visible: 2 at 11 o'clock and 2 o'clock.")

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
