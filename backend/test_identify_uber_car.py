import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import identify_uber_car


class TestIdentifyUberCar(unittest.TestCase):
    def test_main_calls_take_photo_baseline_helper(self):
        image = object()
        with patch.object(
            identify_uber_car,
            "call_take_photo_baseline_vlm",
            return_value="Blue Toyota Prius, plate ABC123.",
        ) as helper:
            result = identify_uber_car.main(image, {})

        self.assertEqual(result, "Blue Toyota Prius, plate ABC123.")
        helper.assert_called_once_with(
            image=image,
            prompt=identify_uber_car.TOOL_PROMPT,
            tool_name=identify_uber_car.TOOL_NAME,
        )

    def test_main_handles_missing_image(self):
        self.assertEqual(
            identify_uber_car.main(None, {}),
            "No camera image available.",
        )

    def test_main_handles_helper_failures(self):
        with patch.object(
            identify_uber_car,
            "call_take_photo_baseline_vlm",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(
                identify_uber_car.main(object(), {}),
                "I couldn't identify the car from that photo.",
            )


if __name__ == "__main__":
    unittest.main()
