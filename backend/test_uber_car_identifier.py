"""Focused tests for uber_car_identifier runtime input handling."""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "tools"))

from generated_tool_runtime import (  # noqa: E402
    FrameStore,
    ToolFrame,
    ToolRuntime,
)
import uber_car_identifier  # noqa: E402


class TestUberCarIdentifier(unittest.IsolatedAsyncioTestCase):
    def runtime(self, emitted, *, session="session"):
        async def emit(event):
            emitted.append(event)
            return True

        return ToolRuntime(
            client_id="client",
            tool_name="uber_car_identifier",
            tool_version="version",
            session_id=session,
            request_id=f"request-{session}",
            frame_store=FrameStore(max_frames=4),
            emit_callback=emit,
        )

    @patch("uber_car_identifier.call_model")
    async def test_take_photo_with_search_criteria(self, mock_call_model):
        """Test on_take_photo reads search_criteria from input_data."""
        mock_call_model.return_value = type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": {"content": "Black Honda Civic, plate ABC123. This matches your search."}})]},
        )()

        emitted = []
        runtime = self.runtime(emitted)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        input_data = {"search_criteria": "black Honda"}

        result = await uber_car_identifier.on_take_photo(runtime, test_image, input_data)

        # Verify call_model was called
        self.assertTrue(mock_call_model.called)

        # Verify the prompt includes the search criteria
        call_args = mock_call_model.call_args
        prompt = call_args[0][1][0]["content"]
        self.assertIn("black Honda", prompt)
        self.assertIn("compare it to these expected characteristics", prompt)

        # Verify result is not empty
        self.assertTrue(result)
        self.assertNotEqual(result, "")

    @patch("uber_car_identifier.call_model")
    async def test_take_photo_without_search_criteria(self, mock_call_model):
        """Test on_take_photo works without search_criteria."""
        mock_call_model.return_value = type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": {"content": "White Toyota Camry, plate XYZ789."}})]},
        )()

        emitted = []
        runtime = self.runtime(emitted)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        input_data = {}

        result = await uber_car_identifier.on_take_photo(runtime, test_image, input_data)

        # Verify call_model was called
        self.assertTrue(mock_call_model.called)

        # Verify the prompt does NOT include comparison instruction
        call_args = mock_call_model.call_args
        prompt = call_args[0][1][0]["content"]
        self.assertNotIn("compare it to these expected characteristics", prompt)

        # Verify result is not empty
        self.assertTrue(result)

    async def test_stream_start_stores_search_criteria(self):
        """Test on_stream_start extracts and stores search_criteria in runtime state."""
        emitted = []
        runtime = self.runtime(emitted)
        input_data = {"search_criteria": "white Toyota Camry"}

        await uber_car_identifier.on_stream_start(runtime, input_data)

        # Verify search_criteria is stored in runtime state
        self.assertEqual(runtime.get_state("search_criteria"), "white Toyota Camry")
        self.assertEqual(runtime.get_state("in_flight"), False)

    async def test_stream_start_without_search_criteria(self):
        """Test on_stream_start handles missing search_criteria gracefully."""
        emitted = []
        runtime = self.runtime(emitted)
        input_data = {}

        await uber_car_identifier.on_stream_start(runtime, input_data)

        # Verify search_criteria is None when not provided
        self.assertIsNone(runtime.get_state("search_criteria"))
        self.assertEqual(runtime.get_state("in_flight"), False)

    @patch("uber_car_identifier.call_model")
    async def test_on_frame_uses_stored_search_criteria(self, mock_call_model):
        """Test on_frame retrieves search_criteria from runtime state."""
        mock_call_model.return_value = type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": {"content": "Red Tesla Model 3. Does not match your search."}})]},
        )()

        emitted = []
        runtime = self.runtime(emitted)

        # Set up runtime state as if on_stream_start was called
        runtime.set_state("search_criteria", "blue Honda Accord")
        runtime.set_state("in_flight", False)

        test_frame = ToolFrame(
            frame_id=1,
            timestamp=1.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            width=100,
            height=100,
        )

        await uber_car_identifier.on_frame(runtime, test_frame)

        # Verify call_model was called
        self.assertTrue(mock_call_model.called)

        # Verify the prompt includes the stored search criteria
        call_args = mock_call_model.call_args
        prompt = call_args[0][1][0]["content"]
        self.assertIn("blue Honda Accord", prompt)
        self.assertIn("compare it to these expected characteristics", prompt)

        # Verify an emit was called
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0]["text"])

    @patch("uber_car_identifier.call_model")
    async def test_on_frame_without_search_criteria(self, mock_call_model):
        """Test on_frame works when no search_criteria is stored."""
        mock_call_model.return_value = type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": {"content": "Silver Ford F-150, plate DEF456."}})]},
        )()

        emitted = []
        runtime = self.runtime(emitted)

        # Set up runtime state without search_criteria
        runtime.set_state("search_criteria", None)
        runtime.set_state("in_flight", False)

        test_frame = ToolFrame(
            frame_id=1,
            timestamp=1.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            width=100,
            height=100,
        )

        await uber_car_identifier.on_frame(runtime, test_frame)

        # Verify call_model was called
        self.assertTrue(mock_call_model.called)

        # Verify the prompt does NOT include comparison instruction
        call_args = mock_call_model.call_args
        prompt = call_args[0][1][0]["content"]
        self.assertNotIn("compare it to these expected characteristics", prompt)

        # Verify an emit was called
        self.assertEqual(len(emitted), 1)


if __name__ == "__main__":
    unittest.main()
