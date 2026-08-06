"""Focused tests for the exit_finder generated tool."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "tools"))

from generated_tool_runtime import FrameStore, ToolFrame, ToolRuntime  # noqa: E402

import exit_finder  # noqa: E402


class TestExitFinderTool(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _to_thread_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    def _runtime(self, emitted):
        async def emit(event):
            emitted.append(event)
            return True

        return ToolRuntime(
            client_id="client",
            tool_name="exit_finder",
            tool_version="v1",
            session_id="session",
            frame_store=FrameStore(),
            emit_callback=emit,
        )

    async def test_take_photo_returns_model_text(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Turn left at the hallway and walk to the marked exit."
                    )
                )
            ]
        )
        image = object()
        with patch.object(
            exit_finder, "call_model", return_value=response
        ) as call, patch.object(
            exit_finder.asyncio, "to_thread", side_effect=self._to_thread_inline
        ):
            result = await exit_finder.on_take_photo(None, image, {})

        self.assertEqual(
            result, "Turn left at the hallway and walk to the marked exit."
        )
        self.assertEqual(call.call_args.args[0], "gemini/gemini-3.1-flash-lite")

    async def test_take_photo_with_no_image_returns_fallback(self):
        result = await exit_finder.on_take_photo(None, None, {})
        self.assertEqual(result, exit_finder.FALLBACK_TEXT)

    async def test_on_frame_emits_final_result(self):
        emitted = []
        runtime = self._runtime(emitted)
        await exit_finder.on_stream_start(runtime, {})

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Exit at 12 o'clock, walk forward.")
                )
            ]
        )
        image = object()
        frame = ToolFrame(1, 1.0, image, 16, 16)

        with patch.object(
            exit_finder, "call_model", return_value=response
        ), patch.object(
            exit_finder.asyncio, "to_thread", side_effect=self._to_thread_inline
        ):
            await exit_finder.on_frame(runtime, frame)

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["text"], "Exit at 12 o'clock, walk forward.")
        self.assertTrue(emitted[0]["final"])
        self.assertFalse(runtime.get_state("in_flight"))

    async def test_on_frame_skips_while_in_flight(self):
        emitted = []
        runtime = self._runtime(emitted)
        await exit_finder.on_stream_start(runtime, {})
        runtime.set_state("in_flight", True)

        image = object()
        frame = ToolFrame(1, 1.0, image, 16, 16)

        with patch.object(exit_finder, "call_model") as call:
            await exit_finder.on_frame(runtime, frame)

        call.assert_not_called()
        self.assertEqual(emitted, [])


if __name__ == "__main__":
    unittest.main()
