"""Detect hand gestures from a single image or recent motion history."""

from __future__ import annotations

import asyncio
from typing import Iterable, List

from litellm_utils import call_model, extract_text

TOOL_NAME = "hand_gesture_detector"
TOOL_PROMPT = (
    "For a blind or low-vision user, identify the visible hand gesture. When you "
    "receive multiple chronological frames, use the full sequence to recognize "
    "motion gestures such as waving hello or moving one finger side to side for "
    "no/incorrect. When you receive one image, only report a clearly visible "
    "static hand posture and say that motion cannot be confirmed from one image. "
    "Reply with one concise spoken sentence. If no clear gesture is visible, say "
    "\"No clear hand gesture detected.\""
)


def _sort_frames_chronologically(frames: Iterable) -> List:
    return sorted(
        list(frames),
        key=lambda frame: (
            getattr(frame, "timestamp", None) is None,
            getattr(frame, "timestamp", None),
            getattr(frame, "frame_id", ""),
        ),
    )


async def _analyze_images(images):
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        images,
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or "No clear hand gesture detected."


async def on_take_photo(runtime, image, input_data):
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    return await _analyze_images([image])


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("is_analyzing", False)
    runtime.set_state("last_processed_frame_id", None)
    runtime.set_state("last_result_text", None)


async def on_frame(runtime, frame):
    if runtime.is_cancelled() or runtime.get_state("is_analyzing"):
        return
    if runtime.get_state("last_processed_frame_id") == frame.frame_id:
        return

    recent_frames = runtime.get_recent_frames(seconds=6) or []
    if not recent_frames:
        recent_frames = [frame]
    ordered_frames = _sort_frames_chronologically(recent_frames)
    images = [candidate.image for candidate in ordered_frames if candidate.image is not None]
    if not images:
        return

    runtime.set_state("is_analyzing", True)
    try:
        text = await _analyze_images(images)
        if runtime.is_cancelled():
            return
        if text != runtime.get_state("last_result_text"):
            await runtime.emit(
                text,
                final=True,
                metadata={
                    "frame_id": frame.frame_id,
                    "frame_count": len(images),
                },
            )
            runtime.set_state("last_result_text", text)
        runtime.set_state("last_processed_frame_id", frame.frame_id)
    finally:
        runtime.set_state("is_analyzing", False)


async def on_stream_stop(runtime):
    runtime.clear_state()

