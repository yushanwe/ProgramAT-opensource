"""Identify a recently completed hand gesture from one image or recent frames."""

from __future__ import annotations

import asyncio
from typing import Iterable, List

from litellm_utils import call_model, extract_text

TOOL_NAME = "identify_recent_hand_gesture"
TOOL_PROMPT = (
    "Identify the person's recent hand gesture for a blind or low-vision user. "
    "When several chronological frames are provided, inspect them earliest to latest "
    "and use the motion sequence to identify the completed gesture or short meaning. "
    "When only one image is provided, identify only a clearly recognizable static hand "
    "pose and say that motion cannot be determined from one image when needed. "
    "Return one concise spoken sentence. If the evidence is insufficient, say so plainly."
)

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
RECENT_WINDOW_SECONDS = 4
MIN_STREAM_FRAMES = 4
MAX_STREAM_FRAMES = 8
MIN_ANALYSIS_INTERVAL_SECONDS = 1.0


def _dedupe_recent_frames(frames: Iterable) -> List:
    deduped = []
    seen_ids = set()
    for frame in frames:
        frame_id = getattr(frame, "frame_id", None)
        if frame_id in seen_ids:
            continue
        seen_ids.add(frame_id)
        deduped.append(frame)
    return deduped


def _select_chronological_frames(frames: List, limit: int) -> List:
    if len(frames) <= limit:
        return frames
    indexes = {
        round(i * (len(frames) - 1) / (limit - 1))
        for i in range(limit)
    }
    return [frame for idx, frame in enumerate(frames) if idx in indexes]


async def _analyze_images(images) -> str:
    response = await asyncio.to_thread(
        call_model,
        MODEL_NAME,
        [{"role": "user", "content": TOOL_PROMPT}],
        images,
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or "I can't tell what hand gesture was made from the available view."


async def on_take_photo(runtime, image, input_data):
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    return await _analyze_images([image])


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("analysis_generation", 0)
    runtime.set_state("analysis_in_flight", False)
    runtime.set_state("last_analysis_time", None)
    runtime.set_state("last_latest_frame_id", None)
    runtime.set_state("last_result", None)


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return

    last_time = runtime.get_state("last_analysis_time")
    if (
        last_time is not None
        and frame.timestamp - last_time < MIN_ANALYSIS_INTERVAL_SECONDS
    ):
        return

    if runtime.get_state("analysis_in_flight"):
        return

    recent_frames = runtime.get_recent_frames(seconds=RECENT_WINDOW_SECONDS) or []
    recent_frames = _dedupe_recent_frames(recent_frames)
    if len(recent_frames) < MIN_STREAM_FRAMES:
        return

    selected_frames = _select_chronological_frames(recent_frames, MAX_STREAM_FRAMES)
    latest_frame_id = getattr(selected_frames[-1], "frame_id", None)
    if latest_frame_id == runtime.get_state("last_latest_frame_id"):
        return

    generation = int(runtime.get_state("analysis_generation", 0)) + 1
    runtime.set_state("analysis_generation", generation)
    runtime.set_state("analysis_in_flight", True)
    runtime.set_state("last_analysis_time", frame.timestamp)
    runtime.set_state("last_latest_frame_id", latest_frame_id)

    try:
        result = await _analyze_images([item.image for item in selected_frames])
        if (
            runtime.is_cancelled()
            or runtime.get_state("analysis_generation") != generation
        ):
            return
        if result != runtime.get_state("last_result"):
            runtime.set_state("last_result", result)
            await runtime.emit(
                result,
                final=True,
                replace=True,
                metadata={
                    "frame_id": latest_frame_id,
                    "frame_count": len(selected_frames),
                },
            )
    finally:
        if runtime.get_state("analysis_generation") == generation:
            runtime.set_state("analysis_in_flight", False)


async def on_stream_stop(runtime):
    runtime.set_state(
        "analysis_generation", int(runtime.get_state("analysis_generation", 0)) + 1
    )
    runtime.set_state("analysis_in_flight", False)


__all__ = [
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
