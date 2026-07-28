"""Identify a pointed-at card or, after a swipe, announce the full visible hand."""

from __future__ import annotations

import asyncio
import re
from typing import Iterable

from litellm_utils import call_model, extract_text

TOOL_NAME = "card_identifier"
TOOL_PROMPT = (
    "Help a blind user identify playing cards in hand. "
    "1. Read only cards whose rank and suit are visually supported. "
    "2. If one current image shows a finger clearly pointing at one specific card, "
    "answer with only that card name, like 'Nine of hearts.' "
    "3. If multiple chronological images show the finger sweeping across the hand so "
    "the full hand becomes readable, answer with only the full card list in reading "
    "order, like 'Nine of hearts, two of spades, jack of diamonds.' "
    "4. If the evidence is insufficient, answer with one short uncertainty sentence. "
    "Do not infer a swipe from one image. Use only standard ranks and suits."
)

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
FRAME_ANALYSIS_INTERVAL_SECONDS = 0.9
SWIPE_WINDOW_SECONDS = 3.5
SWIPE_MIN_FRAMES = 4
MAX_SWIPE_FRAMES = 8
SWIPE_MIN_DISTINCT_POINTED_CARDS = 2
SWIPE_COOLDOWN_SECONDS = 2.0

CARD_LINE_RE = re.compile(
    r"^(?:"
    r"(?:ace|two|three|four|five|six|seven|eight|nine|ten|jack|queen|king)"
    r" of "
    r"(?:clubs|diamonds|hearts|spades)"
    r"(?:, )?"
    r")+\.?$",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return "I can't read the cards clearly enough."
    if normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def _looks_like_card_answer(text: str) -> bool:
    cleaned = _normalize_text(text).rstrip(".")
    return bool(CARD_LINE_RE.fullmatch(cleaned))


def _is_uncertain(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        phrase in lowered
        for phrase in (
            "can't",
            "cannot",
            "unclear",
            "uncertain",
            "insufficient",
            "not enough",
            "not clear",
            "unable",
            "no card",
            "no finger",
            "no hand",
        )
    )


async def _run_model(images: Iterable[object]) -> str:
    response = await asyncio.to_thread(
        call_model,
        MODEL_NAME,
        [{"role": "user", "content": TOOL_PROMPT}],
        list(images),
        {"timeout": 60, "num_retries": 0},
    )
    return _normalize_text(extract_text(response))


def _trim_point_history(history: list[dict], now: float) -> list[dict]:
    cutoff = now - SWIPE_WINDOW_SECONDS
    return [item for item in history if float(item.get("timestamp", 0.0)) >= cutoff]


def _update_point_history(history: list[dict], card_name: str, timestamp: float) -> list[dict]:
    trimmed = _trim_point_history(history, timestamp)
    normalized_card = _normalize_text(card_name)
    if trimmed and trimmed[-1]["card"] == normalized_card:
        trimmed[-1]["timestamp"] = timestamp
        return trimmed
    trimmed.append({"card": normalized_card, "timestamp": timestamp})
    return trimmed


def _has_swipe_evidence(history: list[dict]) -> bool:
    distinct_cards: list[str] = []
    for item in history:
        card_name = item["card"]
        if not distinct_cards or distinct_cards[-1] != card_name:
            distinct_cards.append(card_name)
    return len(distinct_cards) >= SWIPE_MIN_DISTINCT_POINTED_CARDS


async def on_take_photo(runtime, image, input_data):
    del runtime, input_data
    if image is None:
        return "No camera image is available."
    result = await _run_model([image])
    return result


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("last_point_check_timestamp", None)
    runtime.set_state("last_point_result", None)
    runtime.set_state("last_swipe_result", None)
    runtime.set_state("point_history", [])
    runtime.set_state("last_swipe_emit_timestamp", None)


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return

    last_point_check = runtime.get_state("last_point_check_timestamp")
    if (
        last_point_check is not None
        and frame.timestamp - float(last_point_check) < FRAME_ANALYSIS_INTERVAL_SECONDS
    ):
        return

    runtime.set_state("last_point_check_timestamp", frame.timestamp)
    pointed = await _run_model([frame.image])
    if runtime.is_cancelled():
        return

    last_point_result = runtime.get_state("last_point_result")
    if _looks_like_card_answer(pointed) and not _is_uncertain(pointed):
        if pointed != last_point_result:
            runtime.set_state("last_point_result", pointed)
            await runtime.emit(pointed, final=True)
        point_history = runtime.get_state("point_history", [])
        point_history = _update_point_history(point_history, pointed, float(frame.timestamp))
        runtime.set_state("point_history", point_history)
    else:
        point_history = _trim_point_history(
            runtime.get_state("point_history", []),
            float(frame.timestamp),
        )
        runtime.set_state("point_history", point_history)

    recent_frames = runtime.get_recent_frames(seconds=SWIPE_WINDOW_SECONDS)
    if len(recent_frames) < SWIPE_MIN_FRAMES:
        return
    if not _has_swipe_evidence(runtime.get_state("point_history", [])):
        return

    last_swipe_emit_timestamp = runtime.get_state("last_swipe_emit_timestamp")
    if (
        last_swipe_emit_timestamp is not None
        and float(frame.timestamp) - float(last_swipe_emit_timestamp) < SWIPE_COOLDOWN_SECONDS
    ):
        return

    selected_frames = recent_frames[-MAX_SWIPE_FRAMES:]
    swipe_result = await _run_model([item.image for item in selected_frames])
    if runtime.is_cancelled():
        return

    last_swipe_result = runtime.get_state("last_swipe_result")
    if (
        _looks_like_card_answer(swipe_result)
        and not _is_uncertain(swipe_result)
        and "," in swipe_result
        and swipe_result != last_swipe_result
    ):
        runtime.set_state("last_swipe_result", swipe_result)
        runtime.set_state("last_swipe_emit_timestamp", float(frame.timestamp))
        await runtime.emit(swipe_result, final=True)


async def on_stream_stop(runtime):
    if runtime.is_cancelled():
        return

    recent_frames = runtime.get_recent_frames(seconds=SWIPE_WINDOW_SECONDS)
    if len(recent_frames) < SWIPE_MIN_FRAMES:
        return
    point_history = _trim_point_history(
        runtime.get_state("point_history", []),
        float(recent_frames[-1].timestamp),
    )
    if not _has_swipe_evidence(point_history):
        return

    swipe_result = await _run_model([frame.image for frame in recent_frames[-MAX_SWIPE_FRAMES:]])
    if (
        _looks_like_card_answer(swipe_result)
        and not _is_uncertain(swipe_result)
        and "," in swipe_result
        and swipe_result != runtime.get_state("last_swipe_result")
    ):
        await runtime.emit(swipe_result, final=True)


__all__ = [
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
