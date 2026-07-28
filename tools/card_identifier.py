"""Card Identifier tool.

Identifies a single pointed-at playing card from a photo and summarizes the full
hand when the user swipes through cards during streaming.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np

from model_router_client import copilot_llm_call


TOOL_NAME = "card_identifier"
TOOL_PROMPT = (
    "Identify the playing card or cards in view. If one card is clearly indicated, "
    "name only that card. If several cards are visible across frames, list each "
    "distinct card once. Keep the reply concise, speech-ready, and honest about uncertainty."
)

MIN_FRAME_INTERVAL_SECONDS = 0.8
MIN_CHANGED_FRACTION = 0.03
MAX_BUFFERED_FRAMES = 8
MIN_FRAMES_FOR_HAND_SUMMARY = 3

_stream_state: Dict[str, Any] = {
    "active": False,
    "frames": [],
    "last_frame": None,
    "last_analysis_at": 0.0,
    "last_spoken": "",
}


def _invalid_image(image: Any) -> bool:
    return image is None or not isinstance(image, np.ndarray) or image.size == 0


def _normalize_text(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _build_messages(mode: str) -> List[Dict[str, str]]:
    user_prompt = TOOL_PROMPT
    if mode == "single":
        user_prompt += " Prefer a single card if a fingertip or pointing gesture indicates one card."
    else:
        user_prompt += (
            " Treat these frames as a short time sequence of a hand of cards being scanned. "
            "Infer the distinct cards present in the hand and list them in a short spoken phrase."
        )

    return [
        {"role": "system", "content": "Keep responses concise and audio-friendly."},
        {"role": "user", "content": user_prompt},
    ]


def _analyze_single_card(image: np.ndarray) -> str:
    result = copilot_llm_call(
        capability="general_reasoning",
        messages=_build_messages("single"),
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "identify the pointed-at playing card in a single frame",
        },
    )
    return _normalize_text(result.get("response", ""))


def _analyze_hand_sequence(frames: List[np.ndarray]) -> str:
    result = copilot_llm_call(
        capability="temporal_reasoning",
        messages=_build_messages("sequence"),
        images=frames,
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "infer the distinct cards in hand from a short swipe sequence",
        },
    )
    return _normalize_text(result.get("response", ""))


def _frame_changed_enough(previous: Optional[np.ndarray], current: np.ndarray) -> bool:
    if previous is None:
        return True
    if previous.shape != current.shape:
        return True

    diff = np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32)))
    return float(diff) / 255.0 >= MIN_CHANGED_FRACTION


def _append_frame(image: np.ndarray) -> bool:
    now = time.time()
    if now - _stream_state["last_analysis_at"] < MIN_FRAME_INTERVAL_SECONDS:
        return False
    if not _frame_changed_enough(_stream_state["last_frame"], image):
        return False

    _stream_state["frames"].append(image)
    if len(_stream_state["frames"]) > MAX_BUFFERED_FRAMES:
        _stream_state["frames"] = _stream_state["frames"][-MAX_BUFFERED_FRAMES:]
    _stream_state["last_frame"] = image
    _stream_state["last_analysis_at"] = now
    return True


def _reset_stream_state() -> None:
    _stream_state["active"] = False
    _stream_state["frames"] = []
    _stream_state["last_frame"] = None
    _stream_state["last_analysis_at"] = 0.0
    _stream_state["last_spoken"] = ""


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> str:
    if _invalid_image(image):
        return "No camera image available."

    try:
        response = _analyze_single_card(image)
    except Exception:
        return "I could not identify the card."

    return response or "I could not identify the card."


def on_take_photo(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> str:
    return main(image, input_data)


def on_stream_start(*args: Any, **kwargs: Any) -> str:
    _reset_stream_state()
    _stream_state["active"] = True
    return ""


def on_frame(*args: Any, **kwargs: Any) -> str:
    image = None
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, np.ndarray):
            image = value
            break

    if _invalid_image(image):
        return ""

    if not _append_frame(image):
        return ""

    if len(_stream_state["frames"]) < MIN_FRAMES_FOR_HAND_SUMMARY:
        return ""

    try:
        response = _analyze_hand_sequence(_stream_state["frames"])
    except Exception:
        return ""

    if not response or response == _stream_state["last_spoken"]:
        return ""

    _stream_state["last_spoken"] = response
    return response


def on_stream_stop(*args: Any, **kwargs: Any) -> str:
    frames = list(_stream_state["frames"])
    last_spoken = _stream_state["last_spoken"]
    _reset_stream_state()

    if len(frames) < MIN_FRAMES_FOR_HAND_SUMMARY:
        return ""

    try:
        response = _analyze_hand_sequence(frames)
    except Exception:
        return ""

    if not response or response == last_spoken:
        return ""

    return response

