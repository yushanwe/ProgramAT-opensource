"""Nearest Exit Finder

Helps blind and low-vision users locate the nearest exit in unfamiliar indoor
environments. Provides step-by-step directions using clock-face navigation
and distance cues to reach the nearest exit safely.

Features:
- Detects exit doors, exit signs, and emergency exits
- Provides clear directional guidance using clock positions (9-12, 1-3)
- Offers distance estimates and movement instructions
- Adaptive accuracy: fast by default, escalates when scene is ambiguous
- Streaming mode processes changed scenes only
- Audio-optimized output for text-to-speech

Example Usage:
User enters unfamiliar building and activates tool. The tool detects and reports:
"Exit door straight ahead, walk forward"
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "nearest_exit_finder"
TOOL_PROMPT = (
    "Assist a blind user in finding the nearest exit in an indoor environment. "
    "1. Locate visible exit doors, exit signs, or doorways that appear to lead outside. "
    "2. Identify which one is the nearest and most accessible. "
    "3. Provide clear directions using clock-face position (9-12 or 1-3 only) or "
    "body-relative terms like straight ahead, left, or right. "
    "4. Include approximate distance when the visual evidence supports it. "
    "5. Give a brief movement instruction to reach it. "
    "Do not rely on color as the only navigation cue. If no exit is visible or the "
    "evidence is insufficient, state this clearly."
)

SCENE_SIMILARITY_THRESHOLD = 0.985


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized appearance embedding for scene continuity."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(512, dtype=np.float32)
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = (
        cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        .astype(np.float32)
        .reshape(-1)
    )
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm else histogram


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = SCENE_SIMILARITY_THRESHOLD,
) -> bool:
    return cosine_similarity(reference, current) >= threshold


def needs_escalation(response_text: str) -> bool:
    """Check if the response indicates uncertainty requiring more accurate analysis."""
    if not response_text:
        return True

    lower_text = response_text.lower()

    # Uncertainty indicators
    uncertainty_phrases = [
        "no exit",
        "cannot see",
        "unclear",
        "uncertain",
        "might be",
        "possibly",
        "appears to be",
        "seems like",
        "difficult to tell",
        "not visible",
        "insufficient",
        "ambiguous",
        "hard to determine",
    ]

    return any(phrase in lower_text for phrase in uncertainty_phrases)


async def on_take_photo(runtime, image, input_data):
    """Analyze one current image for nearest exit and provide directions.

    Uses conditional cascade: fast model first, escalates to accurate model
    when the result indicates uncertainty or ambiguity.
    """
    del runtime, input_data
    if image is None:
        return "No camera image available"

    # Fast tier: try Gemini Flash Lite first
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    result_text = extract_text(response)

    # Check if we need more accurate analysis
    if needs_escalation(result_text):
        # Accurate tier: escalate to Gemini Flash for better analysis
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash",
            [{"role": "user", "content": TOOL_PROMPT}],
            [image],
            {"timeout": 60, "num_retries": 0},
        )
        result_text = extract_text(response)

    return result_text


async def on_stream_start(runtime, input_data):
    """Initialize state for streaming session."""
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)


async def on_frame(runtime, frame):
    """Process changed scenes only to provide updated exit directions.

    Uses conditional cascade: fast model first, escalates to accurate model
    when the result indicates uncertainty or ambiguity.
    """
    if runtime.is_cancelled():
        return

    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))

    # Check if this is the same scene
    if anchor is not None and is_same_scene(anchor, embedding):
        # Same scene, check if we've already analyzed this generation
        if runtime.get_state("analyzing_generation") == generation:
            return
    else:
        # New scene detected
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)

    # Check if we're already analyzing this generation
    if runtime.get_state("analyzing_generation") == generation:
        return

    runtime.set_state("analyzing_generation", generation)

    try:
        # Fast tier: analyze with Gemini Flash Lite first
        response = await asyncio.to_thread(
            call_model,
            "gemini/gemini-3.1-flash-lite-preview",
            [{"role": "user", "content": TOOL_PROMPT}],
            [frame.image],
            {"timeout": 60, "num_retries": 0},
        )

        # Check if scene changed or cancelled during analysis
        if (
            runtime.is_cancelled()
            or runtime.get_state("scene_generation") != generation
        ):
            return

        text = extract_text(response)

        # Check if we need more accurate analysis
        if needs_escalation(text):
            # Accurate tier: escalate to Gemini Flash
            response = await asyncio.to_thread(
                call_model,
                "gemini/gemini-3.1-flash",
                [{"role": "user", "content": TOOL_PROMPT}],
                [frame.image],
                {"timeout": 60, "num_retries": 0},
            )

            # Check again if scene changed or cancelled
            if (
                runtime.is_cancelled()
                or runtime.get_state("scene_generation") != generation
            ):
                return

            text = extract_text(response)

        # Emit the result
        await runtime.emit(
            text,
            final=True,
            metadata={
                "scene_generation": generation,
                "frame_id": frame.frame_id,
            },
        )
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_stream_stop(runtime):
    """Mark outstanding scene work stale before cancellation."""
    runtime.set_state(
        "scene_generation", int(runtime.get_state("scene_generation", 0)) + 1
    )
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "needs_escalation",
    "on_take_photo",
    "on_stream_start",
    "on_frame",
    "on_stream_stop",
]
