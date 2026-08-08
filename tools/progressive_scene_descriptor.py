"""Progressive Scene Descriptor for Blind Users

Provides a multi-stage description of the scene, starting with a brief list
of objects and transitioning to a more detailed scene summary if the camera
remains fixed on the same view.

Features:
- Two-stage progressive output: brief → detailed
- Scene stability detection to determine when to progress
- Automatic restart on scene change
- Audio-optimized output for text-to-speech

Example Output:
Stage 1 (brief): "A desk, a cabinet, a printer, and a cat."
Stage 2 (detailed): "A cat sitting on top of a white desk, looking out the window."
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from litellm_utils import call_model, extract_text

TOOL_NAME = "progressive_scene_descriptor"
TOOL_PROMPT = (
    "Describe the scene progressively for a blind or low-vision user. "
    "Start with a brief list of main objects, then provide more detail "
    "if the scene remains stable."
)
TOOL_RUNTIME_INPUT = {
    "key": "search_target",
    "label": "What are you looking for?",
    "placeholder": "e.g., my keys, a red mug, the exit sign",
    "prompt_instruction": (
        "The user is specifically looking for: {value}. "
        "Focus on this target in your description and mention if you see it."
    ),
}

# Scene stability configuration
SCENE_SIMILARITY_THRESHOLD = 0.98
STABILITY_FRAME_COUNT = 3  # Number of consecutive stable frames before detailed stage

# Prompts for each stage
BRIEF_PROMPT = (
    "List the main objects visible in this scene in a brief, natural way. "
    "Present them as a simple spoken list. Return one concise sentence. "
    "Example format: 'A desk, a cabinet, a printer, and a cat.' "
    "Return one concise user-facing response suitable for spoken output."
)

DETAILED_PROMPT = (
    "Provide a detailed description of this scene for a blind or low-vision user. "
    "Describe the key objects, their positions, relationships, and any notable details. "
    "Focus on what's most interesting or important. Prefer one or two short sentences "
    "in a single paragraph with no unnecessary line breaks. "
    "Return one concise user-facing response suitable for spoken output."
)

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite"
FALLBACK_TEXT = "I cannot clearly see the scene."


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a compact normalized appearance embedding for scene continuity.

    Args:
        image: OpenCV image (numpy array in BGR format)

    Returns:
        Normalized HSV histogram embedding
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(512, dtype=np.float32)
    resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]
    ).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm else histogram


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = SCENE_SIMILARITY_THRESHOLD,
) -> bool:
    """Check if two scene embeddings represent the same scene."""
    return cosine_similarity(reference, current) >= threshold


async def analyze_brief(image: np.ndarray, search_target: str | None = None) -> str:
    """Generate brief object list for the scene.

    Args:
        image: OpenCV image to analyze
        search_target: Optional specific object the user is looking for

    Returns:
        Brief description text or fallback
    """
    prompt = BRIEF_PROMPT
    if search_target:
        prompt = (
            f"{BRIEF_PROMPT} "
            f"The user is specifically looking for: {search_target}. "
            f"Focus on this target in your description and mention if you see it."
        )

    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def analyze_detailed(image: np.ndarray, search_target: str | None = None) -> str:
    """Generate detailed scene description.

    Args:
        image: OpenCV image to analyze
        search_target: Optional specific object the user is looking for

    Returns:
        Detailed description text or fallback
    """
    prompt = DETAILED_PROMPT
    if search_target:
        prompt = (
            f"{DETAILED_PROMPT} "
            f"The user is specifically looking for: {search_target}. "
            f"Focus on this target in your description and mention if you see it."
        )

    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": prompt}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):
    """Use brief description for one-shot Take Photo requests."""
    del runtime
    if image is None:
        return FALLBACK_TEXT

    search_target = None
    if isinstance(input_data, dict):
        search_target = input_data.get(TOOL_RUNTIME_INPUT["key"])

    return await analyze_brief(image, search_target)


async def on_stream_start(runtime, input_data):
    """Initialize state for progressive streaming."""
    search_target = None
    if isinstance(input_data, dict):
        search_target = input_data.get(TOOL_RUNTIME_INPUT["key"])

    runtime.set_state("search_target", search_target)
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("brief_emitted", False)
    runtime.set_state("detailed_emitted", False)
    runtime.set_state("stable_frame_count", 0)
    runtime.set_state("in_flight", False)
    runtime.set_state("embedding_in_flight", False)


async def on_frame(runtime, frame):
    """Analyze scene and emit progressive descriptions based on stability.

    Logic:
    1. Compute scene embedding
    2. Check if scene is same as anchor
    3. If new scene: emit brief, reset detailed
    4. If same scene and stable: emit detailed after threshold
    """
    if runtime.is_cancelled():
        return

    # Prevent overlapping embedding computation
    if runtime.get_state("embedding_in_flight"):
        return
    runtime.set_state("embedding_in_flight", True)
    try:
        embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    finally:
        runtime.set_state("embedding_in_flight", False)

    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))
    brief_emitted = runtime.get_state("brief_emitted", False)
    detailed_emitted = runtime.get_state("detailed_emitted", False)
    stable_count = runtime.get_state("stable_frame_count", 0)

    # Check scene stability
    if anchor is not None and is_same_scene(anchor, embedding):
        # Same scene - increment stability counter
        stable_count += 1
        runtime.set_state("stable_frame_count", stable_count)
    else:
        # New scene detected - reset everything
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("brief_emitted", False)
        runtime.set_state("detailed_emitted", False)
        runtime.set_state("stable_frame_count", 0)
        brief_emitted = False
        detailed_emitted = False
        stable_count = 0

    # Prevent overlapping analysis
    if runtime.get_state("in_flight"):
        return

    # Get search target from runtime state
    search_target = runtime.get_state("search_target")

    # Stage 1: Emit brief description for new scene
    if not brief_emitted:
        runtime.set_state("in_flight", True)
        try:
            text = await analyze_brief(frame.image, search_target)
            if not runtime.is_cancelled() and runtime.get_state("scene_generation") == generation:
                await runtime.emit(text, partial=False, final=False)
                runtime.set_state("brief_emitted", True)
        finally:
            runtime.set_state("in_flight", False)
        return

    # Stage 2: Emit detailed description if scene is stable
    if not detailed_emitted and stable_count >= STABILITY_FRAME_COUNT:
        runtime.set_state("in_flight", True)
        try:
            text = await analyze_detailed(frame.image, search_target)
            if not runtime.is_cancelled() and runtime.get_state("scene_generation") == generation:
                await runtime.emit(text, partial=False, final=True)
                runtime.set_state("detailed_emitted", True)
        finally:
            runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):
    """Clean up state on stream stop."""
    runtime.clear_state()


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
