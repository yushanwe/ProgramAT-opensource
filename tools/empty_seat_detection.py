"""Find available seating with progressive multi-model output and CLIP scene hold detection."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
from PIL import Image

from litellm_utils import call_model, call_openai_responses_model, extract_text

TOOL_NAME = "empty_seat_detection"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "You are guiding a blind or low-vision user. Count clearly empty chairs or seats, then report the closest empty seat "
    "relative to the user using clock-face direction and rough distance in feet only. Keep it concise, include empty-seat "
    "count and closest-seat position. If no empty seat is visible, say exactly \"0 empty chairs. No empty seat is visible.\""
)
PRECISE_TOOL_PROMPT = (
    "You are guiding a blind or low-vision user who is holding the camera on the same scene for a more precise answer. "
    "Carefully verify which seats are truly unoccupied, then return one concise line with: empty-seat count, and closest "
    "empty seat clock-face direction plus rough distance in feet. If uncertain, state uncertainty briefly. If no empty seat "
    "is visible, say exactly \"0 empty chairs. No empty seat is visible.\""
)

SCENE_SIMILARITY_THRESHOLD = 0.975
FRAME_TO_FRAME_SIMILARITY_THRESHOLD = 0.955
# Require consecutive changed frames before restarting model calls for a new scene.
SCENE_CHANGE_CONFIRMATIONS = 2
SCENE_HOLD_SECONDS = 2.5
EMBEDDING_FRAME_STRIDE = 2
CLIP_EMBEDDING_DIM = 512
HISTOGRAM_FALLBACK_SIZE = (96, 96)
HISTOGRAM_FALLBACK_BINS = [32, 16]
HISTOGRAM_FALLBACK_RANGES = [0, 180, 0, 256]
ANCHOR_BLEND_REFERENCE_WEIGHT = 0.85
ANCHOR_BLEND_CURRENT_WEIGHT = 0.15
NORMALIZATION_EPSILON = 1e-10

PROGRESSIVE_MODELS = (
    ("moondream/moondream3-preview", "litellm"),
    ("gemini/gemini-3.1-flash-lite-preview", "litellm"),
    ("gpt-5", "openai_responses"),
)
PRECISE_MODELS = (("gpt-5", "openai_responses"),)

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_clip_model = None
_clip_processor = None


def _load_clip_encoder():
    global _clip_model, _clip_processor
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor
    from transformers import CLIPModel, CLIPProcessor

    _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
    _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
    _clip_model.eval()
    return _clip_model, _clip_processor


def compute_scene_embedding(image: np.ndarray) -> np.ndarray:
    """Create a normalized CLIP embedding for scene continuity checks."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return np.zeros(CLIP_EMBEDDING_DIM, dtype=np.float32)

    try:
        import torch

        model, processor = _load_clip_encoder()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.inference_mode():
            output = model.get_image_features(**inputs)
        vector = output.detach().cpu().numpy().astype(np.float32).reshape(-1)
    except Exception:
        resized = cv2.resize(image, HISTOGRAM_FALLBACK_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        vector = cv2.calcHist(
            [hsv], [0, 1], None, HISTOGRAM_FALLBACK_BINS, HISTOGRAM_FALLBACK_RANGES
        ).astype(np.float32).reshape(-1)

    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return np.zeros_like(vector)
    return vector / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity or -1.0 sentinel when vectors are incompatible."""
    if first is None or second is None or first.shape != second.shape:
        return -1.0
    return float(np.dot(first, second))


def is_same_scene(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = SCENE_SIMILARITY_THRESHOLD,
) -> bool:
    return cosine_similarity(reference, current) >= threshold


def _blend_scene_anchor(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    blended = (
        ANCHOR_BLEND_REFERENCE_WEIGHT * reference.astype(np.float32)
    ) + (ANCHOR_BLEND_CURRENT_WEIGHT * current.astype(np.float32))
    norm = float(np.linalg.norm(blended))
    if norm < NORMALIZATION_EPSILON:
        current_norm = float(np.linalg.norm(current))
        if current_norm < NORMALIZATION_EPSILON:
            return current
        return current / current_norm
    return blended / norm


def _call_selected_model(
    model_name: str, provider: str, image: np.ndarray, prompt: str
) -> str:
    if provider == "openai_responses":
        return call_openai_responses_model(
            model_name,
            prompt,
            image,
            reasoning_effort="medium",
            timeout=60,
        )
    response = call_model(
        model_name,
        [{"role": "user", "content": prompt}],
        images=[image],
        metadata={"timeout": 60, "num_retries": 0},
    )
    return extract_text(response)


async def _emit_progressive_results(
    runtime,
    image: np.ndarray,
    prompt: str,
    models,
    generation: int | None,
    frame_id: int | None,
    phase: str,
):
    async def run_model(model_name: str, provider: str):
        try:
            text = await asyncio.to_thread(
                _call_selected_model, model_name, provider, image, prompt
            )
            return model_name, text, None
        except Exception as exc:
            return model_name, "", str(exc)

    tasks = [
        asyncio.create_task(run_model(model_name, provider))
        for model_name, provider in models
    ]
    first_text = None
    try:
        for completed in asyncio.as_completed(tasks):
            if runtime is not None and (
                runtime.is_cancelled()
                or (
                    generation is not None
                    and runtime.get_state("scene_generation") != generation
                )
            ):
                break
            model_name, text, _error = await completed
            if text and first_text is None:
                first_text = text
            if runtime is None:
                continue
            if runtime.is_cancelled() or (
                generation is not None
                and runtime.get_state("scene_generation") != generation
            ):
                continue
            if text:
                await runtime.emit(
                    text,
                    partial=True,
                    metadata={
                        "model": model_name,
                        "phase": phase,
                        "scene_generation": generation,
                        "frame_id": frame_id,
                    },
                )
        if runtime is not None and not runtime.is_cancelled() and (
            generation is None or runtime.get_state("scene_generation") == generation
        ):
            await runtime.emit(
                "",
                final=True,
                metadata={
                    "phase": phase,
                    "scene_generation": generation,
                    "frame_id": frame_id,
                },
            )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    return first_text or "No seating result available."


async def on_take_photo(runtime, image, input_data):
    del input_data
    if image is None:
        return "No camera image is available."
    result = await _emit_progressive_results(
        runtime,
        image,
        TOOL_PROMPT,
        PROGRESSIVE_MODELS,
        generation=None,
        frame_id=1,
        phase="take_photo",
    )
    if runtime is None:
        return result
    # In executable runtime mode, outputs are delivered via runtime.emit events.
    return None


async def on_stream_start(runtime, input_data):
    del input_data
    runtime.set_state("scene_anchor", None)
    runtime.set_state("scene_generation", 0)
    runtime.set_state("analyzing_generation", None)
    runtime.set_state("completed_generation", None)
    runtime.set_state("precise_generation", None)
    runtime.set_state("scene_started_at", None)
    runtime.set_state("last_embedding", None)
    runtime.set_state("last_embedding_frame_id", 0)
    runtime.set_state("scene_change_streak", 0)


async def _run_stream_phase(
    runtime,
    frame,
    generation: int,
    prompt: str,
    models,
    phase: str,
    completion_key: str | None,
):
    try:
        await _emit_progressive_results(
            runtime,
            frame.image,
            prompt,
            models,
            generation=generation,
            frame_id=frame.frame_id,
            phase=phase,
        )
        if (
            completion_key
            and runtime.get_state("scene_generation") == generation
        ):
            runtime.set_state(completion_key, generation)
    finally:
        if runtime.get_state("analyzing_generation") == generation:
            runtime.set_state("analyzing_generation", None)


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return

    previous_embedding = runtime.get_state("last_embedding")
    cached_frame_id = int(runtime.get_state("last_embedding_frame_id", 0))
    if (
        previous_embedding is not None
        and frame.frame_id - cached_frame_id < EMBEDDING_FRAME_STRIDE
    ):
        embedding = previous_embedding
    else:
        embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
        runtime.set_state("last_embedding", embedding)
        runtime.set_state("last_embedding_frame_id", frame.frame_id)
    anchor = runtime.get_state("scene_anchor")
    generation = int(runtime.get_state("scene_generation", 0))
    if anchor is None:
        same_scene = False
    else:
        anchor_similarity = cosine_similarity(anchor, embedding)
        frame_to_frame_similarity = (
            cosine_similarity(previous_embedding, embedding)
            if previous_embedding is not None
            else -1.0
        )
        same_scene = (
            anchor_similarity >= SCENE_SIMILARITY_THRESHOLD
            or frame_to_frame_similarity >= FRAME_TO_FRAME_SIMILARITY_THRESHOLD
        )

    if not same_scene and anchor is not None:
        streak = int(runtime.get_state("scene_change_streak", 0)) + 1
        runtime.set_state("scene_change_streak", streak)
        if streak < SCENE_CHANGE_CONFIRMATIONS:
            return
    else:
        runtime.set_state("scene_change_streak", 0)

    if not same_scene:
        next_generation = generation + 1
        if runtime.get_state("analyzing_generation") == next_generation:
            return
        generation = next_generation
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
        runtime.set_state("scene_started_at", frame.timestamp)
        runtime.set_state("completed_generation", None)
        runtime.set_state("precise_generation", None)
        runtime.set_state("analyzing_generation", generation)
        await _run_stream_phase(
            runtime,
            frame,
            generation,
            TOOL_PROMPT,
            PROGRESSIVE_MODELS,
            "base",
            "completed_generation",
        )
        return
    runtime.set_state("scene_anchor", _blend_scene_anchor(anchor, embedding))

    started_at = runtime.get_state("scene_started_at")
    if started_at is None:
        runtime.set_state("scene_started_at", frame.timestamp)
        started_at = frame.timestamp

    if runtime.get_state("analyzing_generation") == generation:
        return

    elapsed = max(0.0, float(frame.timestamp) - float(started_at))
    completed_generation = runtime.get_state("completed_generation")
    precise_generation = runtime.get_state("precise_generation")

    if (
        completed_generation == generation
        and precise_generation != generation
        and elapsed >= SCENE_HOLD_SECONDS
    ):
        runtime.set_state("analyzing_generation", generation)
        await _run_stream_phase(
            runtime,
            frame,
            generation,
            PRECISE_TOOL_PROMPT,
            PRECISE_MODELS,
            "precise",
            "precise_generation",
        )


async def on_stream_stop(runtime):
    runtime.set_state("analyzing_generation", None)


__all__ = [
    "compute_scene_embedding",
    "cosine_similarity",
    "is_same_scene",
    "on_frame",
    "on_stream_start",
    "on_stream_stop",
    "on_take_photo",
]
