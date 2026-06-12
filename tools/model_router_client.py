"""Shared model router client for ProgramAT tools."""

from __future__ import annotations

import os
from typing import Any

from litellm_utils import resolve_api_key, resolve_model_name, extract_text, pil_image_to_data_uri

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in some test envs
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency in some test envs
    Image = None

try:
    import litellm
except ImportError:  # pragma: no cover - optional dependency in some test envs
    litellm = None


SUPPORTED_TASK_CATEGORIES = {
    'simple_parsing',
    'object_detection',
    'object_localization',
    'ocr',
    'visual_understanding',
    'visual_reasoning',
    'navigation',
    'summarization',
    'code_generation',
    'general_reasoning',
}


def _image_to_data_uri(image: Any) -> str:
    """Convert OpenCV/PIL image input into a JPEG data URI for vision models."""
    if image is None:
        return ''

    if Image is None:
        raise RuntimeError('Pillow is not available for image processing.')

    # PIL image input
    if hasattr(image, 'save') and hasattr(image, 'mode'):
        return pil_image_to_data_uri(image)

    # OpenCV BGR ndarray input
    if cv2 is not None and hasattr(image, 'shape'):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image_to_data_uri(pil_image)

    raise ValueError('Unsupported image format. Expected OpenCV ndarray or PIL image.')


def route_model(
    task_category: str,
    prompt: str,
    image: Any = None,
    input_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a single model-backed request through a shared completion pathway."""
    if task_category not in SUPPORTED_TASK_CATEGORIES:
        return {
            'success': False,
            'error': f'Unsupported task category: {task_category}',
            'text': 'I could not process that request.',
            'task_category': task_category,
        }

    if litellm is None:
        return {
            'success': False,
            'error': 'LiteLLM is not installed.',
            'text': 'Model services are unavailable right now.',
            'task_category': task_category,
        }

    config = input_data or {}
    requested_model = config.get('model') or os.environ.get('LLM_MODEL') or os.environ.get('GEMINI_MODEL') or 'gemini-3-flash-preview'
    model_name = resolve_model_name(requested_model)
    api_key = resolve_api_key(model_name, config.get('api_key', ''))

    if not api_key:
        return {
            'success': False,
            'error': 'No provider API key configured.',
            'text': 'I need an API key before I can analyze this image.',
            'task_category': task_category,
        }

    try:
        content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
        if image is not None:
            image_data_uri = _image_to_data_uri(image)
            if image_data_uri:
                content.append({'type': 'image_url', 'image_url': {'url': image_data_uri}})

        response = litellm.completion(
            model=model_name,
            messages=[{'role': 'user', 'content': content}],
            api_key=api_key,
        )
        text = extract_text(response)
        return {
            'success': True,
            'text': text,
            'task_category': task_category,
        }
    except Exception as exc:
        return {
            'success': False,
            'error': str(exc),
            'text': 'I could not complete that analysis.',
            'task_category': task_category,
        }


def run_visual_understanding(image: Any, prompt: str, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience wrapper for visual-understanding tool tasks."""
    return route_model(
        task_category='visual_understanding',
        prompt=prompt,
        image=image,
        input_data=input_data,
    )
