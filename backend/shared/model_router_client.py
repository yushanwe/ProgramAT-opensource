"""Small shared wrapper for routed model-backed tool calls."""

from __future__ import annotations

import base64
import io
import os
from typing import Any


ALLOWED_TASK_CATEGORIES = {
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

DEFAULT_MODELS = {
    'simple_parsing': 'gemini-2.5-flash-lite',
    'object_detection': 'gemini-3-flash-preview',
    'object_localization': 'gemini-3-flash-preview',
    'ocr': 'gemini-3-flash-preview',
    'visual_understanding': 'gemini-3-flash-preview',
    'visual_reasoning': 'gemini-3-flash-preview',
    'navigation': 'gemini-3-flash-preview',
    'summarization': 'gemini-2.5-flash-lite',
    'code_generation': 'gemini-2.5-flash-lite',
    'general_reasoning': 'gemini-2.5-flash-lite',
}


class ModelRouterError(RuntimeError):
    """Raised when a routed model request cannot be completed."""


def _resolve_model_name(model_name: str, default_model: str) -> str:
    raw = (model_name or default_model).strip()
    known_providers = (
        'openrouter/',
        'gemini/',
        'openai/',
        'anthropic/',
        'ollama/',
        'groq/',
        'vertex_ai/',
    )

    if any(raw.startswith(provider) for provider in known_providers):
        return raw
    if raw.startswith('gemini'):
        return f'gemini/{raw}'
    if raw.startswith('claude'):
        return f'anthropic/{raw}'
    return raw


def _resolve_api_key(model_name: str, explicit_api_key: str = '') -> str:
    if explicit_api_key:
        return explicit_api_key

    normalized = (model_name or '').lower()
    if normalized.startswith('gemini'):
        return os.environ.get('GEMINI_API_KEY', '')
    if normalized.startswith('claude'):
        return os.environ.get('ANTHROPIC_API_KEY', '')
    if normalized.startswith('openai') or normalized.startswith('gpt'):
        return os.environ.get('OPENAI_API_KEY', '')
    return os.environ.get('OPENAI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')


def _extract_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get('text', ''))
            elif hasattr(part, 'text'):
                parts.append(getattr(part, 'text', '') or '')
        return ''.join(parts).strip()
    return str(content).strip()


def _pil_image_to_data_uri(pil_image: Any, quality: int = 85) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{image_base64}'


def _require_supported_task(task_category: str) -> None:
    if task_category not in ALLOWED_TASK_CATEGORIES:
        supported = ', '.join(sorted(ALLOWED_TASK_CATEGORIES))
        raise ModelRouterError(f'Unsupported task category: {task_category}. Supported categories: {supported}.')


def _import_litellm():
    try:
        import litellm
    except ImportError as exc:
        raise ModelRouterError('LiteLLM is not installed.') from exc
    return litellm


def _image_to_data_uri(image: Any) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ModelRouterError('Pillow is not installed.') from exc

    if image is None:
        raise ModelRouterError('No image available for routed model request.')

    try:
        shape = getattr(image, 'shape', None)
        if not shape:
            raise TypeError('Image has no shape.')

        if len(shape) == 2:
            pil_image = Image.fromarray(image)
        elif len(shape) == 3 and shape[2] >= 3:
            rgb_image = image[:, :, ::-1]
            pil_image = Image.fromarray(rgb_image)
        else:
            raise TypeError(f'Unsupported image shape: {shape}.')
    except Exception as exc:
        raise ModelRouterError(f'Unable to convert image for routed model request: {exc}') from exc

    return _pil_image_to_data_uri(pil_image)


def route_model_vision_request(
    task_category: str,
    prompt: str,
    image: Any,
    *,
    model_name: str | None = None,
    api_key: str = '',
    system_instruction: str | None = None,
) -> str:
    _require_supported_task(task_category)

    litellm = _import_litellm()
    resolved_model = _resolve_model_name(model_name or DEFAULT_MODELS[task_category], DEFAULT_MODELS[task_category])
    resolved_api_key = _resolve_api_key(resolved_model, api_key)

    if not resolved_api_key:
        raise ModelRouterError('API key not configured for routed model request.')

    image_data_uri = _image_to_data_uri(image)
    messages = []

    if system_instruction:
        messages.append({'role': 'system', 'content': system_instruction})

    messages.append(
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': image_data_uri}},
            ],
        }
    )

    response = litellm.completion(
        model=resolved_model,
        messages=messages,
        api_key=resolved_api_key,
    )
    return _extract_text(response)


def route_visual_reasoning(
    image: Any,
    prompt: str,
    *,
    model_name: str | None = None,
    api_key: str = '',
    system_instruction: str | None = None,
) -> str:
    return route_model_vision_request(
        'visual_reasoning',
        prompt,
        image,
        model_name=model_name,
        api_key=api_key,
        system_instruction=system_instruction,
    )
