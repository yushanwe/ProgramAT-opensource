"""
Lightweight client for routing tool LLM calls through the shared model stack.
"""

from __future__ import annotations

import os
from typing import Any


TASK_CATEGORY_PROMPTS = {
    'simple_parsing': 'Extract the requested information accurately and succinctly.',
    'object_detection': 'Identify visible objects relevant to the user request.',
    'object_localization': 'Identify the requested object and describe where it is.',
    'ocr': 'Read visible text accurately and keep the response concise.',
    'visual_understanding': 'Describe what is visible with concise, user-facing language.',
    'visual_reasoning': 'Reason about the visible scene and answer directly for the user.',
    'navigation': 'Focus on safe navigation details and spatial relationships.',
    'summarization': 'Summarize the important details briefly and clearly.',
    'code_generation': 'Generate correct code that follows the provided constraints.',
    'general_reasoning': 'Answer the request directly with concise reasoning.',
}


def _select_model_name(input_data: dict[str, Any] | None) -> str:
    requested = ''
    if isinstance(input_data, dict):
        requested = str(input_data.get('model_name') or '').strip()

    env_model = str(os.environ.get('LLM_MODEL') or '').strip()
    model_name = requested or env_model or 'gemini-3-flash-preview'
    if model_name == 'gemini-2.0-flash':
        return 'gemini-3-flash-preview'
    return model_name


def _image_to_data_uri(image: Any) -> str:
    from PIL import Image
    from litellm_utils import pil_image_to_data_uri

    if hasattr(image, 'save') and hasattr(image, 'size'):
        pil_image = image
    else:
        shape = getattr(image, 'shape', ())
        if len(shape) >= 3 and shape[2] >= 3:
            image = image[:, :, ::-1]
        pil_image = Image.fromarray(image)

    return pil_image_to_data_uri(pil_image)


def llm_call(
    task_category: str,
    prompt: str,
    image: Any | None = None,
    input_data: dict[str, Any] | None = None,
) -> str:
    """
    Run a single shared-model call for a tool request.
    """

    import litellm
    from litellm_utils import extract_text, resolve_api_key, resolve_model_name

    model_name = _select_model_name(input_data)
    normalized_model = resolve_model_name(model_name)
    system_prompt = TASK_CATEGORY_PROMPTS.get(
        task_category,
        TASK_CATEGORY_PROMPTS['general_reasoning'],
    )

    if image is None:
        user_content: Any = prompt
    else:
        user_content = [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': _image_to_data_uri(image)}},
        ]

    response = litellm.completion(
        model=normalized_model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ],
        api_key=resolve_api_key(model_name),
    )
    return extract_text(response)
