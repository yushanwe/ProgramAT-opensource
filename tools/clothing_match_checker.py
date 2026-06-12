"""Clothing match checker tool."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_INSTRUCTION = (
    'You are helping a blind user decide whether their outfit matches. '
    'Be kind, direct, and practical. Do not mention technical uncertainty unless needed.'
)


def load_model_router_client() -> tuple[type[Exception], Any]:
    cached = globals().get('_clothing_match_router_client')
    if cached:
        return cached

    candidate_paths = []
    file_path = globals().get('__file__')
    if file_path:
        repo_root = Path(file_path).resolve().parents[1]
        candidate_paths.append(repo_root / 'backend' / 'shared' / 'model_router_client.py')

    cwd = Path.cwd()
    candidate_paths.append(cwd / 'shared' / 'model_router_client.py')
    candidate_paths.append(cwd / 'backend' / 'shared' / 'model_router_client.py')

    for candidate in candidate_paths:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location('programat_model_router_client', candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                loaded = (module.ModelRouterError, module.route_visual_reasoning)
                globals()['_clothing_match_router_client'] = loaded
                return loaded

    raise RuntimeError('Could not load backend/shared/model_router_client.py')


def build_clothing_match_prompt(input_data: dict[str, Any] | None = None) -> str:
    input_data = input_data or {}
    occasion = str(input_data.get('occasion', '')).strip()
    style_goal = str(input_data.get('style_goal', '')).strip()

    prompt = (
        'Look at the visible outfit and judge whether the clothing matches well. '
        'Focus on color coordination, pattern compatibility, and overall formality. '
        'Mention the main visible top and bottom items when possible. '
        'If the full outfit is not visible enough, say so clearly and ask the user to show more. '
        'Respond in one or two short spoken sentences.'
    )

    if occasion:
        prompt += f' The outfit is meant for {occasion}.'
    if style_goal:
        prompt += f' The user wants it to feel {style_goal}.'

    return prompt


def classify_audio_type(message: str) -> str:
    normalized = message.lower()
    warning_signals = (
        "can't see enough",
        'cannot see enough',
        'not visible enough',
        'show more',
        'cannot tell',
        "can't tell",
    )
    if any(signal in normalized for signal in warning_signals):
        return 'warning'
    return 'speech'


def format_match_feedback(response_text: str) -> str:
    cleaned = ' '.join(str(response_text or '').split())
    if not cleaned:
        return 'I could not judge whether the outfit matches.'
    if cleaned[-1] not in '.!?':
        cleaned += '.'
    return cleaned


def main(image: Any, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
    if image is None:
        message = 'No camera image available for clothing match checking.'
        return {
            'audio': {'type': 'error', 'text': message},
            'text': message,
        }

    input_data = input_data or {}
    prompt = build_clothing_match_prompt(input_data)
    model_name = input_data.get('model') or os.environ.get('LLM_MODEL') or os.environ.get('GEMINI_MODEL')
    api_key = str(input_data.get('api_key', '') or '')

    try:
        model_router_error, route_visual_reasoning = load_model_router_client()
        response_text = route_visual_reasoning(
            image,
            prompt,
            model_name=model_name,
            api_key=api_key,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
        )
        message = format_match_feedback(response_text)
        return {
            'audio': {'type': classify_audio_type(message), 'text': message},
            'text': message,
        }
    except Exception as exc:
        model_router_error = globals().get('_clothing_match_router_client', (RuntimeError, None))[0]
        if isinstance(exc, model_router_error):
            message = f'Clothing match checker is unavailable: {exc}'
        else:
            message = 'Clothing match checker could not analyze this image right now.'
        return {
            'audio': {'type': 'error', 'text': message},
            'text': message,
        }
