"""Clothing match checker tool for ProgramAT."""

from __future__ import annotations

from typing import Any

from model_router_client import run_visual_understanding


def _build_prompt() -> str:
    return (
        "You are helping a blind user check outfit coordination. "
        "Look at the shirt/top and pants/bottom and decide if they match well. "
        "Respond in plain natural speech with one short verdict sentence and one short reason. "
        "If only one garment is visible, say what is missing and ask for a fuller view."
    )


def _format_match_feedback(text: str) -> str:
    cleaned = ' '.join((text or '').split())
    if not cleaned:
        return 'I could not judge the outfit match from this view.'
    return cleaned


def main(image: Any, input_data: dict[str, Any] | None = None) -> str | dict[str, Any]:
    """Check whether visible clothing items match."""
    if image is None:
        return {
            'audio': {
                'type': 'error',
                'text': 'No camera image available to check clothing match.',
            },
            'text': 'No camera image available to check clothing match.',
        }

    result = run_visual_understanding(image=image, prompt=_build_prompt(), input_data=input_data)
    if not result.get('success'):
        return {
            'audio': {
                'type': 'error',
                'text': result.get('text', 'I could not check your outfit right now.'),
            },
            'text': result.get('error', result.get('text', 'Outfit analysis failed.')),
        }

    spoken = _format_match_feedback(result.get('text', ''))
    return {
        'audio': {
            'type': 'speech',
            'text': spoken,
            'interrupt': False,
        },
        'text': spoken,
    }
