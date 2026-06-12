"""
Checks whether the visible outfit pieces match and gives brief spoken feedback.
"""

from __future__ import annotations

from model_router_client import llm_call


def normalize_config(input_data=None):
    if isinstance(input_data, dict):
        return input_data
    return {}


def build_matching_prompt(occasion='everyday', include_suggestion=True):
    guidance = (
        "Look for the main top and main bottom clothing item on the person. "
        f"Judge whether they match for a {occasion} outfit based on color, pattern, and formality. "
        "If either the top or bottom is not visible enough, say that clearly instead of guessing. "
        "Keep the answer audio-friendly and limited to two short sentences."
    )
    if include_suggestion:
        guidance += " If they do not match, include one short suggestion."
    return guidance


def clean_response(response_text):
    text = ' '.join(str(response_text or '').split())
    if not text:
        return 'I could not judge whether the outfit matches.'

    sentences = []
    for sentence in text.replace('!', '.').replace('?', '.').split('.'):
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)

    if not sentences:
        return text

    return '. '.join(sentences[:2]) + '.'


def error_result(message):
    return {
        'audio': {
            'type': 'error',
            'text': message,
        },
        'text': message,
    }


def main(image, input_data=None):
    config = normalize_config(input_data)
    if image is None:
        return error_result('I need a photo showing your outfit to compare it.')

    prompt = build_matching_prompt(
        occasion=str(config.get('occasion') or 'everyday'),
        include_suggestion=bool(config.get('include_suggestion', True)),
    )

    try:
        response = llm_call(
            task_category='visual_reasoning',
            prompt=prompt,
            image=image,
            input_data=config,
        )
        return clean_response(response)
    except Exception:
        return error_result('I could not check whether your clothing matches right now.')
