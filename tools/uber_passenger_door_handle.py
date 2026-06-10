"""
Uber Vehicle and Passenger Door Handle Finder

Capability Pipeline (minimal stages):
1) visual_reasoning
   - Goal: Identify whether the target Uber vehicle is in view and where it is.
   - Input: Full camera frame + user-provided Uber details.
   - Output: target_found, vehicle_description, vehicle_clock, and confidence.
2) object_localization
   - Goal: Locate the passenger door handle on that identified vehicle.
   - Input: Same frame + stage-1 vehicle context.
   - Output: handle_visible, handle_direction, and concise navigation cue.

This tool is designed for blind and low-vision users. It returns concise,
audio-friendly guidance as either a spoken string or an audio/text dictionary.
"""

import cv2
import numpy as np
from typing import Any, Dict, Optional
import json
from PIL import Image

from litellm_utils import resolve_model_name, resolve_api_key, extract_text, pil_image_to_data_uri

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False

STREAMING_WORD_LIMIT = 15
DEFAULT_MODEL = 'gemini-3-flash-preview'


def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1280, 1280)) -> np.ndarray:
    """Resize large frames while keeping aspect ratio for faster inference."""
    h, w = image.shape[:2]
    max_w, max_h = max_size
    if w <= max_w and h <= max_h:
        return image

    scale = min(max_w / float(w), max_h / float(h))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _normalize_text(value: Any) -> str:
    if not value:
        return ''
    return str(value).strip()


def build_target_description(input_data: Optional[Dict[str, Any]]) -> str:
    """Build a compact, human-readable description of the target Uber."""
    data = input_data if isinstance(input_data, dict) else {}

    uber_details = data.get('uber_details') if isinstance(data.get('uber_details'), dict) else {}

    color = _normalize_text(uber_details.get('color') or data.get('vehicle_color') or data.get('color'))
    make = _normalize_text(uber_details.get('make') or data.get('vehicle_make') or data.get('make'))
    model = _normalize_text(uber_details.get('model') or data.get('vehicle_model') or data.get('model'))
    plate = _normalize_text(uber_details.get('plate') or data.get('license_plate') or data.get('plate'))

    pieces = [p for p in [color, make, model] if p]
    descriptor = ' '.join(pieces) if pieces else 'the assigned Uber vehicle'

    if plate:
        descriptor = f"{descriptor}, plate {plate}"

    return descriptor


def build_analysis_prompt(target_description: str, is_streaming: bool) -> str:
    """Prompt for two-stage analysis in one structured response."""
    word_limit_rule = (
        "For navigation_text, use 15 words max."
        if is_streaming
        else "For navigation_text, keep it concise and natural for speech."
    )

    return (
        "You are assisting a blind user finding their Uber and passenger door handle. "
        "Return ONLY valid JSON with this exact schema: "
        "{"
        '"target_found": boolean, '
        '"vehicle_description": string, '
        '"vehicle_clock": string, '
        '"handle_visible": boolean, '
        '"handle_direction": string, '
        '"navigation_text": string'
        "}. "
        "Stage 1 (visual_reasoning): decide if the target Uber is visible. "
        "Stage 2 (object_localization): if visible, locate the passenger door handle and give guidance. "
        f"Target Uber details: {target_description}. "
        "Use only camera-visible directions. For clock-face, only use 1-3 and 9-12. "
        "Prefer clock-face for vehicle position. "
        "If handle is not visible, guide user to move camera toward passenger-side door area. "
        f"{word_limit_rule}"
    )


def call_backend_visual_capability(image_data_uri: str, prompt: str, model_name: str, api_key: str) -> str:
    """
    Use backend capability interface when available, otherwise fallback to LiteLLM.
    """
    capability_fn = globals().get('run_visual_understanding')
    if callable(capability_fn):
        result = capability_fn(prompt=prompt, image=image_data_uri)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)

    if not LITELLM_AVAILABLE:
        raise RuntimeError('LiteLLM not available')

    try:
        response = litellm.completion(
            model=resolve_model_name(model_name),
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_data_uri}},
                    ],
                }
            ],
            api_key=api_key,
        )
        return extract_text(response)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def parse_analysis_json(raw_text: str) -> Dict[str, Any]:
    """Parse model output to expected dict shape, tolerating markdown wrappers."""
    text = (raw_text or '').strip()
    if text.startswith('```'):
        text = text.replace('```json', '').replace('```', '').strip()

    parsed = json.loads(text)

    return {
        'target_found': bool(parsed.get('target_found', False)),
        'vehicle_description': _normalize_text(parsed.get('vehicle_description')),
        'vehicle_clock': _normalize_text(parsed.get('vehicle_clock')),
        'handle_visible': bool(parsed.get('handle_visible', False)),
        'handle_direction': _normalize_text(parsed.get('handle_direction')),
        'navigation_text': _normalize_text(parsed.get('navigation_text')),
    }


def enforce_word_limit(text: str, max_words: int = STREAMING_WORD_LIMIT) -> str:
    """Trim speech output for streaming etiquette."""
    words = (text or '').split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def create_audio_response(analysis: Dict[str, Any], target_description: str, is_streaming: bool) -> Dict[str, Any]:
    """Build final user guidance from stage outputs."""
    if not analysis.get('target_found'):
        base_text = (
            f"I can't confirm {target_description}. Pan slowly left and right to find your Uber."
        )
        speak_text = enforce_word_limit(base_text) if is_streaming else base_text
        return {
            'audio': {'type': 'speech', 'text': speak_text, 'rate': 1.0, 'interrupt': False},
            'text': speak_text,
        }

    nav_text = analysis.get('navigation_text') or ''

    if not nav_text:
        vehicle_clock = analysis.get('vehicle_clock') or 'ahead'
        if analysis.get('handle_visible'):
            handle_dir = analysis.get('handle_direction') or 'near center'
            nav_text = f"Your Uber is at {vehicle_clock}. Handle is {handle_dir}. Reach there."
        else:
            nav_text = f"Your Uber is at {vehicle_clock}. Move camera toward passenger door area."

    if is_streaming:
        nav_text = enforce_word_limit(nav_text)

    audio_type = 'speech'
    if analysis.get('target_found') and analysis.get('handle_visible'):
        audio_type = 'success'

    return {
        'audio': {
            'type': audio_type,
            'text': nav_text,
            'rate': 1.0,
            'interrupt': bool(analysis.get('handle_visible')),
        },
        'text': nav_text,
    }


def run(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Primary processing function for compatibility with building-block usage."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {'type': 'error', 'text': 'No camera image available.', 'rate': 1.0},
            'text': 'No camera image available.',
        }

    config = input_data if isinstance(input_data, dict) else {}
    is_streaming = bool(config.get('is_streaming', False))
    model_name = _normalize_text(config.get('model')) or DEFAULT_MODEL

    api_key = resolve_api_key(model_name, _normalize_text(config.get('api_key')))
    if not api_key and not callable(globals().get('run_visual_understanding')):
        return {
            'audio': {'type': 'error', 'text': 'Vision API key is not configured.', 'rate': 1.0},
            'text': 'Vision API key is not configured.',
        }

    target_description = build_target_description(config)

    try:
        prepared = resize_image_if_needed(image)
        image_data_uri = pil_image_to_data_uri(convert_cv2_to_pil(prepared))
        prompt = build_analysis_prompt(target_description, is_streaming)

        raw = call_backend_visual_capability(
            image_data_uri=image_data_uri,
            prompt=prompt,
            model_name=model_name,
            api_key=api_key,
        )
        analysis = parse_analysis_json(raw)
    except json.JSONDecodeError:
        fallback = 'I could not interpret the scene yet. Pan slowly and try again.'
        if is_streaming:
            fallback = enforce_word_limit(fallback)
        return {'audio': {'type': 'warning', 'text': fallback, 'rate': 1.0}, 'text': fallback}
    except RuntimeError as exc:
        error_text = str(exc).lower()
        if 'api key' in error_text or 'auth' in error_text:
            fallback = 'Vision service authorization failed. Check API key configuration.'
        elif 'rate' in error_text or 'quota' in error_text:
            fallback = 'Vision service is busy. Please wait a moment and try again.'
        else:
            fallback = 'Vision service is unavailable right now. Please try again in a moment.'
        if is_streaming:
            fallback = enforce_word_limit(fallback)
        return {'audio': {'type': 'warning', 'text': fallback, 'rate': 1.0}, 'text': fallback}
    except KeyError:
        fallback = 'Scene analysis was incomplete. Move slightly and try again.'
        if is_streaming:
            fallback = enforce_word_limit(fallback)
        return {'audio': {'type': 'warning', 'text': fallback, 'rate': 1.0}, 'text': fallback}
    except ValueError:
        fallback = 'Input settings look invalid. Reset tool settings and try again.'
        if is_streaming:
            fallback = enforce_word_limit(fallback)
        return {'audio': {'type': 'warning', 'text': fallback, 'rate': 1.0}, 'text': fallback}
    except TypeError:
        fallback = 'Image processing failed. Reframe the camera and try again.'
        if is_streaming:
            fallback = enforce_word_limit(fallback)
        return {'audio': {'type': 'warning', 'text': fallback, 'rate': 1.0}, 'text': fallback}

    return create_audio_response(analysis, target_description, is_streaming)


def main(image: np.ndarray, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """ProgramAT tool entry point."""
    return run(image, input_data)


__all__ = [
    'main',
    'run',
    'build_target_description',
    'build_analysis_prompt',
    'parse_analysis_json',
    'create_audio_response',
    'enforce_word_limit',
]
