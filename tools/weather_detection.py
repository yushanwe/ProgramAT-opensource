"""
Weather Detection Tool

Determines likely outdoor weather from a camera frame and provides
an audio-friendly result for blind or low vision users.
"""

import os
import re
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

WEATHER_LABELS = ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy', 'stormy', 'unclear']


def resize_image_if_needed(image: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    """Resize image while preserving aspect ratio."""
    height, width = image.shape[:2]
    max_width, max_height = max_size

    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def build_weather_prompt(streaming: bool = False) -> str:
    """Build weather classification prompt for Gemini Vision."""
    if streaming:
        return (
            "Classify the outdoor weather in this image. "
            "Return exactly one word from: sunny, cloudy, rainy, snowy, foggy, stormy, unclear."
        )

    return (
        "Classify the outdoor weather in this image. "
        "First word must be one of: sunny, cloudy, rainy, snowy, foggy, stormy, unclear. "
        "Then add a short umbrella recommendation in one sentence for a blind user."
    )


def estimate_weather_from_pixels(image: np.ndarray) -> str:
    """Fallback weather estimate when Gemini is unavailable."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return 'unclear'

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    saturation = float(np.mean(hsv[:, :, 1]))

    # Very bright and low saturation often indicates overcast/fog/snow scenes.
    if brightness > 190 and saturation < 35:
        return 'foggy'
    if brightness > 180 and saturation < 60:
        return 'cloudy'

    # Bright with moderate saturation is often sunny daytime.
    if brightness > 145 and saturation > 60:
        return 'sunny'

    # Dark scenes are usually cloudy/rainy/stormy. Prefer rainy for umbrella guidance.
    if brightness < 70:
        return 'rainy'
    if brightness < 95:
        return 'cloudy'

    return 'unclear'


def normalize_weather_label(raw_text: str) -> str:
    """Extract a weather label from model output."""
    if not raw_text:
        return 'unclear'

    lowered = raw_text.lower().strip()
    for label in WEATHER_LABELS:
        if re.search(rf'\b{label}\b', lowered):
            return label

    first_word = re.sub(r'[^a-z]', '', lowered.split()[0]) if lowered.split() else ''
    return first_word if first_word in WEATHER_LABELS else 'unclear'


def get_recommendation(weather: str) -> str:
    """Get a concise action recommendation for umbrella decisions."""
    if weather in ('rainy', 'stormy'):
        return 'Take an umbrella.'
    if weather == 'snowy':
        return 'Take warm waterproof gear.'
    if weather == 'foggy':
        return 'Use caution outside.'
    if weather in ('sunny', 'cloudy'):
        return 'No umbrella needed.'
    return 'Weather is unclear.'


def format_weather_for_audio(weather: str, streaming: bool = False) -> str:
    """Format weather output for speech."""
    weather_word = weather if weather in WEATHER_LABELS else 'unclear'
    recommendation = get_recommendation(weather_word)

    if streaming:
        short_map = {
            'rainy': 'Rainy. Take an umbrella.',
            'stormy': 'Stormy. Take an umbrella.',
            'snowy': 'Snowy. Wear warm waterproof gear.',
            'foggy': 'Foggy. Move carefully outside.',
            'sunny': 'Sunny. No umbrella needed.',
            'cloudy': 'Cloudy. No umbrella needed.',
            'unclear': 'Weather unclear right now.'
        }
        return short_map.get(weather_word, 'Weather unclear right now.')

    return f"It looks {weather_word} outside. {recommendation}"


def analyze_weather_with_gemini(
    image: np.ndarray,
    api_key: Optional[str] = None,
    model_name: str = 'gemini-3-flash-preview',
    streaming: bool = False
) -> Dict[str, Any]:
    """Analyze weather with Gemini Vision."""
    if not GEMINI_AVAILABLE:
        fallback = estimate_weather_from_pixels(image)
        return {
            'success': True,
            'weather': fallback,
            'source': 'fallback'
        }

    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY', '')

    if not api_key:
        fallback = estimate_weather_from_pixels(image)
        return {
            'success': True,
            'weather': fallback,
            'source': 'fallback'
        }

    try:
        processed = resize_image_if_needed(image)
        pil_image = convert_cv2_to_pil(processed)

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([build_weather_prompt(streaming), pil_image])

        parsed = normalize_weather_label(getattr(response, 'text', ''))
        return {
            'success': True,
            'weather': parsed,
            'source': 'gemini'
        }
    except Exception as exc:
        fallback = estimate_weather_from_pixels(image)
        return {
            'success': False,
            'weather': fallback,
            'source': 'fallback',
            'error': str(exc)
        }


def main(image: np.ndarray, input_data: Optional[Dict] = None) -> Any:
    """
    Main entry point for weather detection.

    Args:
        image: Camera frame as OpenCV BGR numpy array.
        input_data: Optional dict containing:
            - stream_mode / streaming / track_mode: bool
            - api_key: Optional Gemini API key override
            - model: Optional model override

    Returns:
        Audio-friendly weather output string or error dict.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            'audio': {
                'type': 'error',
                'text': 'No camera image available for weather detection.'
            },
            'text': 'No camera image available for weather detection.'
        }

    config = input_data if isinstance(input_data, dict) else {}
    streaming = bool(
        config.get('stream_mode', False)
        or config.get('streaming', False)
        or config.get('track_mode', False)
    )
    api_key = config.get('api_key')
    model = config.get('model', 'gemini-3-flash-preview')

    result = analyze_weather_with_gemini(
        image=image,
        api_key=api_key,
        model_name=model,
        streaming=streaming
    )

    return format_weather_for_audio(result.get('weather', 'unclear'), streaming=streaming)


__all__ = [
    'main',
    'resize_image_if_needed',
    'convert_cv2_to_pil',
    'build_weather_prompt',
    'estimate_weather_from_pixels',
    'normalize_weather_label',
    'get_recommendation',
    'format_weather_for_audio',
    'analyze_weather_with_gemini'
]
