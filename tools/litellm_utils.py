import io
import base64
import logging

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency in lightweight test envs
    Image = None


logger = logging.getLogger(__name__)


def extract_text(response) -> str:
    """Extract text content from a LiteLLM response object."""
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


def pil_image_to_data_uri(pil_image, quality: int = 85) -> str:
    """Convert a PIL image to a JPEG base64 data URI."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{image_base64}'


def _image_to_data_uri(image) -> str:
    """Convert a ProgramAT tool image input to a JPEG data URI."""
    if isinstance(image, str):
        return image
    if Image is None:
        raise ImportError('Pillow is required for take-photo image conversion')
    if isinstance(image, Image.Image):
        return pil_image_to_data_uri(image.convert('RGB'))
    if image is None:
        raise ValueError('No image available')
    if not hasattr(image, 'shape') or not hasattr(image, '__getitem__'):
        raise TypeError('Unsupported image type for take-photo conversion')

    if getattr(image, 'ndim', 0) == 2:
        pil_image = Image.fromarray(image).convert('RGB')
    elif getattr(image, 'shape', [None, None, None])[2] == 4:
        pil_image = Image.fromarray(image[:, :, [2, 1, 0, 3]], mode='RGBA').convert('RGB')
    else:
        pil_image = Image.fromarray(image[:, :, ::-1])

    return pil_image_to_data_uri(pil_image)


def call_take_photo_baseline_vlm(image, prompt: str):
    """Run a single routed take-photo reasoning call and return spoken text."""
    if image is None:
        return {
            'audio': {'type': 'error', 'text': 'No photo available.'},
            'text': 'No photo available.',
        }

    try:
        from model_router_client import copilot_llm_call

        result = copilot_llm_call(
            capability='general_reasoning',
            messages=[
                {'role': 'system', 'content': 'Keep responses concise and audio-friendly.'},
                {'role': 'user', 'content': prompt.strip()},
            ],
            images=[_image_to_data_uri(image)],
            metadata={
                'tool_name': 'take_photo_baseline_vlm',
                'route_text': 'baseline take-photo tool response',
            },
        )
        return result['response']
    except Exception:
        logger.exception('Take-photo baseline VLM call failed')
        return {
            'audio': {'type': 'error', 'text': 'I could not analyze that photo.'},
            'text': 'I could not analyze that photo.',
        }