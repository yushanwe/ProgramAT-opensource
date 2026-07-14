import io
import sys
import base64
from pathlib import Path
from PIL import Image

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import model_router  # noqa: E402  (path insertion must precede this import)


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


def pil_image_to_data_uri(pil_image: Image.Image, quality: int = 85) -> str:
    """Convert a PIL image to a JPEG base64 data URI."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{image_base64}'


def call_take_photo_baseline_vlm(image, prompt: str) -> str:
    """Send a single image with a prompt to the baseline VLM via the capability router.

    This is the standard entry point for take-photo tools. It routes the image
    and prompt through the ``general_reasoning`` capability and returns the
    model's text response directly.

    Args:
        image: The camera frame.  Accepts any image type supported by
            ``copilot_llm_call`` (numpy array, PIL Image, base64 data URI,
            or raw bytes).
        prompt: The task-specific instruction to send to the model.

    Returns:
        Audio-friendly text response from the model.
    """
    result = model_router.copilot_llm_call(
        capability="general_reasoning",
        messages=[{"role": "user", "content": prompt}],
        images=[image],
        metadata={"tool_name": "take_photo_baseline"},
    )
    return result.get("response", "")
