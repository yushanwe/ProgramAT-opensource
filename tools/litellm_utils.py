from __future__ import annotations

import io
import base64
import importlib.util
from functools import lru_cache
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


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
    if Image is None:
        raise ImportError("Pillow is required for pil_image_to_data_uri")
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{image_base64}'


@lru_cache(maxsize=1)
def _load_backend_litellm_utils():
    backend_path = Path(__file__).resolve().parent.parent / "backend" / "litellm_utils.py"
    spec = importlib.util.spec_from_file_location(
        "_programat_backend_litellm_utils",
        backend_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend litellm_utils from {backend_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_take_photo_baseline_vlm(image, prompt: str, tool_name: str | None = None) -> str:
    return _load_backend_litellm_utils().call_take_photo_baseline_vlm(
        image=image,
        prompt=prompt,
        tool_name=tool_name,
    )