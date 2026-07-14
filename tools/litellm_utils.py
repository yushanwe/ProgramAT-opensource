import io
import sys
import base64
from pathlib import Path
from PIL import Image

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

# Ensure the tools directory is on sys.path so model_router_client is importable
# when litellm_utils itself is used as the import root (backend exec context).
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from model_router_client import copilot_llm_call


def call_take_photo_baseline_vlm(image, prompt: str) -> str:
    """Call the baseline VLM for a take-photo tool.

    Take-photo mode tools capture a single frame, send it to the vision model
    with a task-specific prompt, and return the spoken result directly.  They
    make exactly one model call and no specialist calls.

    Converts *image* (an OpenCV BGR numpy array, a PIL Image, or an existing
    base-64 data URI string) to the format expected by the backend router,
    then issues a single ``general_reasoning`` capability call with *prompt*.
    Returns the model's plain-text response.
    """
    if image is None:
        return "No camera image available"

    if np is not None and isinstance(image, np.ndarray):
        if cv2 is None:
            return "Camera image processing is unavailable. Please try again later."
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_data_uri = pil_image_to_data_uri(pil_image)
    elif isinstance(image, str):
        image_data_uri = image
    else:
        image_data_uri = pil_image_to_data_uri(image)

    result = copilot_llm_call(
        capability="general_reasoning",
        messages=[{"role": "user", "content": prompt}],
        images=[image_data_uri],
        metadata={"tool_name": "take_photo_baseline_vlm", "route_text": prompt},
    )
    return result.get("response", "")


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