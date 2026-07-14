import io
import base64
from PIL import Image

from model_router_client import copilot_llm_call


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


def call_take_photo_baseline_vlm(*, image, prompt: str, tool_name: str = "take_photo_tool") -> str:
    """Run the default single-photo routed VLM path and return concise text."""
    result = copilot_llm_call(
        capability="general_reasoning",
        messages=[
            {"role": "system", "content": "Keep responses concise and audio-friendly."},
            {"role": "user", "content": prompt},
        ],
        images=[image],
        metadata={
            "tool_name": tool_name,
            "route_text": "single-frame baseline vision reasoning for take-photo tools",
        },
    )
    response = (result or {}).get("response", "")
    if isinstance(response, str):
        response = response.strip()
    return response or "I couldn't determine that from this photo."