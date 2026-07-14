import io
import base64
from PIL import Image


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


def call_take_photo_baseline_vlm(image, prompt: str, tool_name: str = "take_photo_baseline") -> str:
    """Send an image and prompt to the Copilot-routed general-reasoning model.

    This is the canonical entry point for single-shot take-photo tools.  Each
    tool defines one ``TOOL_PROMPT`` and delegates all model interaction here.

    Args:
        image: Camera frame – numpy ndarray (BGR), PIL Image, or a base64
               data-URI string accepted by the backend router.
        prompt: Task-specific instruction string.
        tool_name: Identifier for the calling tool, forwarded to backend logging.

    Returns:
        Audio-friendly response string from the vision-language model.
    """
    from model_router_client import copilot_llm_call  # local import keeps module lightweight

    result = copilot_llm_call(
        capability="general_reasoning",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for blind and low-vision users. Be concise and audio-friendly."},
            {"role": "user", "content": prompt},
        ],
        images=[image],
        metadata={
            "tool_name": tool_name,
            "route_text": "take-photo baseline visual question answering",
        },
    )
    return result.get("response", "Service temporarily unavailable. Please try again.")