from litellm_utils import call_take_photo_vlm

TOOL_NAME = "temporal_sign_gesture_identifier"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Identify the sign or short signed phrase that is most clearly shown by the visible hand posture in this image. "
    "Return only a concise, audio-friendly sign label or short phrase. "
    "If no sign is clear, say 'No clear sign visible.'"
)


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
