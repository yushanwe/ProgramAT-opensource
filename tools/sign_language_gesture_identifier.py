from litellm_utils import call_take_photo_vlm

TOOL_NAME = "sign_language_gesture_identifier"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Identify the sign language gesture or short signed phrase shown by the signer's hands. "
    "State the sign name or meaning concisely in plain language. "
    "If no clear sign language gesture is visible, say 'No clear sign detected.'"
)


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_vlm(
        image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME
    )
