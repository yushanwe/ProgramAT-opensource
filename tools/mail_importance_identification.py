"""Take-photo tool that labels important mail in a captured image."""

from litellm_utils import call_take_photo_baseline_vlm


TOOL_PROMPT = (
    "Identify only the important pieces of mail in this photo, such as bills, legal, "
    "medical, government, bank, or deadline-related mail. Briefly label each important "
    "item. If nothing looks important, say so."
)


def main(image, input_data=None):
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
