from litellm_utils import call_take_photo_baseline_vlm

TOOL_PROMPT = "Tell me when someone is pointing, waving, giving a thumbs up, raising a hand, or making another common hand gesture, and briefly explain what it likely means."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
