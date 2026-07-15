from litellm_utils import call_take_photo_baseline_vlm

TOOL_PROMPT = "Detect any hand gesture in the image and name it concisely (for example: thumbs up, pointing, waving, raised hand). If you cannot recognize the gesture, say so. Do not explain what the gesture means."


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT)
