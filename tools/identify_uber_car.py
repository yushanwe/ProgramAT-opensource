"""Take-photo tool to identify the car in front of the user."""

from litellm_utils import call_take_photo_baseline_vlm

TOOL_NAME = "identify_uber_car"
TOOL_PROMPT = (
    "Identify the car directly in front of the user so they can verify whether it is their Uber. "
    "Name the brand, make, model, color, and license plate number if visible. "
    "If any detail is not visible or uncertain, say that clearly. "
    "If no car is clearly visible, say 'No car is clearly visible.' "
    "Return a concise, audio-friendly answer."
)


def main(image, input_data):
    if image is None:
        return "No camera image available."

    try:
        return call_take_photo_baseline_vlm(
            image=image,
            prompt=TOOL_PROMPT,
            tool_name=TOOL_NAME,
        )
    except Exception:
        return "I couldn't identify the car from that photo."
