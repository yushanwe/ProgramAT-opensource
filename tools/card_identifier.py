"""Take-photo tool that reads visible playing cards and announces rank and suit."""

from litellm_utils import call_take_photo_baseline_vlm


TOOL_NAME = "card_identifier"
TOOL_PROMPT = (
    "Read the playing cards visible in the user's hand in this photo. "
    "Report each visible card with rank and suit as a concise spoken list, for example "
    "'Ace of spades, ten of hearts'. "
    "If no card is readable, say 'No readable cards visible.'"
)


def main(image, input_data=None):
    if image is None:
        return "No camera image available to read cards."

    try:
        return call_take_photo_baseline_vlm(
            image=image,
            prompt=TOOL_PROMPT,
            tool_name=TOOL_NAME,
        )
    except Exception:
        return "I couldn't read the cards from this photo. Please hold them steady and try again."
