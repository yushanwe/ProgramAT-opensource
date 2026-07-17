from litellm_utils import call_take_photo_baseline_vlm

TOOL_NAME = "play_card_identifier"
TOOL_PROMPT = (
    "Look at the playing cards shown. Identify which card was most recently played — "
    "it may appear face-up on the table or separate from the main hand. "
    "Name the card by its rank and suit, for example 'Ace of Spades' or 'Seven of Hearts'. "
    "If no recently played card is clearly visible, say so briefly."
)


def main(image, input_data):
    if image is None:
        return "No image available to identify the played card."
    return call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
