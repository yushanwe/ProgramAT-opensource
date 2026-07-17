import re

from litellm_utils import call_take_photo_baseline_vlm

TOOL_NAME = "card_identification"
TOOL_PROMPT = (
    "List all playing cards visible in this image by rank and suit "
    "(for example: ace of spades, seven of hearts). "
    "List them left to right as they appear. "
    "If no playing cards are visible, say 'No cards visible.'"
)

_CARD_PATTERN = re.compile(
    r"((?:ace|two|three|four|five|six|seven|eight|nine|ten|jack|queen|king|[2-9]|10)"
    r"\s+of\s+"
    r"(?:spades?|hearts?|diamonds?|clubs?))",
    re.IGNORECASE,
)

_before_hand = None  # stores parsed card list from the before-photo invocation


def _parse_cards(description):
    """Return a list of card strings found in a VLM description."""
    return [m.lower() for m in _CARD_PATTERN.findall(description)]


def main(image, input_data):
    global _before_hand
    if image is None:
        return "No image available. Please take a photo."
    result = call_take_photo_baseline_vlm(image=image, prompt=TOOL_PROMPT, tool_name=TOOL_NAME)
    cards = _parse_cards(result)
    if _before_hand is None:
        _before_hand = cards if cards else []
        if cards:
            return f"Before hand captured: {result}. Play a card, then take another photo."
        return "No playing cards detected. Please try again with your full hand visible."
    before = list(_before_hand)
    _before_hand = None
    missing = [c for c in before if c not in cards]
    if len(missing) == 1:
        return f"You played the {missing[0]}."
    if len(missing) > 1:
        return f"Cards that disappeared: {', '.join(missing)}."
    return f"No missing card detected. Before: {', '.join(before) or 'unknown'}. After: {result}."
