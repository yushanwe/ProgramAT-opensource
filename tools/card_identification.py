import re
from collections import Counter

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

# Module-level state for sequential single-user use.
# This tool is designed for sequential frame processing; not intended for
# concurrent multi-user access (consistent with camera_aiming.py pattern).
_before_hand = None  # stores parsed card list from the before-photo invocation


def _parse_cards(description):
    """Return a list of card strings found in a VLM description."""
    return [match.lower() for match in _CARD_PATTERN.findall(description)]


def _find_missing_cards(before, after):
    """Return cards present in before but removed from after, handling duplicates."""
    before_counts = Counter(before)
    after_counts = Counter(after)
    missing = []
    for card, count in before_counts.items():
        diff = count - after_counts.get(card, 0)
        missing.extend([card] * max(diff, 0))
    return missing


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
    missing = _find_missing_cards(before, cards)
    if len(missing) == 1:
        return f"You played the {missing[0]}."
    if len(missing) > 1:
        return f"Cards that disappeared: {', '.join(missing)}."
    return f"No missing card detected. Before: {', '.join(before) or 'unknown'}. After: {result}."
