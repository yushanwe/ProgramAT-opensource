"""
Test script for card_identification.py
Tests the card identification tool's before/after comparison logic.
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# backend/ must come before tools/ so that 'litellm_utils' resolves to the
# backend copy that exports call_take_photo_baseline_vlm.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'tools'))

import card_identification
from card_identification import _parse_cards, main


def _blank_image():
    """Return a minimal BGR image for testing."""
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestParseCards(unittest.TestCase):
    def test_simple_card(self):
        self.assertEqual(_parse_cards("ace of spades"), ["ace of spades"])

    def test_multiple_cards(self):
        result = _parse_cards("ace of spades, seven of hearts, three of diamonds")
        self.assertEqual(result, ["ace of spades", "seven of hearts", "three of diamonds"])

    def test_numeric_rank(self):
        self.assertEqual(_parse_cards("7 of clubs"), ["7 of clubs"])

    def test_no_cards(self):
        self.assertEqual(_parse_cards("No cards visible."), [])

    def test_case_insensitive(self):
        result = _parse_cards("Ace of Spades")
        self.assertEqual(result, ["ace of spades"])

    def test_partial_sentence(self):
        result = _parse_cards("I can see the jack of clubs in the image.")
        self.assertEqual(result, ["jack of clubs"])


class TestMain(unittest.TestCase):
    def setUp(self):
        """Reset module-level state before each test."""
        card_identification._before_hand = None

    def test_no_image_returns_error(self):
        result = main(None, {})
        self.assertIn("No image", result)

    def test_first_call_stores_before_hand(self):
        image = _blank_image()
        vlm_response = "ace of spades, seven of hearts, three of diamonds"
        with patch(
            "card_identification.call_take_photo_baseline_vlm",
            return_value=vlm_response,
        ):
            result = main(image, {})
        self.assertIn("before hand captured", result.lower())
        self.assertIsNotNone(card_identification._before_hand)
        self.assertEqual(
            card_identification._before_hand,
            ["ace of spades", "seven of hearts", "three of diamonds"],
        )

    def test_second_call_identifies_played_card(self):
        image = _blank_image()
        card_identification._before_hand = ["ace of spades", "seven of hearts", "three of diamonds"]
        after_response = "ace of spades, three of diamonds"
        with patch(
            "card_identification.call_take_photo_baseline_vlm",
            return_value=after_response,
        ):
            result = main(image, {})
        self.assertIn("seven of hearts", result)
        self.assertIsNone(card_identification._before_hand)

    def test_second_call_resets_state(self):
        image = _blank_image()
        card_identification._before_hand = ["ace of spades"]
        with patch(
            "card_identification.call_take_photo_baseline_vlm",
            return_value="No cards visible.",
        ):
            main(image, {})
        self.assertIsNone(card_identification._before_hand)

    def test_no_cards_in_before_photo(self):
        image = _blank_image()
        with patch(
            "card_identification.call_take_photo_baseline_vlm",
            return_value="No cards visible.",
        ):
            result = main(image, {})
        self.assertIn("No playing cards detected", result)
        self.assertEqual(card_identification._before_hand, [])

    def test_no_missing_card_detected(self):
        image = _blank_image()
        card_identification._before_hand = ["ace of spades", "seven of hearts"]
        with patch(
            "card_identification.call_take_photo_baseline_vlm",
            return_value="ace of spades, seven of hearts",
        ):
            result = main(image, {})
        self.assertIn("No missing card", result)

    def test_multiple_missing_cards(self):
        image = _blank_image()
        card_identification._before_hand = [
            "ace of spades", "seven of hearts", "three of diamonds"
        ]
        with patch(
            "card_identification.call_take_photo_baseline_vlm",
            return_value="ace of spades",
        ):
            result = main(image, {})
        self.assertIn("seven of hearts", result)
        self.assertIn("three of diamonds", result)


if __name__ == "__main__":
    unittest.main()
