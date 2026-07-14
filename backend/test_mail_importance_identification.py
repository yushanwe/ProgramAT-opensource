#!/usr/bin/env python3
"""Focused tests for the mail importance take-photo tool."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import mail_importance_identification


class TestMailImportanceIdentification(unittest.TestCase):
    def test_main_delegates_to_take_photo_baseline_helper(self):
        mock_image = object()

        with patch.object(
            mail_importance_identification,
            'call_take_photo_baseline_vlm',
            return_value='Important: utility bill.',
        ) as mock_call:
            result = mail_importance_identification.main(mock_image, {})

        self.assertEqual(result, 'Important: utility bill.')
        mock_call.assert_called_once_with(
            image=mock_image,
            prompt=mail_importance_identification.TOOL_PROMPT,
        )


if __name__ == '__main__':
    unittest.main()
