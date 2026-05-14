"""
Test script for currency_bill_identifier.py
"""

import os
import sys
import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import currency_bill_identifier as tool


def test_extract_denomination_from_text():
    print('Testing extract_denomination_from_text()...')

    test_cases = [
        ('20 DOLLARS FEDERAL RESERVE NOTE', 20),
        ('$50 UNITED STATES OF AMERICA', 50),
        ('ONE HUNDRED DOLLARS', 100),
        ('This is not currency text', None),
        ('TEN TEN random text', 10),  # strong repeated signal without currency marker
    ]

    for text, expected in test_cases:
        result = tool.extract_denomination_from_text(text)
        status = '✓' if result == expected else '✗'
        print(f'  {status} "{text}" -> {result} (expected {expected})')
        assert result == expected, f'Expected {expected}, got {result}'

    print()


def test_format_denomination_audio():
    print('Testing format_denomination_audio()...')
    assert tool.format_denomination_audio(1) == '1 dollar'
    assert tool.format_denomination_audio(20) == '20 dollars'
    print('  ✓ Audio formatting works\n')


def test_main_invalid_image():
    print('Testing main() invalid image handling...')
    result = tool.main(None, {})
    print(f'  Result: {result}')
    assert result == 'No camera image available'
    print()


def test_main_tracking_behavior():
    print('Testing main() tracking behavior...')

    original_identify = tool.identify_currency_bill

    try:
        # Use monkeypatch-style replacement to avoid external OCR calls
        tool.identify_currency_bill = lambda image, api_key=None: {'denomination': 20, 'detected_text': 'TWENTY DOLLARS'}

        image = np.ones((100, 200, 3), dtype=np.uint8)

        first = tool.main(image, {'track_mode': True})
        second = tool.main(image, {'track_mode': True})

        print(f'  First response: {first}')
        print(f'  Second response: {second!r}')

        assert isinstance(first, dict)
        assert first.get('text') == '20 dollars'
        assert second == ''

        # Change denomination -> should announce again
        tool.identify_currency_bill = lambda image, api_key=None: {'denomination': 10, 'detected_text': 'TEN DOLLARS'}
        third = tool.main(image, {'track_mode': True})
        print(f'  Third response: {third}')
        assert isinstance(third, dict)
        assert third.get('text') == '10 dollars'

    finally:
        tool.identify_currency_bill = original_identify
        tool.reset_tracking()

    print()


def run_all_tests():
    print('=' * 60)
    print('CURRENCY BILL IDENTIFIER TEST SUITE')
    print('=' * 60)
    print()

    test_extract_denomination_from_text()
    test_format_denomination_audio()
    test_main_invalid_image()
    test_main_tracking_behavior()

    print('=' * 60)
    print('All tests completed!')
    print('=' * 60)


if __name__ == '__main__':
    run_all_tests()
