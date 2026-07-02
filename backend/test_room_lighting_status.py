#!/usr/bin/env python3
"""Tests for room_lighting_status tool."""

import os
import sys

import numpy as np

BACKEND_DIR = os.path.dirname(__file__)
TOOLS_DIR = os.path.join(BACKEND_DIR, "..", "tools")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(1, TOOLS_DIR)

import room_lighting_status


def make_test_image() -> np.ndarray:
    """Create a small synthetic BGR image."""
    return np.full((120, 160, 3), 180, dtype=np.uint8)


def test_no_image() -> None:
    result = room_lighting_status.main(None, {})
    assert isinstance(result, dict)
    assert result.get("audio", {}).get("type") == "error"


def test_general_reasoning_stage() -> None:
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return {"response": "LEVEL: normal\nMESSAGE: Lighting is normal."}

    room_lighting_status.copilot_llm_call = fake_call
    output = room_lighting_status.main(make_test_image(), {})

    assert isinstance(output, str)
    assert "normal" in output.lower()
    assert len(calls) == 1
    assert calls[0]["capability"] == "general_reasoning"
    assert calls[0]["goal"] == "Determine the ambient brightness of the room."


def test_low_light_warning() -> None:
    def fake_call(**_kwargs):
        return {"response": "LEVEL: dark\nMESSAGE: Room is dark. Turn on a light."}

    room_lighting_status.copilot_llm_call = fake_call
    output = room_lighting_status.main(make_test_image(), {})

    assert isinstance(output, dict)
    assert output.get("audio", {}).get("type") == "warning"
    assert "dark" in output.get("text", "").lower()


def test_significant_change_warning() -> None:
    def fake_call(**_kwargs):
        return {"response": "LEVEL: bright\nMESSAGE: Room is now bright."}

    room_lighting_status.copilot_llm_call = fake_call
    output = room_lighting_status.main(make_test_image(), {"previous_level": "dark"})

    assert isinstance(output, dict)
    assert output.get("audio", {}).get("type") == "warning"
    assert output.get("audio", {}).get("interrupt") is True
    assert "changed from dark to bright" in output.get("text", "").lower()


def main() -> int:
    tests = [
        test_no_image,
        test_general_reasoning_stage,
        test_low_light_warning,
        test_significant_change_warning,
    ]

    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"✓ {test.__name__}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
