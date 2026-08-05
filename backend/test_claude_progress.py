import sys
from unittest.mock import Mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.modules.setdefault('cv2', Mock())
sys.modules.setdefault('numpy', Mock())
sys.modules.setdefault('PIL', Mock())
websockets_mock = Mock()
sys.modules.setdefault('websockets', websockets_mock)
sys.modules.setdefault('websockets.exceptions', Mock(ConnectionClosed=Exception))

import stream_server


def test_parse_claude_progress_running_states():
    body = """# Creating Uber Car Identifier

Tasks

- [x] Read issue and repository CLAUDE.md
- [x] Inspect existing tool patterns
- [ ] Implement tools/uber_vehicle_identification.py
- [ ] Validate with backend/validate_generated_tools.py
"""

    parsed = stream_server.parse_claude_progress_comment(
        body,
        issue_number=263,
        comment_id=5193800710,
        updated_at="2026-08-05T12:00:00Z",
    )

    assert parsed["status"] == "running"
    assert parsed["title"] == "Creating Uber Car Identifier"
    assert [step["status"] for step in parsed["steps"]] == [
        "completed",
        "completed",
        "in_progress",
        "pending",
    ]
    assert parsed["steps"][0]["label"] == "Reading your tool requirements"
    assert parsed["steps"][2]["label"] == "Building the tool"
    assert parsed["steps"][2]["raw_label"] == "Implement tools/uber_vehicle_identification.py"


def test_parse_claude_progress_failed_detection():
    body = """# Creating Uber Car Identifier

Tasks

- [x] Read issue and repository CLAUDE.md
- [ ] Implement tools/uber_vehicle_identification.py

Failed with validation error while updating the generated tool.
"""

    parsed = stream_server.parse_claude_progress_comment(body)

    assert parsed["status"] == "failed"
    assert parsed["steps"][0]["status"] == "completed"
    assert parsed["steps"][1]["status"] == "failed"


def test_parse_claude_progress_completed():
    body = """Tasks

- [x] Read issue and repository CLAUDE.md
- [x] Commit and push
"""

    parsed = stream_server.parse_claude_progress_comment(body)

    assert parsed["status"] == "completed"
    assert [step["status"] for step in parsed["steps"]] == ["completed", "completed"]


def test_looks_like_claude_progress_comment_requires_bot_and_checklist():
    comment = Mock()
    comment.user.login = "claude[bot]"
    comment.body = "- [x] Read issue and repository CLAUDE.md"
    assert stream_server._looks_like_claude_progress_comment(comment) is True

    comment.user.login = "someone-else"
    assert stream_server._looks_like_claude_progress_comment(comment) is False
