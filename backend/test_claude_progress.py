import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claude_progress import (
    looks_like_claude_progress_comment,
    parse_claude_progress_comment,
    select_claude_progress_comment,
)


def make_comment(comment_id: int, login: str, body: str, *, created_at=None, updated_at=None):
    return SimpleNamespace(
        id=comment_id,
        user=SimpleNamespace(login=login),
        body=body,
        created_at=created_at,
        updated_at=updated_at,
    )


class ClaudeProgressTests(unittest.TestCase):
    def test_parse_comment_with_checklist(self):
        parsed = parse_claude_progress_comment("""### Progress

- [x] Read issue and repository CLAUDE.md
- [ ] Implement tools/example.py
""")
        self.assertEqual(parsed["status"], "available")
        self.assertEqual(parsed["steps"][0]["status"], "completed")
        self.assertEqual(parsed["steps"][1]["status"], "in_progress")

    def test_parse_comment_without_tasks_heading(self):
        parsed = parse_claude_progress_comment("""### Creating Uber Car Identifier

- [x] Read issue and repository CLAUDE.md
- [ ] Validate with backend/validate_generated_tools.py
""")
        self.assertTrue(parsed["body"].startswith("### Creating Uber Car Identifier"))
        self.assertEqual(parsed["steps"][1]["status"], "in_progress")

    def test_parse_prose_only_running_comment(self):
        parsed = parse_claude_progress_comment(
            "I am inspecting the existing tool patterns and implementing the new tool now."
        )
        self.assertEqual(parsed["status"], "available")
        self.assertEqual(parsed["steps"], [])

    def test_parse_error_comment(self):
        parsed = parse_claude_progress_comment(
            "Validation failed with an error while checking the generated tool."
        )
        self.assertEqual(parsed["status"], "failed")

    def test_parse_completed_summary_comment(self):
        parsed = parse_claude_progress_comment(
            "Implemented the tool, validated it, and opened PR #123."
        )
        self.assertEqual(parsed["status"], "available")
        self.assertIn("PR #123", parsed["body"])

    def test_parse_delimited_summary_and_expert_content(self):
        parsed = parse_claude_progress_comment(
            """<!-- USER_SUMMARY_START -->
25% Processing CLAUDE.md

I found the current runtime input flow.
Next I will validate the update.
<!-- USER_SUMMARY_END -->

<!-- EXPERT_DETAIL_START -->
### Progress

- [x] Read issue and repository CLAUDE.md
- [ ] Validate the generated tool

### Current analysis

Comparing the runtime input path.
<!-- EXPERT_DETAIL_END -->"""
        )
        self.assertEqual(parsed["summary_text"], "25% Processing CLAUDE.md\n\nI found the current runtime input flow.\nNext I will validate the update.")
        self.assertIn("### Progress", parsed["expert_markdown"])
        self.assertEqual(parsed["status_line"]["percent"], 25)
        self.assertEqual(parsed["steps"][1]["status"], "in_progress")

    def test_parse_without_delimiters_falls_back_to_full_body_summary(self):
        parsed = parse_claude_progress_comment(
            "25% Processing CLAUDE.md\n\nInspecting the existing implementation."
        )
        self.assertEqual(parsed["summary_text"], "25% Processing CLAUDE.md\n\nInspecting the existing implementation.")
        self.assertEqual(parsed["expert_markdown"], "")

    def test_parse_uses_prefix_as_summary_when_only_expert_marker_exists(self):
        parsed = parse_claude_progress_comment(
            """25% Processing CLAUDE.md

I found the runtime input path.

<!-- EXPERT_DETAIL_START -->
### Current analysis
Comparing the runtime path now.
<!-- EXPERT_DETAIL_END -->"""
        )
        self.assertEqual(parsed["summary_text"], "25% Processing CLAUDE.md\n\nI found the runtime input path.")
        self.assertEqual(parsed["expert_markdown"], "### Current analysis\nComparing the runtime path now.")

    def test_parse_splits_at_progress_heading_without_markers(self):
        parsed = parse_claude_progress_comment(
            """50% Implementing runtime input

I found that the current streaming handler does not preserve the user criteria.
I'm updating the state handoff now.
Next I will validate both streaming and Take Photo.

### Progress
- [x] Inspected existing runtime input flow
- [ ] Update streaming handler

### Current analysis
Tracing the state handoff.
"""
        )
        self.assertEqual(
            parsed["summary_text"],
            "50% Implementing runtime input\n\nI found that the current streaming handler does not preserve the user criteria.\nI'm updating the state handoff now.\nNext I will validate both streaming and Take Photo."
        )
        self.assertTrue(parsed["expert_markdown"].startswith("### Progress"))
        self.assertIn("### Current analysis", parsed["expert_markdown"])
        self.assertEqual(parsed["steps"][1]["status"], "in_progress")

    def test_create_issue_comment_lookup(self):
        comments = [
            make_comment(1, "someone", "regular user comment"),
            make_comment(2, "claude[bot]", "I am implementing this now."),
        ]
        selected = select_claude_progress_comment(comments)
        self.assertEqual(selected.id, 2)

    def test_update_pr_conversation_lookup(self):
        comments = [
            make_comment(10, "github-actions[bot]", "Implemented the update on the PR branch."),
            make_comment(11, "claude[bot]", "Fixed the PR feedback and opened PR #45."),
        ]
        selected = select_claude_progress_comment(comments)
        self.assertEqual(selected.id, 11)

    def test_comment_identity_does_not_require_checklist(self):
        comment = make_comment(22, "claude[bot]", "Working through the update request now.")
        self.assertTrue(looks_like_claude_progress_comment(comment))

    def test_update_boundary_ignores_older_claude_comments(self):
        comments = [
            make_comment(
                30,
                "claude[bot]",
                "Older Claude update",
                created_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 8, 1, 10, 5, tzinfo=timezone.utc),
            ),
            make_comment(
                31,
                "someone",
                "User update request",
                created_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
            ),
        ]
        with self.assertRaises(LookupError):
            select_claude_progress_comment(
                comments,
                after_comment_id=31,
                after_timestamp="2026-08-07T12:00:00+00:00",
            )

    def test_update_boundary_selects_newer_claude_comment(self):
        comments = [
            make_comment(
                40,
                "claude[bot]",
                "Older Claude update",
                created_at=datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 8, 6, 10, 5, tzinfo=timezone.utc),
            ),
            make_comment(
                41,
                "someone",
                "User update request",
                created_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
            ),
            make_comment(
                42,
                "claude[bot]",
                "New Claude progress comment",
                created_at=datetime(2026, 8, 7, 12, 1, tzinfo=timezone.utc),
                updated_at=datetime(2026, 8, 7, 12, 2, tzinfo=timezone.utc),
            ),
        ]
        selected = select_claude_progress_comment(
            comments,
            after_comment_id=41,
            after_timestamp="2026-08-07T12:00:00+00:00",
        )
        self.assertEqual(selected.id, 42)


if __name__ == "__main__":
    unittest.main()
