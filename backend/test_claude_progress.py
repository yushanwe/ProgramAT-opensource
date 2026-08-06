import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claude_progress import (
    looks_like_claude_progress_comment,
    parse_claude_progress_comment,
    select_claude_progress_comment,
)


def make_comment(comment_id: int, login: str, body: str):
    return SimpleNamespace(id=comment_id, user=SimpleNamespace(login=login), body=body)


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


if __name__ == "__main__":
    unittest.main()
