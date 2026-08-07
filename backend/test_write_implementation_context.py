import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_implementation_context import write_implementation_context


class WriteImplementationContextTests(unittest.TestCase):
    def test_write_implementation_context_persists_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / ".programat" / "implementation_context.json"
            action_log_path = root / ".programat" / "claude_action_log.txt"

            context = write_implementation_context(
                metadata_path=metadata_path,
                action_log_path=action_log_path,
                repository="yushanwe/ProgramAT-opensource",
                issue_number=263,
                pr_number=271,
                branch="claude/issue-263-example",
                commit_sha="abc123",
                tool_name=None,
                tool_path=None,
                comments_payload={
                    "comments": [
                        {
                            "id": 5193800710,
                            "user": {"login": "claude[bot]"},
                            "body": "### Uber Car Identifier\n- [x] Review requirements\n- [ ] Validate generated tool\n\n### Current analysis\nImplementing now.\n\n### Implementation summary\n- Model: `gemini/gemini-3.1-flash-lite`\n- Validation: passed\n",
                        }
                    ]
                },
                pr_payload={"files": [{"path": "tools/uber_car_identifier.py"}]},
                raw_action_log="Requirement already satisfied\nProgress update\nCommitted branch\nAuthorization: Bearer secret-token\n",
                workflow_run_id=31020551473,
                workflow_job_url="https://github.com/example/repo/actions/runs/1/job/2",
            )

            self.assertTrue(metadata_path.exists())
            self.assertTrue(action_log_path.exists())
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["repository"], "yushanwe/ProgramAT-opensource")
            self.assertEqual(saved["pr_number"], 271)
            self.assertEqual(saved["tool_path"], "tools/uber_car_identifier.py")
            self.assertEqual(saved["claude_comment_id"], 5193800710)
            self.assertEqual(saved["implementation_summary"]["validation"], "passed")
            self.assertEqual(saved["action_log_path"], ".programat/claude_action_log.txt")
            self.assertEqual(context["tool_name"], "uber_car_identifier")

            log_text = action_log_path.read_text(encoding="utf-8")
            self.assertIn("Committed branch", log_text)
            self.assertNotIn("Requirement already satisfied", log_text)
            self.assertNotIn("secret-token", log_text)


if __name__ == "__main__":
    unittest.main()
