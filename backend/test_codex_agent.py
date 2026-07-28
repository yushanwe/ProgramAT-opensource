import asyncio
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codex_agent


class TestCodexAgent(unittest.TestCase):
    def test_prompt_uses_repository_instructions_and_preserves_update(self):
        prompt = codex_agent.build_prompt(
            42,
            "Indoor exit finder",
            "## Task\nFind the most likely indoor exit.",
            "Use the stronger model only when uncertain.",
        )

        self.assertIn("Read and follow .github/copilot-instructions.md.", prompt)
        self.assertIn("Find the most likely indoor exit.", prompt)
        self.assertIn("Use the stronger model only when uncertain.", prompt)
        self.assertIn("Do not create branches, commits, pushes", prompt)

    def test_slug_is_safe_and_bounded(self):
        slug = codex_agent._slug("Indoor Exit Finder! " * 10)
        self.assertRegex(slug, r"^[a-z0-9-]+$")
        self.assertLessEqual(len(slug), 40)

    def test_publish_url_uses_configured_issue_repository(self):
        self.assertEqual(
            codex_agent._github_repo_url("yushanwe/ProgramAT-opensource"),
            "https://github.com/yushanwe/ProgramAT-opensource.git",
        )
        with self.assertRaises(ValueError):
            codex_agent._github_repo_url("https://github.com/wrong/repo")

    def test_cancel_terminates_active_process(self):
        process = Mock()
        process.returncode = None
        process.terminate = Mock()
        process.wait = AsyncMock(return_value=0)
        run = codex_agent.CodexRun(
            "codex-issue-7-test",
            7,
            "codex/issue-7-test",
            codex_agent.Path("/tmp/programat-codex-test"),
            process=process,
        )
        codex_agent.active_runs[7] = run
        try:
            self.assertTrue(asyncio.run(codex_agent.cancel_run(7)))
            self.assertTrue(run.cancelled)
            process.terminate.assert_called_once()
        finally:
            codex_agent.active_runs.clear()

    def test_remote_sha_requires_the_exact_branch(self):
        output = "abc123\trefs/heads/codex/issue-42\n"
        self.assertEqual(
            codex_agent._remote_sha(output, "refs/heads/codex/issue-42"),
            "abc123",
        )
        with self.assertRaises(RuntimeError):
            codex_agent._remote_sha(output, "refs/heads/main")

    def test_git_name_only_path_has_no_status_prefix(self):
        self.assertEqual(
            codex_agent._normalize_git_paths("tools/card_identifier.py\n"),
            ["tools/card_identifier.py"],
        )

    def test_modified_tool_proceeds_through_validation_commit_and_push(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote.git"
            worktree = root / "worktree"

            def git(*args, cwd=root):
                return subprocess.run(
                    ("git", *args),
                    cwd=cwd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()

            git("init", "--bare", str(remote))
            git("init", str(worktree))
            git("config", "user.name", "Codex Test", cwd=worktree)
            git("config", "user.email", "codex@example.test", cwd=worktree)
            git("config", "commit.gpgsign", "false", cwd=worktree)
            git("checkout", "-b", "codex/issue-199", cwd=worktree)
            tool = worktree / "tools" / "card_identifier.py"
            tool.parent.mkdir()
            tool.write_text("VALUE = 1\n", encoding="utf-8")
            git("add", "tools/card_identifier.py", cwd=worktree)
            git("commit", "-m", "initial", cwd=worktree)
            git(
                "push", str(remote), "HEAD:refs/heads/codex/issue-199",
                cwd=worktree,
            )
            remote_before = git(
                "rev-parse", "refs/heads/codex/issue-199", cwd=remote
            )

            tool.write_text("VALUE = 2\n", encoding="utf-8")
            status = git("status", "--short", cwd=worktree)
            changed = codex_agent._normalize_git_paths(
                git(
                    "diff", "--name-only", "--diff-filter=ACMR",
                    "HEAD", "--", "tools/", cwd=worktree,
                )
            )
            python_tools = [
                path for path in changed
                if path.startswith("tools/") and path.endswith(".py")
            ]

            self.assertEqual(status, "M tools/card_identifier.py")
            self.assertEqual(changed, ["tools/card_identifier.py"])
            self.assertEqual(python_tools, ["tools/card_identifier.py"])
            subprocess.run(
                (sys.executable, "-m", "py_compile", *python_tools),
                cwd=worktree,
                check=True,
            )
            git("add", "--", *python_tools, cwd=worktree)
            git("commit", "-m", "update card identifier", cwd=worktree)
            commit_sha = git("rev-parse", "HEAD", cwd=worktree)
            git(
                "push", str(remote), "HEAD:refs/heads/codex/issue-199",
                cwd=worktree,
            )
            remote_after = git(
                "rev-parse", "refs/heads/codex/issue-199", cwd=remote
            )

            self.assertNotEqual(remote_before, remote_after)
            self.assertEqual(remote_after, commit_sha)


if __name__ == "__main__":
    unittest.main()
