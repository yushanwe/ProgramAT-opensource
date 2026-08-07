import argparse
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claude_progress import select_claude_progress_comment  # noqa: E402
from implementation_traceability import (  # noqa: E402
    extract_implementation_summary,
    sanitize_action_logs,
)


def _coerce_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    data = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(data)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {}


def _comment_to_namespace(comment: Dict[str, Any]) -> SimpleNamespace:
    user = comment.get("user")
    if isinstance(user, dict):
        user = SimpleNamespace(**user)
    return SimpleNamespace(**{**comment, "user": user})


def _extract_progress_comment(comments_payload: Dict[str, Any]) -> Dict[str, Any]:
    comments = comments_payload.get("comments") or comments_payload.get("items")
    if not isinstance(comments, list) or not comments:
        return {}
    try:
        comment = select_claude_progress_comment(
            [_comment_to_namespace(item) for item in comments if isinstance(item, dict)]
        )
    except LookupError:
        return {}
    body = getattr(comment, "body", "") or ""
    summary = extract_implementation_summary(body)
    return {
        "claude_comment_id": getattr(comment, "id", None),
        "claude_summary": body.strip() or None,
        "implementation_summary": summary,
    }


def _first_tool_path(pr_payload: Dict[str, Any]) -> Optional[str]:
    files = pr_payload.get("files") or pr_payload.get("items")
    if not isinstance(files, list):
        return None
    for item in files:
        path = item.get("path") if isinstance(item, dict) else None
        if isinstance(path, str) and path.startswith("tools/") and path.endswith(".py"):
            return path
    return None


def build_implementation_context(
    *,
    repository: str,
    issue_number: int,
    branch: str,
    commit_sha: str,
    workflow_run_id: int,
    workflow_job_url: Optional[str],
    action_log_path: str,
    pr_number: Optional[int] = None,
    tool_name: Optional[str] = None,
    tool_path: Optional[str] = None,
    comments_payload: Optional[Dict[str, Any]] = None,
    pr_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    progress = _extract_progress_comment(comments_payload or {})
    resolved_tool_path = tool_path or _first_tool_path(pr_payload or {})
    resolved_tool_name = tool_name or (
        Path(resolved_tool_path).stem if resolved_tool_path else None
    )
    context: Dict[str, Any] = {
        "repository": repository,
        "issue_number": issue_number,
        "pr_number": pr_number,
        "branch": branch,
        "commit_sha": commit_sha,
        "tool_name": resolved_tool_name,
        "tool_path": resolved_tool_path,
        "claude_comment_id": progress.get("claude_comment_id"),
        "workflow_run_id": workflow_run_id,
        "workflow_job_url": workflow_job_url,
        "claude_summary": progress.get("claude_summary"),
        "implementation_summary": progress.get("implementation_summary") or {},
        "action_log_path": action_log_path,
    }
    return context


def write_implementation_context(
    *,
    metadata_path: Path,
    action_log_path: Path,
    repository: str,
    issue_number: int,
    branch: str,
    commit_sha: str,
    workflow_run_id: int,
    workflow_job_url: Optional[str],
    raw_action_log: str,
    pr_number: Optional[int] = None,
    tool_name: Optional[str] = None,
    tool_path: Optional[str] = None,
    comments_payload: Optional[Dict[str, Any]] = None,
    pr_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_log = sanitize_action_logs(raw_action_log)
    action_log_path.write_text(cleaned_log + ("\n" if cleaned_log else ""), encoding="utf-8")
    try:
        action_log_ref = action_log_path.relative_to(metadata_path.parent.parent).as_posix()
    except ValueError:
        action_log_ref = action_log_path.as_posix()
    context = build_implementation_context(
        repository=repository,
        issue_number=issue_number,
        pr_number=pr_number,
        branch=branch,
        commit_sha=commit_sha,
        tool_name=tool_name,
        tool_path=tool_path,
        comments_payload=comments_payload,
        pr_payload=pr_payload,
        workflow_run_id=workflow_run_id,
        workflow_job_url=workflow_job_url,
        action_log_path=action_log_ref,
    )
    metadata_path.write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--action-log-path", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--pr-number")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--tool-name")
    parser.add_argument("--tool-path")
    parser.add_argument("--comments-json")
    parser.add_argument("--pr-files-json")
    parser.add_argument("--raw-log-file")
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--workflow-job-url")
    args = parser.parse_args()

    raw_action_log = ""
    if args.raw_log_file:
        raw_action_log = Path(args.raw_log_file).read_text(encoding="utf-8", errors="replace")

    write_implementation_context(
        metadata_path=Path(args.metadata_path),
        action_log_path=Path(args.action_log_path),
        repository=args.repository,
        issue_number=int(args.issue_number),
        pr_number=_coerce_int(args.pr_number),
        branch=args.branch,
        commit_sha=args.commit_sha,
        tool_name=args.tool_name,
        tool_path=args.tool_path,
        comments_payload=_load_json(args.comments_json),
        pr_payload=_load_json(args.pr_files_json),
        raw_action_log=raw_action_log,
        workflow_run_id=int(args.workflow_run_id),
        workflow_job_url=args.workflow_job_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
