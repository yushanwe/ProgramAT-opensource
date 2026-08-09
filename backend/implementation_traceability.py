import json
import logging
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import copilot_db
from claude_progress import parse_claude_progress_comment

logger = logging.getLogger(__name__)

TRACEABILITY_MAX_LOG_CHARS = 40000
TRACEABILITY_KEEP_LINE_PATTERNS = (
    r'progress',
    r'current analysis',
    r'implementation decisions',
    r'recent work',
    r'next step',
    r'implementation summary',
    r'validate_generated_tools',
    r'validation',
    r'error',
    r'failed',
    r'fix',
    r'commit',
    r'push',
    r'opened pr',
    r'branch',
    r'tools?/',
)
TRACEABILITY_DROP_LINE_PATTERNS = (
    r'^\s*$',
    r'collecting ',
    r'installing collected packages',
    r'requirement already satisfied',
    r'looking in indexes',
    r'warning: running pip',
    r'added \d+ packages?',
    r'found \d+ vulnerabilities',
    r'gh api ',
    r'github api request',
    r'x-ratelimit',
    r'accept: application/vnd\.github',
    r'user-agent: pygithub',
)
SECRET_PATTERNS = (
    re.compile(r'gh[pousr]_[A-Za-z0-9_]+'),
    re.compile(r'github_pat_[A-Za-z0-9_]+'),
    re.compile(r'(?i)(token|secret|password|authorization|bearer)\s*[:=]\s*\S+'),
)
CLAUDE_WORKFLOW_NAME = 'Claude VAT Implementation'
CLAUDE_WORKFLOW_JOB_HINT = 'claude-vat'
IMPLEMENTATION_CONTEXT_PATH = '.programat/implementation_context.json'
DEFAULT_ACTION_LOG_PATH = '.programat/claude_action_log.txt'


def _get_conn():
    return copilot_db.get_connection()


def init_traceability_table() -> None:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS implementation_traceability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            issue_number INTEGER,
            pr_number INTEGER,
            comment_id INTEGER,
            tool_name TEXT,
            tool_path TEXT,
            commit_sha TEXT,
            workflow_run_id INTEGER,
            workflow_job_id INTEGER,
            implementation_summary_json TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_traceability_issue ON implementation_traceability(issue_number)'
    )
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_traceability_pr ON implementation_traceability(pr_number)'
    )
    conn.commit()
    conn.close()


def _slugify_tool_name(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', (value or '').lower()).strip('_')
    return slug or 'generated_tool'


def infer_tool_path(tool_name: Optional[str], existing_path: Optional[str] = None) -> Optional[str]:
    if existing_path:
        return existing_path
    if not tool_name:
        return None
    return f"tools/{_slugify_tool_name(tool_name)}.py"


def _decode_json(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _encode_json(value: Optional[Dict[str, Any]]) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def upsert_traceability_record(
    *,
    mode: str,
    issue_number: Optional[int] = None,
    pr_number: Optional[int] = None,
    comment_id: Optional[int] = None,
    tool_name: Optional[str] = None,
    tool_path: Optional[str] = None,
    commit_sha: Optional[str] = None,
    workflow_run_id: Optional[int] = None,
    workflow_job_id: Optional[int] = None,
    implementation_summary: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT * FROM implementation_traceability
        WHERE (pr_number = ? AND ? IS NOT NULL)
           OR (issue_number = ? AND ? IS NOT NULL)
           OR (comment_id = ? AND ? IS NOT NULL)
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        ''',
        (pr_number, pr_number, issue_number, issue_number, comment_id, comment_id),
    )
    row = cursor.fetchone()
    existing = dict(row) if row else {}
    merged_summary = _decode_json(existing.get('implementation_summary_json'))
    merged_summary.update(implementation_summary or {})
    merged_metadata = _decode_json(existing.get('metadata_json'))
    merged_metadata.update(metadata or {})
    resolved_tool_name = tool_name or existing.get('tool_name')
    resolved_tool_path = infer_tool_path(tool_name or existing.get('tool_name'), tool_path or existing.get('tool_path'))

    values = {
        'mode': mode or existing.get('mode') or 'create',
        'issue_number': issue_number if issue_number is not None else existing.get('issue_number'),
        'pr_number': pr_number if pr_number is not None else existing.get('pr_number'),
        'comment_id': comment_id if comment_id is not None else existing.get('comment_id'),
        'tool_name': resolved_tool_name,
        'tool_path': resolved_tool_path,
        'commit_sha': commit_sha or existing.get('commit_sha'),
        'workflow_run_id': workflow_run_id if workflow_run_id is not None else existing.get('workflow_run_id'),
        'workflow_job_id': workflow_job_id if workflow_job_id is not None else existing.get('workflow_job_id'),
        'implementation_summary_json': _encode_json(merged_summary),
        'metadata_json': _encode_json(merged_metadata),
    }

    if row:
        cursor.execute(
            '''
            UPDATE implementation_traceability
            SET mode = ?, issue_number = ?, pr_number = ?, comment_id = ?,
                tool_name = ?, tool_path = ?, commit_sha = ?, workflow_run_id = ?,
                workflow_job_id = ?, implementation_summary_json = ?, metadata_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''',
            (
                values['mode'], values['issue_number'], values['pr_number'], values['comment_id'],
                values['tool_name'], values['tool_path'], values['commit_sha'], values['workflow_run_id'],
                values['workflow_job_id'], values['implementation_summary_json'], values['metadata_json'],
                existing['id'],
            ),
        )
        record_id = existing['id']
    else:
        cursor.execute(
            '''
            INSERT INTO implementation_traceability (
                mode, issue_number, pr_number, comment_id, tool_name, tool_path,
                commit_sha, workflow_run_id, workflow_job_id,
                implementation_summary_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                values['mode'], values['issue_number'], values['pr_number'], values['comment_id'],
                values['tool_name'], values['tool_path'], values['commit_sha'], values['workflow_run_id'],
                values['workflow_job_id'], values['implementation_summary_json'], values['metadata_json'],
            ),
        )
        record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    result = get_traceability_record(record_id=record_id)
    return result or {}


def get_traceability_record(
    *,
    record_id: Optional[int] = None,
    issue_number: Optional[int] = None,
    pr_number: Optional[int] = None,
    comment_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    cursor = conn.cursor()
    if record_id is not None:
        cursor.execute('SELECT * FROM implementation_traceability WHERE id = ?', (record_id,))
    else:
        cursor.execute(
            '''
            SELECT * FROM implementation_traceability
            WHERE (pr_number = ? AND ? IS NOT NULL)
               OR (issue_number = ? AND ? IS NOT NULL)
               OR (comment_id = ? AND ? IS NOT NULL)
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            ''',
            (pr_number, pr_number, issue_number, issue_number, comment_id, comment_id),
        )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    result['implementation_summary'] = _decode_json(result.pop('implementation_summary_json', None))
    result['metadata'] = _decode_json(result.pop('metadata_json', None))
    return result


def extract_implementation_summary(comment_body: str) -> Dict[str, Any]:
    text = comment_body or ''
    parsed = parse_claude_progress_comment(text)
    summary: Dict[str, Any] = {}
    body = parsed.get('expert_markdown') or parsed.get('body') or text
    section_match = re.search(
        r'###\s*Implementation summary\s*(.*?)(?:\n###\s|\Z)',
        body,
        re.IGNORECASE | re.DOTALL,
    )
    source = section_match.group(1) if section_match else body
    field_patterns = {
        'model': r'(?:model(?: choice)?|model)\s*:\s*([^\n]+)',
        'strategy': r'(?:strategy|model strategy|frame strategy)\s*:\s*([^\n]+)',
        'execution_mode': r'(?:execution mode|mode)\s*:\s*([^\n]+)',
        'runtime_input': r'(?:runtime input|input)\s*:\s*([^\n]+)',
        'validation': r'(?:validation|validate(?:d)?)\s*:\s*([^\n]+)',
    }
    for key, pattern in field_patterns.items():
        match = re.search(pattern, source, re.IGNORECASE)
        if match:
            summary[key] = re.sub(r'[`*\[\]]', '', match.group(1)).strip(' .')
    return summary


def _github_api_json(url: str, token: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'ProgramAT-Traceability',
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode('utf-8'))


def _github_api_text(url: str, token: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'ProgramAT-Traceability',
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode('utf-8', errors='replace')


def _match_claude_workflow_run(run: Dict[str, Any]) -> bool:
    name = str(run.get('name') or '')
    path = str(run.get('path') or '')
    return CLAUDE_WORKFLOW_NAME.lower() in name.lower() or path.endswith('claude-assignment.yml')


def _pick_claude_job(jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for job in jobs:
        if CLAUDE_WORKFLOW_JOB_HINT in str(job.get('name') or '').lower():
            return job
    return jobs[0] if jobs else None


def sanitize_action_logs(raw_logs: str, max_chars: int = TRACEABILITY_MAX_LOG_CHARS) -> str:
    if not raw_logs:
        return ''
    kept: List[str] = []
    dropped: List[str] = []
    keep_re = re.compile('|'.join(TRACEABILITY_KEEP_LINE_PATTERNS), re.IGNORECASE)
    drop_re = re.compile('|'.join(TRACEABILITY_DROP_LINE_PATTERNS), re.IGNORECASE)
    for raw_line in raw_logs.splitlines():
        line = raw_line.strip('\r')
        if '***' in line and 'token' in line.lower():
            continue
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            line = '<redacted secret>'
        if drop_re.search(line):
            dropped.append(line)
            continue
        if keep_re.search(line):
            kept.append(line)
        else:
            dropped.append(line)
    combined = kept[:]
    if len('\n'.join(combined)) > max_chars:
        combined = combined[: max(10, len(combined) // 2)]
    text = '\n'.join(combined)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + '\n...[truncated]'
    return text.strip()


def resolve_claude_workflow_trace(
    repo_name: str,
    token: str,
    *,
    saved_run_id: Optional[int] = None,
    saved_job_id: Optional[int] = None,
    commit_sha: Optional[str] = None,
) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        'workflow_run_id': saved_run_id,
        'workflow_job_id': saved_job_id,
        'logs': '',
        'workflow_name': None,
        'job_name': None,
    }
    jobs: List[Dict[str, Any]] = []
    run: Optional[Dict[str, Any]] = None

    if saved_run_id is not None:
        run = _github_api_json(f'https://api.github.com/repos/{repo_name}/actions/runs/{saved_run_id}', token)
    elif commit_sha:
        sha_query = urllib.parse.quote(commit_sha)
        runs = _github_api_json(
            f'https://api.github.com/repos/{repo_name}/actions/runs?head_sha={sha_query}&per_page=20',
            token,
        )
        for candidate in runs.get('workflow_runs', []):
            if _match_claude_workflow_run(candidate):
                run = candidate
                break

    if not run:
        return trace

    trace['workflow_run_id'] = run.get('id')
    trace['workflow_name'] = run.get('name')
    jobs_payload = _github_api_json(
        f'https://api.github.com/repos/{repo_name}/actions/runs/{trace["workflow_run_id"]}/jobs?per_page=100',
        token,
    )
    jobs = jobs_payload.get('jobs', [])
    job = None
    if saved_job_id is not None:
        job = next((item for item in jobs if item.get('id') == saved_job_id), None)
    if job is None:
        job = _pick_claude_job(jobs)
    if job:
        trace['workflow_job_id'] = job.get('id')
        trace['job_name'] = job.get('name')
        logs_url = job.get('logs_url')
        if logs_url:
            trace['logs'] = sanitize_action_logs(_github_api_text(logs_url, token))
    return trace


def resolve_pr_tooling_snapshot(repo_name: str, token: str, pr_number: int, tool_path: Optional[str]) -> Dict[str, Any]:
    from github import Github

    gh = Github(token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))
    files = list(pr.get_files())
    tool_file = None
    for changed in files:
        if changed.filename.startswith('tools/'):
            if tool_path is None or changed.filename == tool_path:
                tool_file = changed
                break
    if tool_file is None:
        tool_file = next((changed for changed in files if changed.filename.startswith('tools/')), None)

    diff_parts = []
    for changed in files:
        patch = getattr(changed, 'patch', None)
        if patch:
            diff_parts.append(f'--- {changed.filename}\n{patch}')
    tool_source = ''
    resolved_tool_path = tool_path
    if tool_file is not None:
        resolved_tool_path = tool_file.filename
        contents = repo.get_contents(tool_file.filename, ref=pr.head.sha)
        tool_source = contents.decoded_content.decode('utf-8', errors='replace')

    return {
        'pr_number': pr.number,
        'commit_sha': pr.head.sha,
        'tool_path': resolved_tool_path,
        'tool_source': tool_source,
        'pr_diff': '\n\n'.join(diff_parts),
    }


def build_traceability_response(
    *,
    repo_name: str,
    token: str,
    record: Dict[str, Any],
    include_logs: bool = False,
) -> Dict[str, Any]:
    pr_number = record.get('pr_number')
    tool_path = record.get('tool_path')
    pr_snapshot: Dict[str, Any] = {}
    if pr_number:
        pr_snapshot = resolve_pr_tooling_snapshot(repo_name, token, int(pr_number), tool_path)
        tool_path = pr_snapshot.get('tool_path') or tool_path

    workflow = resolve_claude_workflow_trace(
        repo_name,
        token,
        saved_run_id=record.get('workflow_run_id'),
        saved_job_id=record.get('workflow_job_id'),
        commit_sha=pr_snapshot.get('commit_sha') or record.get('commit_sha'),
    )

    if workflow.get('workflow_run_id') or workflow.get('workflow_job_id'):
        upsert_traceability_record(
            mode=record.get('mode') or 'create',
            issue_number=record.get('issue_number'),
            pr_number=pr_snapshot.get('pr_number') or record.get('pr_number'),
            comment_id=record.get('comment_id'),
            tool_name=record.get('tool_name'),
            tool_path=tool_path,
            commit_sha=pr_snapshot.get('commit_sha') or record.get('commit_sha'),
            workflow_run_id=workflow.get('workflow_run_id'),
            workflow_job_id=workflow.get('workflow_job_id'),
            implementation_summary=record.get('implementation_summary') or {},
            metadata=record.get('metadata') or {},
        )

    return {
        'metadata': {
            'tool_name': record.get('tool_name'),
            'tool_path': tool_path,
            'pr_number': pr_snapshot.get('pr_number') or record.get('pr_number'),
            'commit_sha': pr_snapshot.get('commit_sha') or record.get('commit_sha'),
            'workflow_run_id': workflow.get('workflow_run_id') or record.get('workflow_run_id'),
            'workflow_job_id': workflow.get('workflow_job_id') or record.get('workflow_job_id'),
            'implementation_summary': record.get('implementation_summary') or {},
        },
        'pr_diff': pr_snapshot.get('pr_diff', ''),
        'tool_source': pr_snapshot.get('tool_source', ''),
        'implementation_summary': record.get('implementation_summary') or {},
        'action_logs': workflow.get('logs', '') if include_logs else '',
    }


def resolve_update_brainstorm_context(
    *,
    repo_name: str,
    token: str,
    issue_number: int,
    pr_number: Optional[int] = None,
) -> Dict[str, Any]:
    from github import Github

    gh = Github(token)
    repo = gh.get_repo(repo_name)
    issue = repo.get_issue(int(issue_number))

    resolved_pr = repo.get_pull(int(pr_number)) if pr_number else None
    if resolved_pr is None:
        pulls = repo.get_pulls(state='all', sort='updated', direction='desc')
        for candidate in pulls:
            text = f'{candidate.title}\n{candidate.body or ""}'
            if f'#{issue_number}' in text or f'issue {issue_number}' in text.lower():
                resolved_pr = candidate
                break
            if str(issue_number) in (candidate.head.ref or ''):
                resolved_pr = candidate
                break

    logger.info(
        "[update_context DEBUG] issue=#%s resolved_pr=%s",
        issue_number,
        f"#{resolved_pr.number} ({resolved_pr.head.ref})" if resolved_pr is not None else "NONE FOUND",
    )

    pr_metadata: Dict[str, Any] = {}
    saved_metadata: Dict[str, Any] = {}
    saved_action_log = ''
    saved_tool_source = ''
    pr_comments: List[str] = []
    issue_comments: List[str] = []

    if resolved_pr is not None:
        pr_metadata = {
            'pr_number': resolved_pr.number,
            'pr_title': resolved_pr.title,
            'pr_body': resolved_pr.body or '',
            'pr_head_branch': resolved_pr.head.ref,
            'pr_head_sha': resolved_pr.head.sha,
        }
        try:
            metadata_file = repo.get_contents(IMPLEMENTATION_CONTEXT_PATH, ref=resolved_pr.head.sha)
            saved_metadata = json.loads(metadata_file.decoded_content.decode('utf-8', errors='replace'))
            logger.info(
                "[update_context DEBUG] fetched %s at sha=%s: tool_path=%s keys=%s",
                IMPLEMENTATION_CONTEXT_PATH, resolved_pr.head.sha,
                saved_metadata.get('tool_path'), list(saved_metadata.keys()),
            )
        except Exception as e:
            saved_metadata = {}
            logger.warning(
                "[update_context DEBUG] FAILED to fetch %s at sha=%s: %s",
                IMPLEMENTATION_CONTEXT_PATH, resolved_pr.head.sha, e,
            )

        action_log_path = str(saved_metadata.get('action_log_path') or DEFAULT_ACTION_LOG_PATH).strip() or DEFAULT_ACTION_LOG_PATH
        try:
            action_log_file = repo.get_contents(action_log_path, ref=resolved_pr.head.sha)
            saved_action_log = sanitize_action_logs(action_log_file.decoded_content.decode('utf-8', errors='replace'))
        except Exception as e:
            saved_action_log = ''
            logger.warning("[update_context DEBUG] FAILED to fetch action log %s: %s", action_log_path, e)

        tool_path = saved_metadata.get('tool_path')
        if isinstance(tool_path, str) and tool_path:
            try:
                tool_file = repo.get_contents(tool_path, ref=resolved_pr.head.sha)
                saved_tool_source = tool_file.decoded_content.decode('utf-8', errors='replace')
                logger.info(
                    "[update_context DEBUG] fetched tool_source from %s: %d chars",
                    tool_path, len(saved_tool_source),
                )
            except Exception as e:
                saved_tool_source = ''
                logger.warning("[update_context DEBUG] FAILED to fetch tool_source %s: %s", tool_path, e)
        else:
            logger.warning(
                "[update_context DEBUG] no tool_path in saved_metadata — cannot fetch tool_source. saved_metadata=%s",
                saved_metadata,
            )

        try:
            pr_comments = [
                (comment.body or '').strip()
                for comment in resolved_pr.get_issue_comments()
                if (comment.body or '').strip()
            ]
        except Exception:
            pr_comments = []

    try:
        issue_comments = [
            (comment.body or '').strip()
            for comment in issue.get_comments()
            if (comment.body or '').strip()
        ]
    except Exception:
        issue_comments = []

    trace_record = get_traceability_record(issue_number=int(issue_number), pr_number=pr_metadata.get('pr_number'))
    fallback_traceability = {}
    if trace_record:
        try:
            fallback_traceability = build_traceability_response(
                repo_name=repo_name,
                token=token,
                record=trace_record,
                include_logs=True,
            )
        except Exception:
            fallback_traceability = {}

    metadata = saved_metadata or fallback_traceability.get('metadata') or {}
    implementation_summary = metadata.get('implementation_summary') or fallback_traceability.get('implementation_summary') or {}
    action_logs = saved_action_log or fallback_traceability.get('action_logs') or ''
    tool_source = saved_tool_source or fallback_traceability.get('tool_source') or ''

    return {
        'repository': repo_name,
        'issue_number': int(issue_number),
        'issue_title': issue.title,
        'issue_body': issue.body or '',
        'issue_comments': issue_comments,
        'pr_number': pr_metadata.get('pr_number') or pr_number,
        'pr_title': pr_metadata.get('pr_title') or '',
        'pr_body': pr_metadata.get('pr_body') or '',
        'pr_comments': pr_comments,
        'pr_head_branch': pr_metadata.get('pr_head_branch') or metadata.get('branch') or '',
        'pr_head_sha': pr_metadata.get('pr_head_sha') or metadata.get('commit_sha') or '',
        'tool_name': metadata.get('tool_name') or trace_record.get('tool_name') if trace_record else None,
        'tool_path': metadata.get('tool_path') or trace_record.get('tool_path') if trace_record else None,
        'tool_source': tool_source,
        'claude_summary': metadata.get('claude_summary') or '',
        'implementation_summary': implementation_summary,
        'implementation_metadata': metadata,
        'action_log': action_logs,
        'pr_diff': fallback_traceability.get('pr_diff') or '',
    }


init_traceability_table()
