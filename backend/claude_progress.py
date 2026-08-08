from datetime import datetime
import re
from typing import Optional


CHECKLIST_LINE_RE = re.compile(r'^\s*[-*]\s+\[(?P<checked>[ xX])\]\s+(?P<label>.+?)\s*$', re.MULTILINE)
FAILURE_PATTERNS = (
    re.compile(r'\b(failed|failure|error|exception|traceback|unable to|could not|blocked)\b', re.IGNORECASE),
    re.compile(r'(^|\n)\s*status\s*:\s*failed\b', re.IGNORECASE),
)
CLAUDE_PROGRESS_TITLE_RE = re.compile(r'^\s{0,3}#{1,6}\s+(.+?)\s*$')
CLAUDE_PROGRESS_COMMENT_USERNAMES = {
    'claude[bot]',
    'claude',
    'github-actions[bot]',
}
CLAUDE_PROGRESS_MARKERS = (
    'claude',
    'working',
    'implementing',
    'fixing',
    'implemented',
    'validate_generated_tools.py',
    'fixes #',
    'pull request',
    'pr #',
)


def normalize_progress_label(raw_label: str) -> str:
    normalized = re.sub(r'`([^`]+)`', r'\1', raw_label).strip()
    lowered = normalized.lower()
    replacements = [
        (r'^(read|review) issue and repository claude\.md$', 'Reading your tool requirements'),
        (r'^(inspect|review) existing tool patterns$', 'Planning the implementation'),
        (r'^implement\b.+$', 'Building the tool'),
        (r'^(validate|run validation|check)\b.+$', 'Checking the generated tool'),
        (r'^(commit and push|save changes|save the completed tool)$', 'Saving the completed tool'),
        (r'^(provide|share)\s+pr link$', 'Sharing the pull request'),
    ]
    for pattern, replacement in replacements:
        if re.match(pattern, lowered):
            return replacement
    normalized = re.sub(r'\b[a-z0-9_./-]+\.(py|ts|tsx|js)\b', 'the tool', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+', ' ', normalized).strip(' .')
    return normalized[:1].upper() + normalized[1:] if normalized else raw_label.strip()


def parse_claude_progress_comment(body: str, *, issue_number: Optional[int] = None,
                                  comment_id: Optional[int] = None,
                                  updated_at: Optional[str] = None) -> dict:
    lines = body.splitlines()
    title = None
    steps = []
    first_pending_index = None

    for line in lines:
        if title is None:
            title_match = CLAUDE_PROGRESS_TITLE_RE.match(line)
            if title_match:
                title = title_match.group(1).strip()
        match = CHECKLIST_LINE_RE.match(line)
        if not match:
            continue
        raw_label = match.group('label').strip()
        checked = match.group('checked').lower() == 'x'
        steps.append({'raw_label': raw_label, 'completed': checked})
        if not checked and first_pending_index is None:
            first_pending_index = len(steps) - 1

    has_failure_text = any(pattern.search(body) for pattern in FAILURE_PATTERNS)
    if has_failure_text:
        overall_status = 'failed'
    elif steps and all(step['completed'] for step in steps):
        overall_status = 'completed'
    else:
        overall_status = 'available'

    normalized_steps = []
    for index, step in enumerate(steps, start=1):
        if has_failure_text and not step['completed'] and first_pending_index == index - 1:
            step_status = 'failed'
        elif step['completed']:
            step_status = 'completed'
        elif first_pending_index == index - 1:
            step_status = 'in_progress'
        else:
            step_status = 'pending'
        normalized_steps.append({
            'id': f'step_{index}',
            'label': normalize_progress_label(step['raw_label']),
            'raw_label': step['raw_label'],
            'status': step_status,
        })

    return {
        'status': overall_status,
        'title': title,
        'issue_number': issue_number,
        'comment_id': comment_id,
        'body': body,
        'steps': normalized_steps,
        'updated_at': updated_at,
    }


def looks_like_claude_progress_comment(comment) -> bool:
    user = getattr(getattr(comment, 'user', None), 'login', '') or ''
    body = getattr(comment, 'body', '') or ''
    if user.lower() not in CLAUDE_PROGRESS_COMMENT_USERNAMES:
        return False
    lowered = body.lower()
    return bool(CHECKLIST_LINE_RE.search(body)) or any(marker in lowered for marker in CLAUDE_PROGRESS_MARKERS)


def _parse_comment_timestamp(comment) -> Optional[datetime]:
    for attr in ("updated_at", "created_at"):
        value = getattr(comment, attr, None)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def _is_after_boundary(comment, *, after_comment_id: Optional[int] = None,
                       after_timestamp: Optional[str] = None) -> bool:
    if after_comment_id is None and not after_timestamp:
        return True

    comment_id = getattr(comment, 'id', None)
    if after_comment_id is not None and comment_id is not None and comment_id <= after_comment_id:
        return False

    if after_timestamp:
        try:
            boundary_time = datetime.fromisoformat(after_timestamp.replace("Z", "+00:00"))
        except ValueError:
            boundary_time = None
        comment_time = _parse_comment_timestamp(comment)
        if boundary_time is not None and comment_time is not None and comment_time < boundary_time:
            return False

    return True


def select_claude_progress_comment(comments, comment_id: Optional[int] = None,
                                   after_comment_id: Optional[int] = None,
                                   after_timestamp: Optional[str] = None):
    if comment_id is not None:
        for comment in comments:
            if getattr(comment, 'id', None) == comment_id:
                return comment
        raise LookupError('Claude progress comment not found')
    for comment in reversed(list(comments)):
        if looks_like_claude_progress_comment(comment) and _is_after_boundary(
            comment,
            after_comment_id=after_comment_id,
            after_timestamp=after_timestamp,
        ):
            return comment
    raise LookupError('Claude progress comment not found')
