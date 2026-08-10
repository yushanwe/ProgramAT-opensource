import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)

CREATE_BRAINSTORM_PROMPT_PREFIX = (
    "You are helping a blind or low-vision user design a camera-based assistive tool. "
    "They have described the tool they want. Your job is to ask them ONE concise, "
    "open-ended question that helps them think more deeply about their idea — "
    "such as an edge case, an environmental condition, or how the tool should behave "
    "when something goes wrong. Do not ask about things already answered below or in previous rounds. "
    "Return only the question itself, nothing else.\n\n"
)

UPDATE_BRAINSTORM_PROMPT_EXTENSION = (
    "This is an update to an existing visual assistive tool. "
    "Use the repository, issue, PR, current tool, Claude summary, action log, and update request "
    "to understand the current implementation. Ask only clarification questions that would materially "
    "affect the requested change. Preserve existing behavior unless the user explicitly asks to change it. "
    "Do not ask generic questions such as \"What existing behavior should remain unchanged?\" "
    "Existing behavior should be assumed preserved by default. Only ask about preservation when there is "
    "a specific ambiguity or conflict in the requested update, and name the concrete behavior involved. "
    "Ground questions in the actual implementation context when possible, such as runtime input, streaming behavior, "
    "output timing, model choice, failure handling, or another specific existing behavior visible in the context. "
    "Do not redesign the entire tool or ask for information already available in the context.\n\n"
)

DEFAULT_CREATE_FALLBACK_QUESTION = (
    "Is there anything specific about how the tool should behave in difficult conditions, "
    "like low lighting or a cluttered background?"
)

DEFAULT_UPDATE_FALLBACK_QUESTION = (
    "Should this update change any specific streaming behavior, spoken output timing, or runtime input handling?"
)

_KEEP_LINE_PATTERNS = [
    re.compile(r"claude", re.IGNORECASE),
    re.compile(r"copilot", re.IGNORECASE),
    re.compile(r"\b(summary|summarized|implemented|implementation|validation|validated|error|fix|fixed)\b", re.IGNORECASE),
    re.compile(r"\b(inspect|inspected|modify|modified|edit|edited|update|updated|commit|pr|pull request)\b", re.IGNORECASE),
]
_DROP_LINE_PATTERNS = [
    re.compile(r"authorization:\s*\S+", re.IGNORECASE),
    re.compile(r"\b(token|secret|api[_ -]?key|password)\b", re.IGNORECASE),
    re.compile(r"\bnpm\b.*\binstall\b", re.IGNORECASE),
    re.compile(r"\bpip\b.*\binstall\b", re.IGNORECASE),
    re.compile(r"\byarn\b.*\binstall\b", re.IGNORECASE),
    re.compile(r"\bapt(-get)?\b", re.IGNORECASE),
    re.compile(r"download(ing)?", re.IGNORECASE),
    re.compile(r"^collecting\s", re.IGNORECASE),
    re.compile(r"^\s*at\s+.+:\d+:\d+", re.IGNORECASE),
]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def sanitize_action_log(action_log: str, max_chars: int = 8000) -> str:
    text = _stringify(action_log)
    if not text:
        return ""

    kept_lines: List[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in _DROP_LINE_PATTERNS):
            continue
        line = re.sub(r"(gh[pousr]_[A-Za-z0-9_]+)", "[REDACTED_TOKEN]", line)
        line = re.sub(r"(Bearer\s+)[A-Za-z0-9._-]+", r"\1[REDACTED]", line, flags=re.IGNORECASE)
        line = re.sub(r"([A-Za-z_]*TOKEN|API[_-]?KEY|SECRET)\s*=\s*\S+", r"\1=[REDACTED]", line, flags=re.IGNORECASE)
        if len(line) > 500 and not any(pattern.search(line) for pattern in _KEEP_LINE_PATTERNS):
            continue
        if line in seen:
            continue
        if any(pattern.search(line) for pattern in _KEEP_LINE_PATTERNS) or len(kept_lines) < 40:
            kept_lines.append(line)
            seen.add(line)

    if not kept_lines:
        return ""

    joined = "\n".join(kept_lines)
    if len(joined) <= max_chars:
        return joined

    prioritized = [line for line in kept_lines if any(pattern.search(line) for pattern in _KEEP_LINE_PATTERNS)]
    fallback = prioritized or kept_lines
    trimmed_lines: List[str] = []
    total = 0
    for line in fallback:
        extra = len(line) + (1 if trimmed_lines else 0)
        if total + extra > max_chars - len("\n[truncated]"):
            break
        trimmed_lines.append(line)
        total += extra
    return "\n".join(trimmed_lines) + "\n[truncated]"


def maybe_read_tool_source(tool_path: str, max_chars: int = 6000) -> str:
    path_text = _stringify(tool_path)
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    source = source.strip()
    if len(source) <= max_chars:
        return source
    return source[: max_chars - len("\n... [truncated]")] + "\n... [truncated]"


def build_update_brainstorm_context(
    repository: str,
    issue_number: Optional[int],
    pr_number: Optional[int],
    tool_name: str,
    tool_path: str,
    update_request: str,
    claude_summary: str = "",
    action_log: str = "",
    tool_source: str = "",
) -> Dict[str, Any]:
    return {
        "repository": _stringify(repository),
        "issue_number": issue_number,
        "pr_number": pr_number,
        "tool_name": _stringify(tool_name),
        "tool_path": _stringify(tool_path),
        "update_request": _stringify(update_request),
        "claude_summary": _stringify(claude_summary),
        "action_log": sanitize_action_log(action_log),
        "tool_source": _stringify(tool_source),
    }


def build_clarified_update_request(
    original_request: str,
    questions: List[str],
    answers: List[str],
    session_transcript: Optional[List[Dict[str, str]]] = None,
) -> str:
    lines = ["Original update request:", _stringify(original_request)]
    transcript = format_brainstorm_transcript(session_transcript)
    if transcript:
        lines.extend(["", "Full brainstorming session:", transcript])
        return "\n".join(lines).strip()

    pairs = [
        (q.strip(), (answers[index] if index < len(answers) else "").strip())
        for index, q in enumerate(questions or [])
        if _stringify(q)
    ]
    answered_pairs = [(q, a) for q, a in pairs if a]
    if answered_pairs:
        lines.extend(["", "Clarifications:"])
        for index, (question, answer) in enumerate(answered_pairs, start=1):
            lines.append(f"{index}. Q: {question}")
            lines.append(f"   A: {answer}")
    return "\n".join(lines).strip()


def format_brainstorm_transcript(
    session_transcript: Optional[List[Dict[str, str]]],
) -> str:
    """Format the complete dialogue while preserving who supplied each detail."""
    labels = {
        "assistant_question": "Q",
        "user_clarification": "User clarification question",
        "assistant_clarification": "Assistant clarification answer",
        "user_answer": "A",
    }
    lines: List[str] = []
    for event in session_transcript or []:
        text = _stringify((event or {}).get("text"))
        kind = _stringify((event or {}).get("kind"))
        if text and kind in labels:
            lines.append(f"{labels[kind]}: {text}")
    return "\n".join(lines)


def _format_brainstorm_history(brainstorm_context: Optional[List[Dict[str, str]]]) -> str:
    if not brainstorm_context:
        return ""
    pairs = []
    for qa_pair in brainstorm_context:
        question = _stringify((qa_pair or {}).get("question"))
        answer = _stringify((qa_pair or {}).get("answer"))
        if question and answer:
            pairs.append(f"Q: {question}\nA: {answer}")
    if not pairs:
        return ""
    return "\n\nPrevious brainstorming rounds:\n" + "\n\n".join(pairs)


def _build_create_prompt(context: Dict[str, Any], brainstorm_context: Optional[List[Dict[str, str]]]) -> str:
    video_summary = _stringify(context.get("video_summary"))
    video_section = f"\n\nVideo demonstration summary:\n{video_summary}" if video_summary else ""
    return (
        CREATE_BRAINSTORM_PROMPT_PREFIX
        + 
        f"Tool title: {_stringify(context.get('title'))}\n"
        f"Description: {_stringify(context.get('description'))}\n"
        f"Proposed solution: {_stringify(context.get('solution'))}\n"
        f"Example usage: {_stringify(context.get('example_usage'))}"
        f"{video_section}"
        f"{_format_brainstorm_history(brainstorm_context)}"
    )


def _build_update_prompt(context: Dict[str, Any], brainstorm_context: Optional[List[Dict[str, str]]]) -> str:
    sections = [
        CREATE_BRAINSTORM_PROMPT_PREFIX + UPDATE_BRAINSTORM_PROMPT_EXTENSION,
        f"Repository: {_stringify(context.get('repository'))}",
        f"Issue number: {context.get('issue_number') or ''}",
        f"PR number: {context.get('pr_number') or ''}",
        f"Current tool: {_stringify(context.get('tool_name'))}",
        f"Current tool path: {_stringify(context.get('tool_path'))}",
        f"User update request: {_stringify(context.get('update_request'))}",
    ]
    if _stringify(context.get("claude_summary")):
        sections.extend(["", "Unified Claude summary:", _stringify(context.get("claude_summary"))])
    if _stringify(context.get("action_log")):
        sections.extend(["", "GitHub Actions log (sanitized):", _stringify(context.get("action_log"))])
    if _stringify(context.get("tool_source")):
        sections.extend(["", "Current tool source excerpt:", _stringify(context.get("tool_source"))])
    history = _format_brainstorm_history(brainstorm_context)
    if history:
        sections.append(history)
    return "\n".join(sections)


async def generate_brainstorming_question(
    mode: str,
    context: Dict[str, Any],
    brainstorm_context: Optional[List[Dict[str, str]]] = None,
    *,
    llm_call: Callable[..., Any],
    text_extractor: Callable[[Any], str],
    logger_instance: Optional[logging.Logger] = None,
) -> str:
    active_logger = logger_instance or logger
    prompt_builder = _build_update_prompt if mode == "update" else _build_create_prompt
    fallback = DEFAULT_UPDATE_FALLBACK_QUESTION if mode == "update" else DEFAULT_CREATE_FALLBACK_QUESTION
    prompt = prompt_builder(context or {}, brainstorm_context)
    try:
        response = await asyncio.to_thread(
            llm_call,
            messages=[{"role": "user", "content": prompt}],
        )
        question = _stringify(text_extractor(response))
        if question:
            return question
    except Exception:
        active_logger.warning("generate_brainstorming_question failed for mode=%s", mode, exc_info=True)
    return fallback
