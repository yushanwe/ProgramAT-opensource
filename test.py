from pathlib import Path
import asyncio
import sys

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from brainstorming import (  # noqa: E402
    build_update_brainstorm_context,
    maybe_read_tool_source,
    UPDATE_BRAINSTORM_PROMPT_EXTENSION,
    CREATE_BRAINSTORM_PROMPT_PREFIX,
)
from litellm_utils import extract_text  # noqa: E402
from model_router import system_llm_call  # noqa: E402


REPOSITORY = "yushanwe/ProgramAT-opensource"
ISSUE_NUMBER = 267
PR_NUMBER = 268

TOOL_NAME = "uber_vehicle_identification"
TOOL_PATH = "tools/uber_vehicle_identification.py"

UPDATE_REQUEST = """
I want it to not only tell me the characteristic of the car but also tell me if there are any signs showing that it might be an uber car.
"""

CLAUDE_SUMMARY = """
Summary
Added tools/uber_vehicle_identification.py, a new executable lifecycle tool that:

Identifies the color, make, model, and license plate of the vehicle at the center of the scene using gemini/gemini-3.1-flash-lite (default vision model per CLAUDE.md).
Declares one TOOL_RUNTIME_INPUT field (expected_vehicle) so the user can enter a free-text description of the car they're expecting (e.g. "white Toyota Camry, plate ABC123" or "black Honda"). When provided, this is folded into the model prompt so the response states whether the identified vehicle matches or is a mismatch.
Falls back to plainly stating it can't identify a detail (rather than guessing) when the plate or other features are unreadable due to distance, angle, glare, or dirt — matching the brainstorming answer in the issue.
Implements both on_take_photo (one-shot) and the on_frame/on_stream_start/on_stream_stop streaming lifecycle using the latest_frame streaming context specified in the issue (single current-frame analysis, no multi-frame/temporal logic since the task depends only on what's visible now).
Uses an in-flight guard set before the first await in on_frame to avoid duplicate overlapping model calls, per repo convention.
Validated with backend/validate_generated_tools.py ../tools/uber_vehicle_identification.py — guardrails passed. No other files were modified.
"""

ACTION_LOG = """
Paste the GitHub Actions log here.
"""

ACTION_LOG_PATH = "./job-logs.txt"

QUESTION_COUNT = 3

PRINT_DEBUG = True


def load_action_log() -> str:
    if ACTION_LOG_PATH.strip():
        return Path(ACTION_LOG_PATH.strip()).read_text(encoding="utf-8")
    return ACTION_LOG


async def main() -> None:
    tool_source = maybe_read_tool_source(TOOL_PATH)
    action_log = load_action_log()
    context = build_update_brainstorm_context(
        repository=REPOSITORY,
        issue_number=ISSUE_NUMBER,
        pr_number=PR_NUMBER,
        tool_name=TOOL_NAME,
        tool_path=TOOL_PATH,
        update_request=UPDATE_REQUEST,
        claude_summary=CLAUDE_SUMMARY,
        action_log=action_log,
        tool_source=tool_source,
    )

    if PRINT_DEBUG:
        preview = {
            "repository": context["repository"],
            "issue_number": context["issue_number"],
            "pr_number": context["pr_number"],
            "tool_name": context["tool_name"],
            "tool_path": context["tool_path"],
            "update_request_preview": context["update_request"][:200],
            "claude_summary_preview": context["claude_summary"][:200],
            "action_log_preview": context["action_log"][:500],
            "tool_source_included": bool(context["tool_source"]),
        }
        print("Sanitized context preview:")
        for key, value in preview.items():
            print(f"- {key}: {value}")
        print()

    prompt = (
        CREATE_BRAINSTORM_PROMPT_PREFIX
        + UPDATE_BRAINSTORM_PROMPT_EXTENSION
        + f"Ask for the {max(1, int(QUESTION_COUNT))} most important clarification questions in one response. "
          "Number them 1, 2, 3. Keep each question concise and do not include any extra explanation.\n\n"
        + f"Repository: {context['repository']}\n"
        + f"Issue number: {context['issue_number'] or ''}\n"
        + f"PR number: {context['pr_number'] or ''}\n"
        + f"Current tool: {context['tool_name']}\n"
        + f"Current tool path: {context['tool_path']}\n"
        + f"User update request: {context['update_request']}\n"
    )
    if context["claude_summary"]:
        prompt += f"\nUnified Claude summary:\n{context['claude_summary']}\n"
    if context["action_log"]:
        prompt += f"\nGitHub Actions log (sanitized):\n{context['action_log']}\n"
    if context["tool_source"]:
        prompt += f"\nCurrent tool source excerpt:\n{context['tool_source']}\n"

    response = await asyncio.to_thread(
        system_llm_call,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_output = extract_text(response).strip()

    print("Update brainstorming questions:\n")
    print(raw_output)

    if PRINT_DEBUG:
        print("\nRaw model response:\n")
        print(raw_output)


if __name__ == "__main__":
    asyncio.run(main())
