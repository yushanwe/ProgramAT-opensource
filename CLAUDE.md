# ProgramAT

ProgramAT tools are Python modules in `tools/` that answer one user-facing visual question from camera input and return or emit short, speech-friendly results for blind and low-vision users. When Claude generates or edits a tool, the root task is usually to change only that tool file. Do not modify shared backend, frontend, transport, validator, or runtime code just to make one generated tool work.

## What Claude builds

Build one focused tool module under `tools/`. The backend runtime supplies decoded frames, isolated per-session state, cancellation, and result transport. A good tool chooses the simplest frame input and model strategy that satisfies the request, keeps its prompt tight, and returns or emits output that is understandable when spoken aloud. Treat accessibility as a product requirement: results should help without requiring the user to see the screen.

The current default contract for generated tools is the executable lifecycle runtime. Legacy declarative take-photo and hosted-video tools still exist for compatibility, but new generated work should follow the executable lifecycle unless the repository code you are editing already uses a legacy declarative format and the task specifically requires preserving it.

Reuse approved shared helpers and established repository patterns instead of rebuilding model or detector infrastructure inside each new tool. Finish when the implementation works, passes validation, and has focused tests; do not add unrelated refactors or extra documentation just to support one generated tool.

## Available Capabilities

### Functions and runtime APIs

Generated lifecycle tools may use the shared helpers and runtime APIs that the backend already injects and validates:

- `call_model(model_name, messages, images=None, metadata=None)`: approved LiteLLM-backed helper for text or vision model calls. Pass an explicit repository-supported model name every time.
- `call_openai_responses_model(model_name, prompt, image, reasoning_effort="medium", timeout=None)`: approved helper for one OpenAI Responses multimodal call when a tool needs that path explicitly.
- `extract_text(response)`: normalize a LiteLLM response into plain text before trimming or applying a fallback.
- `runtime.current_frame` or `runtime.get_latest_frame()`: access the newest available decoded frame.
- `runtime.get_recent_frames(count=None, seconds=None)`: access chronological recent frames when the task depends on motion, order, or recent change.
- `runtime.get_state(key, default=None)`, `runtime.set_state(key, value)`, `runtime.clear_state()`: store tool-owned session state. State is isolated by client, tool version, and session.
- `runtime.is_cancelled()`: stop work or suppress emits when the session has gone stale.
- `await runtime.emit(text, partial=False, final=False, replace=False, metadata=None)`: send a normal streaming result event from the tool.
- Local utility code inside the tool is allowed when it stays self-contained and validator-safe, for example `cv2`/`numpy` image preprocessing or repository-aligned detector usage with `ultralytics.YOLO` when the task truly benefits.

Do not call provider SDKs directly except through approved shared helpers already used by the backend contract. Do not invent new transport callbacks, event emitters, or network paths. If you are unsure whether an import or helper is supported, inspect `backend/validate_generated_tools.py` and a current passing tool before adding it.

### Model profiles

Select from repository-defined model profiles rather than inventing new profile names or directly wiring provider clients:

- `gemini-3.1-flash-lite`: the default general-purpose vision model; low latency, low cost, strong enough for most OCR, scene description, and visual QA.
- `gpt-5`: the strongest configured vision-and-reasoning option; use when the task truly needs higher accuracy or harder reasoning and can tolerate more latency and cost.
- `moondream`: a fast, cheaper vision option for simpler image questions when lower accuracy is acceptable.
- `yolo`: a local object detector for object presence or localization, not general scene reasoning.
- `gpt-4o-mini`: a lightweight text model for evaluation or aggregation, not the default vision model.

In lifecycle tools, pass the provider model string expected by the helper call, such as `"gemini/gemini-3.1-flash-lite-preview"` or `"gpt-5"`, and keep the choice aligned with the registry-backed intent above.

For object detection, use the shared `yolo` profile backed by `yolo11n.pt` when the requested target is already covered by the standard COCO classes. When the target is not in COCO and open-vocabulary detection is genuinely needed, follow the existing repository pattern used by current tools with `ultralytics.YOLO("yolov8s-world.pt")` plus the shared `yolo_model_cache`, and reuse existing helper logic or class definitions instead of copying the full COCO list or loading duplicate detectors per tool. If the task also needs semantic interpretation, relationships, navigation reasoning, or other higher-level judgment, use an appropriate VLM rather than assuming YOLO alone is enough.

### Strategies

Model strategy and frame strategy are separate choices.

- Single model: use one model call when the user wants one ordinary answer.
- Conditional cascade: start with a faster model and only escalate when the first result is empty, uncertain, insufficient, or obviously weak.
- Progressive execution: use only when the user explicitly wants multiple useful results over time, such as a quick answer followed by a stronger answer. In executable streaming tools, each emitted result must stand on its own and contain real text.

Do not assume streaming means progressive. Do not assume multi-frame input means multiple model stages. Choose each independently.

### Frame and multi-frame input

Use the current frame or latest frame when the answer depends on what is visible now: OCR, object identification, color, current layout, scene description, classification, or a held hand pose. Use several recent chronological frames only when the answer depends on motion, sequence, duration, before-and-after change, a gesture over time, or what just happened. Streaming does not automatically require multi-frame input.

## What A Tool May Define

New generated tools should usually follow this executable lifecycle shape:

```python
import asyncio

from litellm_utils import call_model, extract_text

TOOL_NAME = "concise_snake_case_name"  # required
TOOL_PROMPT = (
    "One concise task-specific instruction shared across Take Photo and Streaming."
)  # required

DEFAULT_MODEL = "gemini/gemini-3.1-flash-lite-preview"  # optional
FALLBACK_TEXT = "I am not sure from this view."  # optional but recommended
USE_RECENT_FRAMES = False  # optional explanatory constant if helpful


async def analyze_image(image):
    response = await asyncio.to_thread(
        call_model,
        DEFAULT_MODEL,
        [{"role": "user", "content": TOOL_PROMPT}],
        [image],
        {"timeout": 60, "num_retries": 0},
    )
    text = extract_text(response).strip()
    return text or FALLBACK_TEXT


async def on_take_photo(runtime, image, input_data):  # optional
    if image is None:
        return FALLBACK_TEXT
    return await analyze_image(image)


async def on_stream_start(runtime, input_data):  # optional
    runtime.set_state("in_flight", False)


async def on_frame(runtime, frame):  # optional
    if runtime.is_cancelled():
        return
    if runtime.get_state("in_flight"):
        return
    runtime.set_state("in_flight", True)  # claim before the first await
    try:
        text = await analyze_image(frame.image)
        if not runtime.is_cancelled():
            await runtime.emit(text, final=True)
    finally:
        runtime.set_state("in_flight", False)


async def on_stream_stop(runtime):  # optional
    runtime.clear_state()
```

Required definitions for lifecycle tools are `TOOL_NAME`, `TOOL_PROMPT`, and at least one supported entry point: `on_take_photo()` or `on_frame()`. `on_stream_start()` and `on_stream_stop()` are optional but recommended when streaming needs state or cleanup. Local helper functions are optional and should stay narrow and tool-specific.

Do not introduce deprecated or non-default interfaces in new tools such as `TOOL_POLICY`, `execute_tool_policy()`, `EXECUTION_MODE`, mode-specific prompt constants, or runtime-owned hosted-video configuration unless you are deliberately maintaining an existing legacy tool that already uses that format.

### Prompt rules

- Define one concise `TOOL_PROMPT` for the core task.
- Keep the same task semantics across Take Photo and Streaming.
- Decide whether the task is one direct visual operation or one fused model call with a short dependent sequence of operations.
- Use one direct instruction without steps for simple tasks such as identifying an object, reading visible text, reporting a color, classifying one visible item, or answering another question that can be completed as one visual operation.
- When the final answer depends on several visual or reasoning subproblems, place a short ordered sequence inside the same `TOOL_PROMPT`, such as: identify reliable exit evidence, choose the most likely reachable exit, determine its observable clock-face direction, and give one concise instruction toward it.
- Numbered steps are optional; Claude may use either a compact ordered paragraph or a short numbered sequence when that makes the prompt clearer.
- This decomposition stays inside one fused prompt and one model operation. It must not turn into runtime stages, planner pipelines, multiple specialist calls, or separate emits unless the chosen model strategy explicitly requires that behavior.
- Include only the task-relevant details: the user context when it matters, the visual evidence to inspect, a short dependent order of operations when useful, the expected concise speech-friendly output, any task-specific normalization, and an uncertainty fallback.
- Ask for only the final user-facing answer, not analysis, JSON, chain-of-thought, or the result of each internal step.
- Include a short uncertainty fallback for unreadable, unavailable, or insufficient evidence.
- Keep the wording natural when spoken by text-to-speech.

### Output rules

- Every returned or emitted user-facing result must contain useful non-empty text.
- Results must be concise, plain-language, independently understandable, and audio-friendly.
- Normalize helper output with `extract_text(...).strip()` and then apply a fallback.
- Do not return or emit raw JSON, detection dictionaries, confidence metadata, abbreviations, debug output, stack traces, model metadata, `None`, empty strings, or internal structures.
- A weak, empty, or uncertain model response does not prove the target is absent; express uncertainty instead.
- Take Photo and Streaming may choose different frame or model strategies, but they should still implement the same core task rather than two unrelated behaviors.

## What Claude Must Not Do

Keep these restrictions in one place and follow them strictly:

- Do not modify backend transport, WebSocket handling, frontend code, `ToolRunner`, `stream_server.py`, shared runtime code, or validator behavior just to support one tool. Those changes easily break unrelated tool execution or delivery.
- Do not access API keys or environment variables directly. The validator rejects environment access, and shared helpers already own credential resolution.
- Do not call provider SDKs directly or construct arbitrary network requests. Use the approved shared model helpers instead, or local processing that passes validation.
- Do not use subprocesses, multiprocessing, raw filesystem reads or writes, unsafe dynamic imports, or unsupported networking imports. These patterns are validator failures and can break sandbox assumptions.
- Do not import implementation code from another tool module. Shared behavior belongs in approved shared helpers, not cross-tool coupling.
- Do not make a generated tool depend on another generated tool module. Each tool must remain self-contained at the tool level even while reusing approved shared helpers.
- Do not store client-specific or session-specific mutable state in module globals. Use `runtime.get_state()` and `runtime.set_state()` so state stays isolated per session.
- Do not let `on_frame()` await expensive work before recording an in-flight or generation claim with `runtime.set_state(...)`. Overlapping frames can otherwise launch duplicate model work before state updates land.
- Do not create untracked `asyncio.create_task()` work. If you truly need background tasks, track, cancel, and await them during shutdown; otherwise keep the flow inline and simple.
- Do not allow Take Photo or Streaming code paths to trigger the same model request or emit the same result accidentally. Duplicate calls create repeated speech and confusing output.
- Do not add motion thresholds, cooldowns, scene-change detectors, temporal windows, detached tasks, or complex state machines unless the task genuinely depends on them. Extra machinery often causes stale state, missed results, or unnecessary complexity.
- Do not return `None`, empty strings, raw structures, debugging data, or exception text to the user. Empty or internal output becomes unusable spoken feedback.
- Do not treat model failure or uncertainty as confirmed absence. Say that the tool is unsure instead.
- Do not use directions behind the camera for navigation. For clock-face guidance, stay within observable forward directions `9-12` and `1-3`.
- Do not use color or another purely visual landmark as the sole navigation cue. Prefer body-relative, clock-face, approximate-distance, or stable structural cues that remain usable without vision.
- Do not add heavy new dependencies, especially GPU-oriented ones, unless the task explicitly requires them and the current runtime plus validator clearly support them.

## Final Checks

Before finishing, verify:

- Only necessary files changed, and the default case stayed limited to the target tool file.
- The generated-tool validator still passes.
- Any relevant focused tests or existing validation scripts for the changed tool still pass.
- Single-frame versus multi-frame behavior was chosen deliberately.
- The selected model profile and strategy match the user request.
- Every return and emit contains useful non-empty text.
- State, tracked work, and cancellation are cleaned up correctly.
- No unsupported import, provider call, direct networking code, transport change, or stray background task was added.
- The implementation is no more complex than the task requires.
