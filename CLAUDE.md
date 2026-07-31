# ProgramAT

AI-powered assistive-technology platform. Blind and low-vision users build
custom camera-based tools by describing them in natural language, then run
those tools live against their phone camera and hear the result spoken back.

**The user is blind.** Audio, speech, beeps, and haptics are the primary output
channel, not an accessibility add-on. Judge every change by whether it works
without sight.

## Repository map

This monorepo has three main components and one data flow:

```text
phone camera -> ProgramATApp -> WebSocket -> backend -> tools/*.py -> spoken result
```

- `ProgramATApp/` — React Native mobile app (iOS-first), TypeScript.
- `backend/` — async WebSocket server that runs the tools. Python.
- `tools/` — pluggable camera vision tools, one per file. Python.

`backend/CLAUDE.md` and `ProgramATApp/CLAUDE.md` contain genuinely
directory-specific notes. This root file is the source of truth for
generated-tool instructions.

## Documentation

Read these instead of guessing:

- `README.md` — setup, API keys, running the app and server, usage.
- `ARCHITECTURE.md` — system design and how the components interact.
- `DEV_PRODUCTION_MODES.md` — development, production, and review modes.
- `CONTRIBUTING.md` — contribution categories, reviews, PR requirements.

## Conventions

- Most contributions are a single new tool in `tools/`.
- Branches: `feat/...`, `fix/...`, `docs/...` (kebab-case). `copilot/*` is reserved for Copilot-generated tool work.
- Never commit secrets such as `backend/.env`, `backend/credentials.json`, or service-account JSON.
- This repo is a fork of `program-at/ProgramAT-opensource` (`upstream`).

## Generated-tool contract

ProgramAT tools are executable Python modules in `tools/`. New tools own their
model selection, prompts, frame selection, temporal logic, call timing,
concurrency, state, and output behavior. The backend supplies decoded frames,
isolated runtime state, cancellation, approved model-call helpers, and result
transport.

Normally implement both entry points of the same tool:

```python
TOOL_NAME = "concise_snake_case_name"
TOOL_PROMPT = "One concise task instruction shared by Take Photo and Streaming."

async def on_take_photo(runtime, image, input_data):
    ...

async def on_stream_start(runtime, input_data):
    ...

async def on_frame(runtime, frame):
    ...

async def on_stream_stop(runtime):
    ...
```

Infer supported behavior from the hooks. Do not declare `EXECUTION_MODE`,
`TOOL_POLICY`, or call `execute_tool_policy()` in new tools. Those declarative
contracts remain only for compatibility with older tools.

Every visual tool should support Take Photo unless a single image is genuinely
meaningless. Implement meaningful Streaming behavior when possible. A static
task can process one changed latest frame; a temporal task can select recent
chronological frames. Do not force multiple frames when one is enough.

## One shared prompt

Build one `TOOL_PROMPT` from the task, expected output, constraints, and
examples, and use it in both Take Photo and Streaming.

Before writing it, briefly decide whether one direct instruction is enough or
whether the task has dependent parts that benefit from a few ordered steps:

- If one instruction is enough, keep it direct and do not add steps.
- If later conclusions depend on earlier findings, put a short numbered sequence such as 1, 2, 3 inside the same `TOOL_PROMPT`.
- Do not create separate prompts, planner stages, or model calls for individual steps.
- Do not add steps merely to restate the issue fields.

```python
TOOL_PROMPT = (
    "Assist a blind user in finding an indoor exit. "
    "1. Find visible doors, doorways, or exit signs. "
    "2. Decide which one is the most likely exit. "
    "3. Report its clock-face direction and give concise guidance toward it."
)
```

A simple task should stay simple:

```python
TOOL_PROMPT = "Read the medication label and report the text concisely."
```

The prompt should preserve the requested output, be concise and speech-ready
for blind and low-vision users, normalize a stable format when useful, and
state when evidence is insufficient. It must work with one image or several
chronological frames: never invent motion from one image, and use
chronological order when several frames are supplied.

Take Photo and Streaming may use different models, frames, timing,
orchestration, state, and output behavior while sharing this prompt.

## Runtime API

`frame` provides `frame_id`, `timestamp`, `image`, `width`, `height`, and
`image_base64`.

```python
runtime.current_request_id
runtime.current_frame
runtime.get_latest_frame()
runtime.get_recent_frames(count=None, seconds=None)
runtime.get_state(key, default=None)
runtime.set_state(key, value)
runtime.clear_state()
runtime.is_cancelled()
await runtime.emit(text, partial=False, final=False, replace=False, metadata=None)
```

State is isolated by client, tool source version, and streaming session. Keep
scene anchors, prior results, generations, and temporary tasks in runtime
state rather than module globals. Check cancellation and a tool-owned
generation or scene identifier before emitting work that may be stale.

Before implementing any temporal or multi-frame Streaming tool, inspect the
real repository code instead of inventing scheduling from scratch. Read the
generated-tool runtime API in `backend/generated_tool_runtime.py`, especially
`runtime.get_recent_frames(...)`, `runtime.get_latest_frame()`,
`runtime.emit(...)`, `runtime.is_cancelled()`, and `cancel_tasks(...)`. Also
inspect the Streaming lifecycle and cancellation flow in
`backend/stream_server.py`, including backend frame scheduling, visual-state
gating, and how `on_stream_stop` is invoked. Then inspect one or two existing
Streaming tools that already behave correctly for the requested pattern and
reuse their cleanup and state-management style where it fits. Prefer adapting a
proven repository pattern over inventing a new scheduler. For Streaming output,
also inspect where `[GeneratedTool] emit ... accepted=%s` is logged, how
`backend/stream_server.py` converts generated-tool events into websocket
messages, and how `ProgramATApp/ToolRunner.tsx` plus
`ProgramATApp/progressiveResults.ts` accept or ignore those events.

The tool may define a small literal frame configuration when the backend needs
settings before delivering frames. Do not use configuration as a substitute
for executable frame selection. Prefer runtime selection for latest-frame,
changed-scene, sparse-history, or dense chronological behavior.

## Model calls

Select models explicitly in tool code. Default ordinary visual tasks to Gemini
3.1 Flash Lite, but choose a faster, stronger, or specialized configured model
when justified. The registry in `backend/model_registry.py` is advisory; the
backend does not replace the selected model.

```python
import asyncio
from litellm_utils import call_model, extract_text

response = await asyncio.to_thread(
    call_model,
    "gemini/gemini-3.1-flash-lite-preview",
    [{"role": "user", "content": TOOL_PROMPT}],
    [image],
)
text = extract_text(response)
```

Use `call_openai_responses_model()` only for a configured model that requires
the Responses API. Do not split prompt steps into separate model calls. Take
Photo and Streaming may each choose the model appropriate to their entry point.

Choose model strategy from the requested output behavior:

- Use one model when the user does not request escalation or multiple results. Use one faster model if the user wants it to be faster, and use one more accurate model if the user wants it to be more accurate.
- Use a conditional cascade when the user wants normal cases to be fast but difficult, uncertain, empty, or insufficient cases to receive additional accuracy. Call the fast model first, inspect its result, and call the stronger model only when escalation is needed; deliver one final answer.
- Use parallel progressive only when the user explicitly wants multiple model results delivered as they complete. Do not infer progressive output merely from requests for speed, accuracy, multiple models, or better handling of difficult cases.

Model strategy remains tool-controlled. The backend must not turn a multi-model
tool into parallel progressive execution.

All three strategies are implemented directly in `on_take_photo` and `on_frame`
using `asyncio.create_task`, `asyncio.to_thread`, and `await runtime.emit(...)`. Do
not declare `TOOL_POLICY` in any new tool — not even to request parallel progressive
output. `TOOL_POLICY` is a legacy contract for older declarative take-photo tools
only; the validator rejects it in any tool that also declares executable lifecycle
hooks (`on_take_photo`, `on_frame`, etc.).

## Entry-point behavior

Take Photo receives one current image. It should use `TOOL_PROMPT`, explicitly
call its chosen model or models, and return or emit concise results. If motion
or history is essential, report that it cannot be determined from one image.

Streaming decides when and what to analyze. It may inspect the latest frame,
skip similar frames, process confirmed scene changes, select recent frames,
maintain state, or emit progressive results. Static tasks should generally
avoid unnecessary temporal windows. Temporal tasks may pass selected frames in
chronological order to the same `TOOL_PROMPT`.

Intentional parallel execution is valid when the tool needs different models
for the same input, independent analyses, progressive results, parallel
latency reduction, or several distinct useful tasks. The unsafe pattern is
accidental duplicate work, not concurrency itself.

Assume `on_frame()` may run again while earlier async work is still running.
Before an expensive call, decide what unit of work it represents, such as one
scene analysis, one temporal window, one model in an intentional multi-model
batch, or one distinct phase. Avoid launching another identical call for the
same unit of work unless retries or parallel duplicates are explicitly
intended. Repeated frames should not independently start the same model call
while equivalent work is already in flight, but intentional calls to different
models may run concurrently, intentional progressive execution may emit each
result as it completes, and a new scene, generation, temporal window, or
distinct phase may start new work.

Record an in-flight task, generation, window id, or request key before
awaiting. Validate cancellation and relevance again before emitting, clear
completed task state in `finally` or a completion callback, and cancel or
invalidate obsolete work when appropriate. Do not rely on
`if no_result_yet: result = await call_model(...); save_result(result)`,
because several overlapping `on_frame()` calls can all pass that check before
any result is saved. Do not use one global lock around all model calls; guard
only equivalent work so unrelated or intentionally parallel calls remain
concurrent. A lock or in-flight map only prevents duplicates when equivalent
work shares the same stable key. A per-frame or constantly changing generation
key does not deduplicate temporal work.

This is enforced, not just advisory: `backend/validate_generated_tools.py`
statically rejects any `on_frame()` whose first `await` in its own body
precedes any `runtime.set_state(...)` call, both in CI and at load time
before the backend `exec()`s the tool. If a generated tool computes something
cheap (an embedding, a change-detection score) before deciding whether to
launch expensive work, wrap even that computation in a tracked task and claim
its slot first, exactly like the awaited model call below — do not call
`await asyncio.to_thread(...)` directly in `on_frame()`'s body before any
state claim.

Two patterns satisfy this rule:
- **`asyncio.create_task` pattern (preferred)**: put all `await` calls inside the nested `run()` async function and launch it with `asyncio.create_task(run())`. `on_frame()`'s own body never suspends — there is no `await` for the validator to find, and `runtime.set_state(...)` is called at the end of `on_frame()`'s body before the function returns.
- **`set_state`-first pattern**: call `runtime.set_state(...)` at least once in `on_frame()`'s own body before its first `await`. The simplest form is an `analyzing` flag: claim it before awaiting a helper, reset it in `finally`. All awaited work moves into the helper `async def`.

```python
# set_state-first: analyzing flag delegates all awaited work to a helper
async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return
    if runtime.get_state("analyzing"):
        return
    runtime.set_state("analyzing", True)   # claimed before the first await
    try:
        await _describe_frame(runtime, frame)  # all awaits are inside the helper
    finally:
        runtime.set_state("analyzing", False)

async def _describe_frame(runtime, frame):
    # The helper may freely await — it is not on_frame's own body.
    embedding = await asyncio.to_thread(compute_scene_embedding, frame.image)
    generation = int(runtime.get_state("scene_generation", 0))
    anchor = runtime.get_state("scene_anchor")
    if anchor is None or not is_same_scene(anchor, embedding):
        generation += 1
        runtime.set_state("scene_generation", generation)
        runtime.set_state("scene_anchor", embedding)
    if runtime.is_cancelled():
        return
    text = await asyncio.to_thread(call_model, ...)
    if runtime.is_cancelled() or runtime.get_state("scene_generation") != generation:
        return
    await runtime.emit(extract_text(text), final=True, replace=True)
```

The following are the most common wrong patterns — each produces the validator error above:

```python
# WRONG — on_frame() awaits with no prior set_state; fails even without in-flight tracking
async def on_frame(runtime, frame):
    response = await asyncio.to_thread(call_model, ...)  # validator rejects this
    await runtime.emit(extract_text(response), final=True)

# WRONG — set_state is read but not written before the await
async def on_frame(runtime, frame):
    if runtime.get_state("in_flight"):
        return
    response = await asyncio.to_thread(call_model, ...)  # validator rejects this
    runtime.set_state("in_flight", False)  # too late: comes after the await

# WRONG — even a cheap await (embedding, change-detection score) before set_state is rejected
async def on_frame(runtime, frame):
    embedding = await asyncio.to_thread(compute_embedding, frame.image)  # validator rejects this
    runtime.set_state("last_embedding", embedding)
```

`await runtime.emit(...)` also counts as an `await` for this rule. Any `await` in `on_frame()`'s direct body — not inside a nested `async def` or lambda — must follow at least one `runtime.set_state(...)` call. Reads (`runtime.get_state(...)`) and `runtime.is_cancelled()` checks do not count.

Use a small runtime-state guard, not backend-owned scheduling. For example:

```python
async def on_frame(runtime, frame):
    scene_generation = int(runtime.get_state("scene_generation", 0))
    key = ("detailed_scene", scene_generation)
    tasks = dict(runtime.get_state("in_flight", {}))
    existing = tasks.get(key)
    if existing is not None and not existing.done():
        return

    async def run():
        try:
            text = await analyze_scene(frame.image)
            if runtime.is_cancelled():
                return
            if runtime.get_state("scene_generation") != scene_generation:
                return
            await runtime.emit(text, final=True, replace=True)
        finally:
            current = dict(runtime.get_state("in_flight", {}))
            if current.get(key) is task:
                current.pop(key, None)
                runtime.set_state("in_flight", current)

    task = asyncio.create_task(run())
    tasks[key] = task
    runtime.set_state("in_flight", tasks)
```

Do not use detached `asyncio.create_task()` scheduling by default. For ordinary
continuous temporal tools, prefer the simpler single-in-flight recent-history
pattern used by the working Copilot implementation. Only switch to detached
tasks, event generations, cooldown states, motion thresholds, or scene-change
machinery when the requested behavior genuinely requires them.

For static single-frame streaming tasks, prefer changed-latest-frame behavior:
compare the current frame to a recent anchor, skip visually equivalent scenes,
and emit only when the scene meaningfully changed or enough time has passed to
justify a refresh. `tools/empty_seat_detection.py` is the reference pattern for
this kind of gating.

For temporal or multi-frame Streaming, first decide which behavior the request
actually needs:

1. Continuous analysis:
   Use when the user expects ongoing updates during Streaming.
2. Event-driven analysis:
   Use for discrete gestures, card plays, state transitions, or one-time
   actions.
3. Periodic window analysis:
   Use when the user expects regular summaries over non-overlapping or
   deliberately advanced windows.

Multi-frame input and repeated execution are separate decisions. A tool may use
multi-frame input once for one event, or repeatedly for ongoing continuous
analysis. Do not assume that using several frames means calling the model on
every frame.

Choose single-frame or multi-frame deliberately:

- Use single-frame analysis when one image is enough, such as object identity, color, visible text, a stable hand pose, or the current spatial layout.
- Use multi-frame analysis when the task depends on change, motion, order, or recent history, such as waving, finger motion, a card being played, a person entering or leaving, a signed sequence, or a before-versus-after change.
- A tool may support both: Take Photo can use one image while Streaming uses recent chronological frames only when temporal evidence is actually useful.
- Do not use multi-frame input merely because the tool runs in Streaming mode.

For every temporal tool, explicitly decide all of the following before writing
code:

1. What one model call represents.
2. How frames are collected and ordered.
3. When a window becomes ready.
4. Whether the next window overlaps the prior one.
5. What prevents accidental duplicate calls.
6. Whether any parallel calls are intentional.
7. What state permits the next analysis.
8. How failures, cancellation, and stream stop reset the tool.

For ordinary continuous temporal tools, prefer a simple recent-history pattern
with one configured time window and one configured analysis interval. Choose:

```python
WINDOW_SECONDS = ...
INTERVAL_SECONDS = ...
MAX_SAMPLE_FRAMES = ...
```

- `WINDOW_SECONDS`: how much recent history one model call sees.
- `INTERVAL_SECONDS`: the minimum time between analysis starts.
- `MAX_SAMPLE_FRAMES`: the maximum number of representative chronological frames sent in one call.

For ordinary continuous temporal tools, prefer this simple proven pattern:

1. `on_frame()` checks cancellation.
2. Wait until the next configured analysis time.
3. If an equivalent analysis is already running, return immediately.
4. Read recent frames from the runtime.
5. Sort and sample them chronologically.
6. Pass the selected images together in one model call.
7. Emit one valid final result.
8. Advance the next analysis time by the configured interval.
9. Clear the analyzing flag in `finally` so later frames can trigger another analysis.

The intended flow is:

```text
frames continuously arrive
-> wait until the next configured analysis time
-> obtain the previous WINDOW_SECONDS of frames
-> sort and sample them chronologically
-> make one model call
-> emit one valid final result
-> wait until the next configured interval
```

This is continuous analysis with single-in-flight scheduling. It is not one
call per frame, and model completion must not immediately trigger the next
request.

Use a pattern equivalent to:

```python
import time


async def on_stream_start(runtime, input_data):
    now = time.monotonic()
    runtime.set_state("analysis_in_flight", False)
    runtime.set_state("next_analysis_at", now + WINDOW_SECONDS)


async def on_frame(runtime, frame):
    if runtime.is_cancelled():
        return

    now = time.monotonic()
    next_analysis_at = runtime.get_state("next_analysis_at")

    if now < next_analysis_at:
        return

    if runtime.get_state("analysis_in_flight", False):
        return

    recent_frames = runtime.get_recent_frames(seconds=WINDOW_SECONDS) or []
    ordered_frames = sorted(recent_frames, key=lambda candidate: candidate.timestamp)
    selected_frames = sample_evenly(
        ordered_frames,
        max_frames=MAX_SAMPLE_FRAMES,
    )
    images = [candidate.image for candidate in selected_frames if candidate.image is not None]

    while next_analysis_at <= now:
        next_analysis_at += INTERVAL_SECONDS
    runtime.set_state("next_analysis_at", next_analysis_at)

    if not images:
        return

    runtime.set_state("analysis_in_flight", True)
    try:
        response = await analyze_images(images)
        text = extract_text(response).strip() or FALLBACK_TEXT

        if runtime.is_cancelled():
            return

        await runtime.emit(
            text,
            final=True,
            metadata={
                "frame_id": frame.frame_id,
                "frame_count": len(images),
                "window_seconds": WINDOW_SECONDS,
            },
        )
    except Exception:
        logger.exception("Temporal analysis failed")
    finally:
        runtime.set_state("analysis_in_flight", False)
```

The exact variable names may differ, but preserve these properties:

- one configured time window per analysis;
- bounded chronological frame sampling;
- one equivalent analysis in flight;
- no backlog of missed calls;
- model completion does not immediately trigger another request;
- state always clears in `finally`;
- every successful analysis produces a valid non-empty final result.

Multi-frame means several ordered frames may be passed in one model call. It
does not mean making one model call for every arriving frame.

For continuous analysis, do not force a gesture-style cooldown or a permanent
`waiting_for_reset` state. Prefer a stable pattern such as:

```text
collect recent frames
-> start one analysis
-> block equivalent work while it is running
-> emit result
-> advance to a meaningfully new frame/window
-> allow the next analysis
```

The next analysis must not reuse the exact same work unit. Use a stable marker
such as `last_processed_frame_id`, `window_start/window_end`, a timestamp
boundary, or `generation + frame range`.

When the simple continuous pattern is not enough, upgrade deliberately:

- Event-driven analysis:
  Use a bounded lifecycle such as `idle -> collecting -> analyzing -> waiting_for_reset -> idle`. Do not use scene change as a substitute for event identity when the motion itself is the signal being analyzed.
- Periodic window analysis:
  Define an explicit non-overlapping or deliberately advanced window schedule. Do not let repeated `on_frame()` calls create accidental sliding-window duplicates unless continuous overlapping windows were explicitly intended by the request.
- Richer event/window identity:
  If one call represents a discrete gesture, card play, transition, or other bounded event, separate frame collection, temporal window scheduling, and model execution. `on_frame()` may keep updating recent history or motion evidence without calling a model. Give the work unit a stable identity such as `window_key = (event_generation, start_frame_id, end_frame_id)`, reserve it before awaiting, avoid equivalent overlapping analyses, and re-check cancellation and staleness before emitting.

When using multiple frames:

- obtain them through the runtime's existing recent-frame API;
- sort frames chronologically;
- select a bounded number of representative frames when history is large;
- preserve earliest-to-latest order;
- avoid passing hundreds of full-resolution frames;
- ensure at least one valid image exists;
- use a text fallback when model output is empty.

For bounded sampling, prefer representative sampling across the full recent
interval rather than only the last few frames when the action may have
occurred earlier.

A new model call must not begin merely because `on_frame()` was called, the
previous request just finished, or the current frame id differs from the prior
one. The analysis interval must be controlled by the tool's clock, not by
model latency.

When deciding between single-frame and multi-frame Streaming:

- Use a changed latest frame when one image is enough and motion history is not required.
- Use recent chronological frames only when a single image would lose essential evidence such as sign motion, before-versus-after state, or a short recent event.
- If Take Photo would have to answer "cannot determine motion from one image", Streaming is a strong candidate for multi-frame.
- Do not upgrade a static scene-description, OCR, object-identification, or navigation task to multi-frame unless the request explicitly depends on change over time.

For the same temporal window, intentional parallel execution remains valid when
several different models are deliberately compared, progressive results are
requested, independent subtasks are useful, or parallel execution reduces
latency. In that case, one temporal window may intentionally launch several
keyed calls such as `call_key = (window_key, model_name, phase)`. Prevent
accidental duplicates of the same temporal work, not intentional parallel
calls for distinct models or subtasks.

Do not confuse motion inside a temporal window, a new scene, and a new
model-call generation. They are not automatically the same thing.

Do not add motion thresholds, HSV scene histograms, scene generations,
cooldown states, detached `asyncio.create_task()` calls, or
`idle -> collecting -> analyzing -> reset` state machines by default. Only add
event-driven machinery when the request explicitly needs one result per
discrete event.

This pattern is unsafe:

```python
if enough_frames and not result_exists:
    result = await call_model(...)
    runtime.set_state("result", result)
```

Several overlapping `on_frame()` calls may all pass that condition before any
one call stores its result. Reserve equivalent temporal work before the first
`await`, for example:

```python
window_key = build_window_key(selected_frames)
in_flight = dict(runtime.get_state("in_flight", {}))
if window_key in in_flight:
    return
if runtime.get_state("last_analyzed_window") == window_key:
    return

task = asyncio.create_task(analyze(selected_frames))
in_flight[window_key] = task
runtime.set_state("in_flight", in_flight)
```

Then remove the task in `finally`, mark the window analyzed, and verify the
current generation before emitting. The exact mechanism may use tasks,
generations, state flags, locks, or window ids, but equivalent temporal work
must be deduplicated before the first `await`.

When using `asyncio.create_task`, keep the task object in runtime state, clear
it in `finally` or a completion callback, and either handle exceptions inside
the task or inspect the finished task so failures are not silently dropped. A
model failure must not permanently freeze the tool's state machine. Log or
surface task exceptions, clear or recover state after model errors, and if the
tool records a cooldown, use `time.time()` at the actual transition into
cooldown rather than a stale timestamp captured before inference. In
`on_stream_stop`, cancel and await active tasks so the stream cannot leave
detached work running.

Before writing temporal streaming code, answer these questions:

1. What event or interval does one model call represent?
2. Which chronological frames belong to that unit?
3. When is the unit ready to analyze?
4. How is it uniquely identified?
5. How is duplicate analysis of the same unit prevented?
6. What permits the next unit to begin?
7. Is any parallel execution intentional, and if so, which distinct models or subtasks are parallel?

Do not use JPEG bytes, base64 fragments, or string-prefix hashes as visual
scene similarity. Use a stable visual signal such as a lightweight embedding,
histogram, tracked generation, explicit temporal window, or another
image-derived comparison designed for repeated frames. Cache heavyweight local
models instead of loading them on every frame.

Before finishing a temporal or multi-frame tool, verify all of the following:

1. Repeated `on_frame()` calls cannot launch equivalent work.
2. Intentional parallel calls still work.
3. A model failure cannot permanently freeze the tool state.
4. `on_stream_stop` cannot leave detached tasks running.
5. Continuous tools can continue producing later results after one analysis finishes.
6. Event-based tools do not repeatedly emit the same event.
7. Empty model outputs are replaced with a fallback.
8. The emit contract matches a known working repository pattern.
9. Unnecessary event-state complexity has been avoided.

Each hook independently decides whether to return once or emit results that
append, replace, remain partial, or finalize. The backend only transports
accepted results and rejects stale or cancelled work.

## Streaming output contract

For a normal standalone Streaming result, use the repository's known working
emit contract:

```python
text = extract_text(response).strip()
text = text or FALLBACK_TEXT

await runtime.emit(
    text,
    final=True,
    metadata={
        "frame_id": frame.frame_id,
        "frame_count": len(images),
    },
)
```

Rules:

- Never emit `None`, an empty string, or whitespace-only text.
- Always normalize model output with `.strip()` and provide a spoken fallback.
- For a normal standalone result, use `final=True` without `replace=True`.
- Use `replace=True` only when replacing a previously emitted partial or provisional result and the repository's consumer supports that flow.
- Do not return `None` after a model call unless the result was already emitted through the supported event contract.
- Check cancellation before emitting.
- If the model call or emit fails, log the exception and recover tool state so future analyses remain possible.
- Include small diagnostic metadata such as frame count or window range, but do not expose debugging text to the user.
- Prefer copying the emit structure from a known working repository tool rather than inventing a new event shape.

`runtime.emit()` strips text and turns `None` into `""`. Backend
`accepted=True` only proves the runtime accepted the event; it does not prove
the payload was non-empty or suitable for frontend display. The current
frontend progressive consumer ignores empty text and may treat `replace=True`
or `final=True` differently depending on the surrounding event flow, so do not
invent replacement semantics for ordinary final results.

## Tool quality and accessibility conventions

Write each generated tool as Python in `tools/`, alongside `backend/` and
`ProgramATApp/`. Tools run on the backend, receive camera data through the
lifecycle runtime, and return or emit results for the app; they must not
connect to the backend or use WebSockets. Assume one user-facing task per tool
and make it runnable from the app's camera feed.

All user-facing results must be concise, descriptive, action-oriented, and
natural when spoken by text-to-speech. Return or emit plain language, not
JSON, code, cryptic abbreviations, debugging text, raw model metadata, stack
traces, or internal structures. Do not return `None` or an empty string as the
final result.

```python
return "No objects detected"
return "Found 5 people in the frame"
await runtime.emit("Text says: Welcome to the building", final=True)
await runtime.emit("Warning: Low light conditions", final=True)
```

An empty, failed, or low-confidence model result does not prove the requested
target is absent. State uncertainty when evidence is insufficient.

For navigation, use body-relative or clock-face directions, approximate
distance, or stable structural cues. Restrict camera-based clock directions to
9–12 and 1–3; directions behind the camera are not observable. Never use color
or another purely visual landmark as the only cue.

Avoid GPU-heavy packages unless strictly necessary. Reuse approved shared
backend helpers and useful patterns, but do not import one tool module from
another or duplicate infrastructure. Prefer changing only the requested tool
file unless the public runtime genuinely lacks a required capability. After the
code works and is tested, avoid unnecessary documentation and finish promptly.

## Safety and accessibility

Allowed:

- approved model-call helpers;
- current and recent runtime frames;
- runtime state, cancellation, and emission;
- local helper functions and normal asyncio control flow;
- OpenCV, NumPy, PIL, and CLIP when appropriate.

Forbidden:

- API keys or environment-variable access;
- direct provider SDKs or arbitrary networking;
- raw WebSockets or backend transport internals;
- subprocesses or unrestricted filesystem writes;
- other clients, sessions, or backend-private globals.

Use body-relative directions or stable nonvisual and structural cues for
navigation. Never use color or another visual-only landmark as the sole cue.

## Examples the contract must support

- Exit Finder: Take Photo analyzes one image; Streaming analyzes a changed latest frame. Both use the same prompt and no unnecessary temporal window.
- Hand Gesture Identifier: Take Photo reports a visible static posture and uncertainty about motion; Streaming may pass several chronological frames to the same prompt.
- Uber Finder: Take Photo may use an accuracy-oriented model; Streaming may use a faster model and update only on scene changes, still using the same task prompt.
