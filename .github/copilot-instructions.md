# ProgramAT code-agent instructions

ProgramAT tools are executable Python modules in `tools/`. New tools own their model selection, prompts, frame selection, temporal logic, call timing, concurrency, state, and output behavior. The backend supplies decoded frames, isolated runtime state, cancellation, approved model-call helpers, and result transport.

## Generated-tool contract

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

Infer supported behavior from the hooks. Do not declare `EXECUTION_MODE`, `TOOL_POLICY`, or call `execute_tool_policy()` in new tools. Those declarative contracts remain only for compatibility with older tools.

Every visual tool should support Take Photo unless a single image is genuinely meaningless. Implement meaningful Streaming behavior when possible. A static task can process one changed latest frame; a temporal task can select recent chronological frames. Do not force multiple frames when one is enough.

## One shared prompt

Build one `TOOL_PROMPT` from the task, expected output, constraints, and examples, and use it in both Take Photo and Streaming.

Before writing it, briefly decide whether one direct instruction is enough or whether the task has dependent parts that benefit from a few ordered steps:

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

The prompt should preserve the requested output, be concise and speech-ready for blind and low-vision users, normalize a stable format when useful, and state when evidence is insufficient. It must work with one image or several chronological frames: never invent motion from one image, and use chronological order when several frames are supplied.

Take Photo and Streaming may use different models, frames, timing, orchestration, state, and output behavior while sharing this prompt.

## Runtime API

`frame` provides `frame_id`, `timestamp`, `image`, `width`, `height`, and `image_base64`.

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

State is isolated by client, tool source version, and streaming session. Keep scene anchors, prior results, generations, and temporary tasks in runtime state rather than module globals. Check cancellation and a tool-owned generation or scene identifier before emitting work that may be stale.

The tool may define a small literal frame configuration when the backend needs settings before delivering frames. Do not use configuration as a substitute for executable frame selection. Prefer runtime selection for latest-frame, changed-scene, sparse-history, or dense chronological behavior.

## Model calls

Select models explicitly in tool code. Default ordinary visual tasks to Gemini 3.1 Flash Lite, but choose a faster, stronger, or specialized configured model when justified. The registry in `backend/model_registry.py` is advisory; the backend does not replace the selected model.

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

Use `call_openai_responses_model()` only for a configured model that requires the Responses API. Do not split prompt steps into separate model calls. Take Photo and Streaming may each choose the model appropriate to their entry point.

Choose model strategy from the requested output behavior:

- Use one model when the user does not request escalation or multiple results. Use one faster model if the user wants it to be faster, and use one more accurate model if the user wants it to be more accurate.
- Use a conditional cascade when the user wants normal cases to be fast but difficult, uncertain, empty, or insufficient cases to receive additional accuracy. Call the fast model first, inspect its result, and call the stronger model only when escalation is needed; deliver one final answer.
- Use parallel progressive only when the user explicitly wants multiple model results delivered as they complete. Do not infer progressive output merely from requests for speed, accuracy, multiple models, or better handling of difficult cases.

Model strategy remains tool-controlled. The backend must not turn a multi-model tool into parallel progressive execution.

## Entry-point behavior

Take Photo receives one current image. It should use `TOOL_PROMPT`, explicitly call its chosen model or models, and return or emit concise results. If motion or history is essential, report that it cannot be determined from one image.

Streaming decides when and what to analyze. It may inspect the latest frame, skip similar frames, process confirmed scene changes, select recent frames, maintain state, or emit progressive results. Static tasks should generally avoid unnecessary temporal windows. Temporal tasks may pass selected frames in chronological order to the same `TOOL_PROMPT`.

Intentional parallel execution is valid when the tool needs different models for the same input, independent analyses, progressive results, parallel latency reduction, or several distinct useful tasks. The unsafe pattern is accidental duplicate work, not concurrency itself.

Assume `on_frame()` may run again while earlier async work is still running. Before an expensive call, decide what unit of work it represents, such as one scene analysis, one temporal window, one model in an intentional multi-model batch, or one distinct phase. Avoid launching another identical call for the same unit of work unless retries or parallel duplicates are explicitly intended. Repeated frames should not independently start the same model call while equivalent work is already in flight, but intentional calls to different models may run concurrently, intentional progressive execution may emit each result as it completes, and a new scene, generation, temporal window, or distinct phase may start new work.

Record an in-flight task, generation, window id, or request key before awaiting. Validate cancellation and relevance again before emitting, clear completed task state in `finally` or a completion callback, and cancel or invalidate obsolete work when appropriate. Do not rely on `if no_result_yet: result = await call_model(...); save_result(result)`, because several overlapping `on_frame()` calls can all pass that check before any result is saved. Do not use one global lock around all model calls; guard only equivalent work so unrelated or intentionally parallel calls remain concurrent.

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

For static single-frame streaming tasks, prefer changed-latest-frame behavior: compare the current frame to a recent anchor, skip visually equivalent scenes, and emit only when the scene meaningfully changed or enough time has passed to justify a refresh. `tools/empty_seat_detection.py` is the reference pattern for this kind of gating.

For multi-frame streaming tasks, one task still means one in-flight analysis. The difference is only frame selection: gather the needed recent chronological frames, analyze that window once, and avoid launching a second overlapping window until the current one finishes or becomes stale. Use multi-frame only when the answer genuinely depends on motion, ordering, duration, transitions, or evidence spread across several recent frames.

When deciding between single-frame and multi-frame Streaming:

- Use a changed latest frame when one image is enough and motion history is not required.
- Use recent chronological frames only when a single image would lose essential evidence such as sign motion, before-versus-after state, or a short recent event.
- If Take Photo would have to answer "cannot determine motion from one image", Streaming is a strong candidate for multi-frame.
- Do not upgrade a static scene-description, OCR, object-identification, or navigation task to multi-frame unless the request explicitly depends on change over time.

Do not use JPEG bytes, base64 fragments, or string-prefix hashes as visual scene similarity. Use a stable visual signal such as a lightweight embedding, histogram, tracked generation, explicit temporal window, or another image-derived comparison designed for repeated frames. Cache heavyweight local models instead of loading them on every frame.

Each hook independently decides whether to return once or emit results that append, replace, remain partial, or finalize. The backend only transports accepted results and rejects stale or cancelled work.

## Tool quality and accessibility conventions

Write each generated tool as Python in `tools/`, alongside `backend/` and `ProgramATApp/`. Tools run on the backend, receive camera data through the lifecycle runtime, and return or emit results for the app; they must not connect to the backend or use WebSockets. Assume one user-facing task per tool and make it runnable from the app's camera feed.

All user-facing results must be concise, descriptive, action-oriented, and natural when spoken by text-to-speech. Return or emit plain language, not JSON, code, cryptic abbreviations, debugging text, raw model metadata, stack traces, or internal structures. Do not return `None` or an empty string as the final result.

```python
return "No objects detected"
return "Found 5 people in the frame"
await runtime.emit("Text says: Welcome to the building", final=True)
await runtime.emit("Warning: Low light conditions", final=True)
```

An empty, failed, or low-confidence model result does not prove the requested target is absent. State uncertainty when evidence is insufficient.

For navigation, use body-relative or clock-face directions, approximate distance, or stable structural cues. Restrict camera-based clock directions to 9–12 and 1–3; directions behind the camera are not observable. Never use color or another purely visual landmark as the only cue.

Avoid GPU-heavy packages unless strictly necessary. Reuse approved shared backend helpers and useful patterns, but do not import one tool module from another or duplicate infrastructure. Prefer changing only the requested tool file unless the public runtime genuinely lacks a required capability. After the code works and is tested, avoid unnecessary documentation and finish promptly.

If you would ever call `validate_rtvi_streaming_tool`, call `validate_temporal_streaming_tool` instead.

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

Use body-relative directions or stable nonvisual and structural cues for navigation. Never use color or another visual-only landmark as the sole cue.

## Examples the contract must support

- Exit Finder: Take Photo analyzes one image; Streaming analyzes a changed latest frame. Both use the same prompt and no unnecessary temporal window.
- Hand Gesture Identifier: Take Photo reports a visible static posture and uncertainty about motion; Streaming may pass several chronological frames to the same prompt.
- Uber Finder: Take Photo may use an accuracy-oriented model; Streaming may use a faster model and update only on scene changes, still using the same task prompt.
