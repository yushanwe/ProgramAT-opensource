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

ProgramAT tools are executable Python modules in `tools/`. A generated tool
should normally modify only one Python file under `tools/`. The tool owns its
prompt, model selection, frame selection, runtime state, and output behavior.
The backend supplies decoded frames, isolated state, cancellation, approved
model helpers, and result transport.

Normally implement the hooks the task actually needs:

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

Do not declare `EXECUTION_MODE`, `TOOL_POLICY`, or call
`execute_tool_policy()` in new tools.

## What generated tools may do

- Call approved model helpers such as `call_model`, `extract_text`, and
  `call_openai_responses_model`.
- Use one current image or recent runtime frames.
- Keep isolated runtime state with `runtime.get_state(...)` and
  `runtime.set_state(...)`.
- Emit spoken results through `runtime.emit(...)`.
- Choose their own model strategy.
- Use normal asyncio control flow and approved packages that pass the
  validator.

## What generated tools must not do

- Modify `ToolRunner`, `stream_server.py`, or shared runtime code merely to
  make one tool work.
- Change generated-tool validation.
- Add WebSocket, transport, or backend event code.
- Access API keys directly, use arbitrary networking, or add unsupported
  imports.
- Bypass `_validated_generated_namespace()`.

Before finishing, make sure the source passes the existing generated-tool
validator.

## Prompt and output

Build one `TOOL_PROMPT` for the tool and use it in both Take Photo and
Streaming. Keep it concise, speech-ready, and clear about uncertainty.

All user-facing results must be non-empty, concise, plain language, and safe
for a blind or low-vision user. Normalize model output:

```python
text = extract_text(response).strip()
text = text or FALLBACK_TEXT
```

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
tool-specific state in the runtime, not in module globals, unless it is a
read-only helper cache such as a lazily loaded local model.

## Choose the input type

Decide first whether the task needs one frame or several frames.

- Use one frame when the answer depends on the current visible state, such as
  OCR, object identification, color, scene description, current layout, or a
  static hand pose.
- Use several chronological frames only when the answer depends on motion,
  order, before-and-after change, a gesture sequence, or recent temporal
  context.

Streaming does not automatically mean multi-frame. Progressive output does not
automatically mean multi-frame.

## Choose the model strategy

Treat model strategy separately from frame strategy.

### Single model

Use one model when the user only needs one normal answer.

### Conditional cascade

Use a fast model first and a stronger model only when the first result is
empty, uncertain, insufficient, or clearly wrong. Return one final answer.

### Progressive

Use progressive only when the user explicitly wants early usefulness followed
by more detail or accuracy, for example:

- “Give me a quick answer first, then a more detailed one.”
- “Use several models and report each result as it finishes.”
- “Start with a fast rough description, then provide a stronger description.”

For executable Streaming tools, progressive means several useful normal
Streaming results over time:

```text
quick result
-> later detailed result
-> optional stronger result
```

Each result should be non-empty and understandable on its own.

Use normal emits such as:

```python
await runtime.emit(text, final=True, metadata={"phase": phase})
```

Do not use `replace=True` by default. Do not emit empty final markers. Do not
depend on `tool_progress_started` or `tool_progress_result`; those belong to
the separate backend-coordinated one-shot progressive path.

## Keep implementations simple

Prefer the simplest implementation that satisfies the request.

For ordinary Streaming tools:

- Prevent accidental duplicate equivalent calls.
- Keep a small in-flight flag or tracked task when needed.
- Clear state in `finally`.
- Check cancellation before emitting.
- Cancel and await tracked work in `on_stream_stop`.

Do not add scene generations, motion thresholds, cooldowns, temporal windows,
detached tasks, or complex state machines unless the task truly requires them.
Do not create two different code paths that can emit the same stage. Do not
create untracked `asyncio.create_task()` work.

Before using unusual imports or patterns, inspect the actual validator and one
accepted generated tool from this repository. Do not assume every Python
standard-library import is allowed.

For intentionally progressive multi-model tools, each model or stage should
fail independently. One failure should not suppress other intentional results.

## Streaming output contract

Two result transports exist:

- Backend-coordinated one-shot progressive:
  uses `tool_progress_started` and `tool_progress_result`.
- Tool-controlled executable Streaming progression:
  a generated tool calls `runtime.emit(...)`, and each meaningful update is
  transported as a normal `tool_stream_result`.

For normal executable Streaming:

- Every emit should contain useful non-empty text.
- Use `final=True` for normal standalone results.
- Use `replace=True` only when the public runtime contract clearly requires it.
- Do not change frontend or backend transport for one generated tool.

`accepted=True` only means the runtime accepted the event locally; it does not
prove frontend delivery.

## Final checks

Before completing a generated tool, verify:

1. Only the intended tool file was changed.
2. The source passes the validator.
3. Single-frame versus multi-frame was chosen deliberately.
4. Model strategy matches the user’s request.
5. Progressive was used only when explicitly requested.
6. Every emit contains useful non-empty text.
7. No unsupported backend or frontend changes were made.
8. No untracked async task remains.
9. The tool can stop cleanly.
10. The implementation is simpler than any unnecessary alternative.

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
