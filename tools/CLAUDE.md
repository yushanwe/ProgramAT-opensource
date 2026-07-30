# tools/ — executable assistive tools

Follow `.github/copilot-instructions.md` as the generated-tool contract. New and updated tools normally define `TOOL_NAME`, one shared `TOOL_PROMPT`, and both Take Photo and Streaming lifecycle hooks:

```python
async def on_take_photo(runtime, image, input_data): ...
async def on_stream_start(runtime, input_data): ...
async def on_frame(runtime, frame): ...
async def on_stream_stop(runtime): ...
```

The tool owns model choices, frame selection, temporal logic, orchestration, state, and output timing. Use approved model helpers and runtime frames, state, cancellation, and emission. Keep output concise and accessible.

In Streaming, assume `on_frame()` may be called again before earlier async work finishes. Intentional parallel work is allowed for different models, independent analyses, progressive results, latency reduction, or distinct useful phases. Before expensive work, decide what unit it represents and record an in-flight task, generation, window id, or request key in runtime state so only equivalent duplicate work is blocked while unrelated or intentional parallel calls remain concurrent.

Do not rely on `if missing_result: await model_call(...); runtime.set_state(...)`, because several frames may pass that check concurrently. Re-check cancellation and relevance before emitting, clear task state in `finally` or a completion callback, and cancel or invalidate obsolete work when appropriate. Do not use JPEG/base64 string fragments as scene similarity. Cache heavyweight local models rather than loading them on every frame.

Do not access credentials, environment variables, provider SDKs, arbitrary networks, WebSockets, subprocesses, unrestricted filesystem writes, other sessions, or backend-private globals.

Old `main(image, input_data)` and declarative `EXECUTION_MODE`/`TOOL_POLICY` tools remain supported for compatibility, but do not use those contracts for new or updated tools.
