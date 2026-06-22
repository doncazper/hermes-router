# Changelog

## 0.6.1 - Routing Telemetry Dogfood Workflow

- Added `model-router telemetry summary` for event/replay coverage, unlabeled
  replayable request ids, skipped private/no-prompt events, route changes, and
  `expected -> actual` mismatch groups.
- Added `model-router telemetry feedback` for inspecting feedback labels without
  printing feedback notes by default.
- Shared replay analysis between the CLI and `scripts/replay_routing_log.py`.
- Documented the privacy-safe dogfood loop and the 20-30 labeled wrong-route
  threshold for revisiting optional advanced routing.

## 0.6.0 - Compatibility Expansion

- Added `/v1/responses` passthrough and routing alongside
  `/v1/chat/completions`.
- Preserved common Responses API request fields when forwarding, including
  `instructions`, `tools`, `tool_choice`, `parallel_tool_calls`, `metadata`,
  `previous_response_id`, streaming, and fallback behavior.
- Reused the same proxy auth, backend selection, model override, fallback,
  streaming, and `human_confirm` blocking paths for Responses API requests.
- Improved tool-call compatibility coverage, including `parallel_tool_calls`
  preservation and per-backend stripping behavior.
- Updated proxy documentation for the Responses API endpoint.

## 0.5.4 - Docs And Product UX Refresh

- Refreshed the README/PyPI long description with a proxy-first setup path.
- Added known-good LM Studio and Ollama setup examples for local routing.
- Added sample transcripts for `model-router init`, `model-router doctor`,
  `model-router-proxy`, and generic OpenAI-compatible agent configuration.
- Documented the wrong-route feedback, replay, and regression-test workflow.
- Clarified that `hermes/plugins/...` is only a legacy Python namespace, not a
  host-application plugin integration point.

## 0.5.3 - Proxy Hardening Release

- Added a live uvicorn/raw-socket streaming disconnect test that runs the real
  proxy against a controlled ASGI upstream, disconnects before the stream
  completes, and verifies upstream stream cleanup.
- Classified ASGI client cancellation during streaming as `stream_interrupted`
  while still re-raising cancellation and closing the upstream stream context.
- Verified the disconnect path can write metadata-only routing events without
  logging raw prompts, request bodies, proxy API keys, or upstream secrets.

## 0.5.2 - Routing Accuracy Calibration

- Calibrated short prompt routing so clear summary/explanation requests like
  `summarize this` and `explain this` route to `balanced_local` instead of
  being over-escalated as ambiguous reasoning work.
- Calibrated simple edit prompts like `make this clearer` and `fix typo` to
  route to `fast_local`.
- Improved coding intent detection for prompts that ask to write code objects,
  such as functions, classes, modules, or scripts, while avoiding false
  positives like `what is the function of mitochondria`.
- Added replay fixtures and parity coverage for the calibrated prompt set.

## 0.5.1 - Dogfood Stability Fixes

- Improved `model-router doctor` and proxy `/health` diagnostics so a backend
  that serves `/v1/models` but does not list the configured model is reported as
  degraded instead of healthy.
- Verified the published PyPI install path, local proxy endpoint behavior,
  streaming, fallback, human-confirm, closed-port diagnostics, missing env-var
  diagnostics, and proxy auth handling during dogfood testing.

## 0.5.0 - Usable Local Proxy Beta

- Added `model-router init` for first-run local proxy setup.
- Added provider presets for LM Studio, Ollama, llama.cpp server, LocalAI, and
  hosted OpenAI-compatible gateways.
- Added `model-router validate-proxy-config` and `model-router doctor`.
- Added backend reachability details to `/health`.
- Added JSONL log rotation controls for routing observability.
- Added PyPI-first release workflow scaffolding and release checklist docs.

## 0.4.1

- Added OpenAI-compatible proxy, hindsight JSONL logging, feedback labels, and
  routing-log replay tooling.
