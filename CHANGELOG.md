# Changelog

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
