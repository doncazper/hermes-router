# Implementation briefs

These briefs are intended for one focused implementation session at a time. They avoid vague UI placeholders and require all visible controls to be connected to shared admin state/actions.

## Shared constraints

- Preserve ModelRouter as a local AI control center and proxy routing/control
  plane.
- Support common local-model workflows through explicit model discovery,
  recommendations, downloads, configured runtime controls, local endpoint
  exposure, and request routing.
- Integrate with proven runtimes such as LM Studio, Ollama, LocalAI, llama.cpp,
  MLX/MLX-LM, vLLM, generic OpenAI-compatible backends, and hosted providers
  instead of building a custom inference engine.
- Keep the current settings UI visual style.
- Use shared backend state/actions for the web UI and TUI.
- Require explicit confirmation for config writes, runtime process changes, downloads, benchmarks, and proxy restarts.
- Add tests for every new behavior.

## M0 — shared admin backend

Objective: move reusable settings state/actions into `hermes/plugins/model_router/admin/` while keeping `model-router settings` working.

Implementation targets:

- `admin.state.build_admin_state`
- `admin.actions.run_admin_action`
- `admin.supervisor.ProxyProcessSupervisor`
- `admin.config_edit.save_proxy_config_patch`
- route/model/backend/telemetry state builders

Acceptance:

- Existing settings routes still respond.
- State includes proxy mode, decision-layer status, routes, model aliases, backends, model library, telemetry, latest receipt, logs, and actions.
- The future TUI can consume the same state.

## M1 — basic router mode

Objective: add a mode where the decision layer is disabled and the proxy acts as a simple backend/model router.

Implementation targets:

- `proxy.routing_mode`: `decision`, `manual`, `model_map`, `passthrough`
- `proxy.default_backend`
- `proxy.default_model`
- `proxy.respect_client_model`
- `proxy.unknown_model_behavior`
- `proxy.safety_gate_mode`
- `model_aliases`

Acceptance:

- Decision mode remains the default.
- Non-decision modes do not call `route_fast`.
- Responses include mode and decision-layer headers.
- Telemetry and receipts explain static/manual routing.
- Unknown model behavior supports fallback or a clear 404.

## M2 — installer v1

Objective: add `model-router install` as the official onboarding command.

Acceptance:

- Detects install method, Python version, optional dependencies, config directory, existing configs, ports, and local runtime signals.
- Uses existing safe initialization helpers.
- Prints endpoint and next actions.
- Supports deterministic JSON output.

## M3 — model library UI

Objective: add real model-management UX to the settings UI.

Sections:

- Installed
- Discover
- Recommended
- Downloads
- Assignments

Acceptance:

- Installed rows come from scans/runtime adapters.
- Discover uses curated catalog first.
- Downloads plan before execution.
- Assignments write config through confirmed actions.
- No fake rows.

## M4 — runtime adapters

Objective: add runtime-independent backend adapters.

Adapter operations:

- health
- discover models
- list loaded models
- load model
- unload model
- capabilities
- logs

Acceptance:

- Generic OpenAI-compatible adapter lands first.
- LM Studio, Ollama, MLX-LM, and llama.cpp adapters can mature independently.
- Unsupported controls are disabled with reasons.

## M5 — TUI v1

Objective: add `model-router tui` using optional Textual dependency.

Tabs:

- Status
- Models
- Routing
- Runtimes
- Telemetry
- Logs
- Settings

Acceptance:

- TUI consumes shared admin state/actions.
- Missing Textual gives a clear install hint.
- Mutating actions require confirmation.
- Empty states are useful.

## M6 — API compatibility

Objective: expand endpoint parity after routing mode is stable.

Order:

1. `/v1/embeddings`
2. `/v1/completions`
3. `/v1/messages`

Acceptance:

- Capability checks drive endpoint availability.
- Decision and basic router modes behave consistently.
- Unsupported endpoints fail clearly.

## M7 — maturity polish

Objective: add maturity labels, dogfood coverage, release docs, and migration guidance.

Acceptance:

- Features are labeled planned, experimental, beta, or stable.
- Experimental failures do not break stable proxy flows.
- Dogfood covers decision mode and basic router modes.
