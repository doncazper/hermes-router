# ModelRouter Productization Roadmap

This document ingests the June 2026 Codex prompt pack into the repo as product
planning truth. The full shared admin state/action contract lives in
`docs/codex/admin-state-contract.yaml`.

## Product Move

ModelRouter should remain a local OpenAI-compatible proxy router and admin
control center, not a chat UI and not an agent workspace. Its default mode should
remain the deterministic decision router.

The new product direction is to make the decision layer optional. Users should
be able to run ModelRouter as:

- `decision`: current smart routing behavior, with prompt classification,
  receipts, policies, fallback, telemetry, and safety gates.
- `manual`: no prompt classification; forward to a configured backend/model.
- `model_map`: no prompt classification; resolve the inbound `model` through
  configured aliases.
- `passthrough`: no prompt classification; forward through a default backend and
  preserve the client model when present.

This matters because it turns ModelRouter from a clever router into a practical
local model gateway: the same endpoint can support smart routing, manual model
selection, aliases, health, telemetry, runtime controls, and model management.

## Guardrails

- Decision mode remains the default and must not regress.
- Non-decision modes must not call `route_fast(...)` or `route(...)`.
- Web UI, future TUI, installer, and admin API should share one state/action
  layer.
- Every visible UI/TUI value must come from shared state, a real action result,
  a useful empty state, or a disabled control with a concrete reason.
- Every mutating action requires explicit user confirmation.
- No silent model downloads.
- No silent hosted-provider enablement.
- No silent config writes.
- No silent runtime start, stop, restart, load, or unload.
- No silent benchmark execution.
- No raw prompt display unless prompt capture is explicitly configured.
- Keep optional compatibility work outside `route_fast(...)`.

## Imported Contract

The attached admin contract was ingested as:

- `docs/codex/admin-state-contract.yaml`

The contract defines:

- Shared top-level state: `app`, `proxy`, `routes`, `model_aliases`,
  `backends`, `model_library`, `installer`, `telemetry`, `latest_receipt`,
  `logs`, `actions`, and `maturity`.
- Required proxy config extensions: `routing_mode`, `default_backend`,
  `default_model`, `respect_client_model`, `unknown_model_behavior`,
  `safety_gate_mode`, and `model_aliases`.
- Required shared actions: proxy lifecycle, config saves, routing mode changes,
  model scan/discover/download/assignment, runtime load/unload, doctor,
  benchmarks, feedback, and catalog updates.
- Render bindings for dashboard, models, routing, runtimes, and future TUI.
- Validation and test requirements for mode behavior and confirmation
  enforcement.

## Recommended Milestones

### M0: Shared Admin Backend Extraction

Create `hermes/plugins/model_router/admin/` and move reusable non-rendering
logic out of `settings_ui.py`. The web settings UI should become FastAPI route
glue plus HTML rendering over shared state/actions.

Minimum useful slice:

- `admin.state.build_admin_state`
- `admin.actions.run_admin_action`
- `admin.supervisor.ProxyProcessSupervisor`
- `admin.config_edit.save_proxy_config_patch`

Done when existing settings behavior still works, new shared-state tests cover
valid/missing/invalid config, and mutating actions reject missing confirmation.

### M1: Basic Router Mode

Add `proxy.routing_mode` with `decision`, `manual`, `model_map`, and
`passthrough` semantics. Preserve current decision behavior exactly.

Done when non-decision modes do not call `route_fast(...)`, headers and
telemetry expose the mode, `/v1/models` exposes aliases predictably, and unknown
models can either fall back or return `404` by config.

Escape hatch: ship only `decision` and `manual` first, and reject the other
modes with clear validation errors.

### M2: Installer v1

Add `model-router install` as deterministic onboarding. It should detect install
method, Python/package version, command availability, optional dependencies,
config state, port availability, and local runtime signals, then print clear next
steps.

Done when `model-router install --json` is deterministic and tested, existing
configs are not overwritten by default, and output tells the user exactly what
to run next.

### M3: Model Library And Discover UI

Add real model-management surfaces in the existing settings UI style:
Installed, Discover, Recommended, Downloads, and Assignments.

Done when every row comes from state, downloads require plan/confirmation,
assignments persist through config actions, and no-model empty states are useful.

### M4: Runtime Adapters

Add a runtime adapter protocol for health, model discovery, loaded models,
capabilities, load/unload support, and logs across generic OpenAI-compatible
backends, LM Studio, Ollama, MLX-LM, llama.cpp, LocalAI, and custom backends.

Done when adapter failures do not crash settings/TUI and unsupported actions are
disabled with reasons.

### M5: TUI v1

Add `model-router tui` as a terminal control center using the shared state/action
layer. This should be useful without a browser.

Escape hatch: ship a read-only TUI first, but it must still use real shared
state and honest empty states.

### M6: API Compatibility Expansion

Add capability-driven forwarding for `/v1/embeddings` and `/v1/completions`.
Only add `/v1/messages` when capability plumbing is ready; otherwise fail
clearly.

Done when unsupported endpoints return useful errors and no endpoint ignores
`routing_mode`.

### M7: Maturity And Release Gate

Track feature maturity for basic router mode, installer, model library, runtime
adapters, TUI, and compatibility endpoints. Show maturity where useful and add
dogfood checks, upgrade/uninstall docs, config migration notes, and release
gates.

## Immediate Next Step

Start with M0. The current dashboard has grown into the control center, but its
state/action logic still lives largely inside `settings_ui.py`. Extracting the
shared admin layer first gives the web UI, future TUI, installer, and admin API a
single control plane. That reduces the risk of implementing basic router mode or
model management twice.

After M0, implement the smallest stable slice of M1: `decision` plus `manual`
mode. That gives users the first concrete "decision layer off" path without
overloading the next release.
