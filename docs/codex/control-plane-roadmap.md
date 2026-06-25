# ModelRouter control-plane roadmap

## Purpose

ModelRouter started as a lightweight decision layer for agents: one local OpenAI-compatible endpoint where Hermes or another agent sends a request and ModelRouter chooses the right model, tool-capable backend, or confirmation gate.

The product is now growing toward a local control plane that also needs the approachable operational flow users expect from tools like LM Studio: model discovery, model selection, runtime health, settings, logs, downloads, and server controls. The goal is not to copy LM Studio. The goal is to keep ModelRouter's unique routing brain while adding enough model-management and installer polish that users can run it without editing YAML first.

## Product positioning

**LM Studio runs and manages local models. ModelRouter decides and controls where AI traffic goes.**

ModelRouter should become the local AI traffic controller:

```text
Agents / OpenAI clients / Claude-style clients / Codex / Hermes
        |
        v
http://127.0.0.1:8082/v1
        |
        v
ModelRouter proxy + control plane
        |
        +--> LM Studio
        +--> Ollama
        +--> MLX-LM
        +--> llama.cpp
        +--> LocalAI
        +--> hosted OpenAI-compatible gateways
        +--> code agents
        +--> RAG / embedding services
        +--> vision / image-generation backends
        +--> human_confirm
```

## Current parity targets to keep in mind

Use these as parity references, not as UI cloning targets:

- LM Studio docs describe model search/download via Hugging Face, local chat, MCP, OpenAI-like local endpoints, and local model/config management: https://lmstudio.ai/docs/app
- LM Studio's OpenAI compatibility includes `/v1/models`, `/v1/responses`, `/v1/chat/completions`, `/v1/embeddings`, and `/v1/completions`: https://lmstudio.ai/docs/developer/openai-compat
- LM Studio's REST API surface includes list, load, download, unload, and download-status model operations: https://lmstudio.ai/docs/developer/rest
- LM Studio's CLI exposes model download/list/load/unload, loaded-process listing, server start/stop/status, runtime, and daemon workflows: https://lmstudio.ai/docs/cli
- Textual is the preferred TUI framework because it is a Python TUI framework that can run in terminals, over SSH, and in a browser-backed mode: https://textual.textualize.io/

## First-class operating modes

The biggest new requirement is that the decision layer must be optional.

### Mode 1: decision router

This is the current differentiator.

- Incoming chat/Responses request enters the proxy.
- Proxy extracts prompt and route hints.
- `route_fast(...)` selects an engine.
- Engine maps to backend.
- Backend overrides outbound `model` unless configured otherwise.
- Safety gates can route to `human_confirm`.
- Receipts explain selected route, rejected providers, fallback, policy, privacy, and feedback action.

### Mode 2: basic router

The decision layer is disabled. ModelRouter behaves like a simple OpenAI-compatible model router/model gateway.

Use cases:

- User wants a single stable local endpoint but wants manual model selection.
- User wants LM Studio-like model selector behavior without prompt classification.
- Agent already knows which model alias to call.
- User is debugging runtime/model behavior without routing logic in the way.
- User wants fallback, telemetry, server controls, and model library but not prompt-aware routing.

Required behavior:

- No prompt classification call.
- No `route_fast(...)` selection.
- Requests go to a configured default backend/model, or to a backend resolved from the client-supplied `model` field.
- Telemetry still records mode, backend, model, status, fallback, latencies, and request id.
- Response headers expose that decision routing is disabled.
- Basic receipts explain static/manual routing, not decision routing.

Recommended config shape:

```yaml
proxy:
  host: 127.0.0.1
  port: 8082
  routing_profile: balanced

  # New. Keep current behavior as default.
  routing_mode: decision   # decision | manual | model_map | passthrough

  # Used when routing_mode is manual or passthrough.
  default_backend: balanced
  default_model: null

  # Used when routing_mode is manual/passthrough/model_map.
  respect_client_model: true
  unknown_model_behavior: fallback_to_default  # fallback_to_default | reject_404

  # Safety behavior when the decision layer is disabled.
  safety_gate_mode: decision_only  # decision_only | always_static | off

model_aliases:
  qwen-fast:
    backend: fast
    model: qwen3-0.6b
    description: Fast rewrite/extraction alias.
  qwen-coder:
    backend: code
    model: qwen2.5-coder-7b
    description: Coding alias.
```

Mode semantics:

| `routing_mode` | Behavior | User-facing name |
| --- | --- | --- |
| `decision` | Current ModelRouter behavior; prompt-aware engine/backend selection. | Smart router |
| `manual` | Always use `default_backend`; use `default_model` unless `respect_client_model` is true and the request model is allowed. | Manual backend |
| `model_map` | Resolve request `model` through `model_aliases`; no prompt classification. | Model aliases |
| `passthrough` | Forward to `default_backend` and preserve request `model`; no model override except optional allowlist/policy. | Passthrough |

Required headers:

```text
X-ModelRouter-Mode: decision|manual|model_map|passthrough
X-ModelRouter-Decision-Layer: enabled|disabled
X-ModelRouter-Request-ID: <privacy-safe id>
X-ModelRouter-Backend: <backend name when used>
X-ModelRouter-Model: <resolved outbound model when known>
X-ModelRouter-Fallback: true|false
X-ModelRouter-Profile: <profile, only meaningful for decision mode>
```

Basic-mode receipt shape:

```json
{
  "summary": "Manual routing sent request to backend balanced with model qwen3-4b.",
  "routing_mode": "manual",
  "decision_layer_enabled": false,
  "selected_engine": null,
  "selected_backend": "balanced",
  "selected_model": "qwen3-4b",
  "reason_codes": ["mode.manual", "backend.default", "model.configured"],
  "fallback_used": false,
  "policy_explanation": "No prompt classification was performed because routing_mode=manual.",
  "safety_explanation": "Safety gate mode is decision_only; static routing did not invoke human_confirm.",
  "wrong_route_next_action": "Change the model alias/default backend or re-enable smart routing."
}
```

## Required surfaces

ModelRouter should have four official surfaces:

| Surface | Command / path | Purpose |
| --- | --- | --- |
| Installer | `model-router install` | Installs, initializes, detects runtimes, validates ports, and starts next action. |
| Proxy | `model-router-proxy` | Serves OpenAI-compatible and Anthropic-compatible requests. |
| Web UI | `model-router settings` | Local browser admin surface for model library, routing, runtimes, telemetry, and settings. |
| TUI | `model-router tui` | Terminal control center with the same state/actions as the web UI. |

## Implementation principles

1. **No placeholders in UI/TUI.** If a field appears, it must either display real state or perform a real confirmed action.
2. **One admin backend.** Web UI and TUI must share `admin.state` and `admin.actions`; do not duplicate logic.
3. **No silent mutation.** Downloads, config writes, proxy restarts, hosted-provider enablement, benchmark execution, runtime load/unload, and service installation require explicit confirmation.
4. **Decision mode remains default.** Existing users should not lose current routing behavior.
5. **Basic router mode is not second-class.** It must have headers, telemetry, receipts, config validation, UI/TUI controls, tests, and docs.
6. **Model library is route-aware.** Marketplace/discover results are not just a list of models; they show fit for fast/balanced/reasoning/code/research/vision/image routes.
7. **Runtime adapters are explicit.** LM Studio, Ollama, MLX-LM, llama.cpp, LocalAI, and generic OpenAI-compatible backends report capability differences honestly.

## Milestone overview

### M0 — Shared admin backend contract

Extract settings state and actions into reusable modules.

Target modules:

```text
hermes/plugins/model_router/admin/__init__.py
hermes/plugins/model_router/admin/state.py
hermes/plugins/model_router/admin/actions.py
hermes/plugins/model_router/admin/config_edit.py
hermes/plugins/model_router/admin/supervisor.py
hermes/plugins/model_router/admin/model_library.py
hermes/plugins/model_router/admin/runtime_adapters.py
hermes/plugins/model_router/admin/downloads.py
hermes/plugins/model_router/admin/benchmarks.py
hermes/plugins/model_router/admin/telemetry.py
```

Acceptance:

- Existing `model-router settings` still works.
- `build_settings_state(...)` behavior is preserved through a compatibility wrapper.
- State JSON includes `routing_mode`, `decision_layer_enabled`, `model_aliases`, `model_library`, `runtime_capabilities`, and `admin_actions`.
- Tests assert every visible web UI field comes from state or an action result.

### M1 — Basic router mode

Implement `routing_mode` in proxy config and request forwarding.

Acceptance:

- Decision mode remains default and backwards-compatible.
- `manual`, `model_map`, and `passthrough` modes avoid `route_fast(...)`.
- Headers and telemetry expose mode and decision-layer state.
- `/v1/models` returns model-router aliases and basic-mode models in a predictable way.
- Tests cover chat and Responses requests in all modes.
- Unknown model behavior is configurable as `fallback_to_default` or `reject_404`.

### M2 — Installer v1

Add `model-router install` as guided setup.

Acceptance:

- Detects install context: editable checkout, uv tool, pipx, pip.
- Runs config init using existing safe product setup.
- Detects LM Studio, Ollama, MLX-LM, llama.cpp, LocalAI-like endpoints, Python version, port availability, and optional dependencies.
- Offers explicit next actions: start proxy, open settings UI, launch TUI, run doctor, plan downloads.
- Does not silently download models, install services, enable hosted APIs, or mutate shell profiles.

### M3 — Model Library / Discover UI

Add a proper model marketplace-like page without copying LM Studio.

Acceptance:

- Installed tab is backed by local scans and runtime adapters.
- Discover tab is backed by curated catalog first, optional Hugging Face search second.
- Recommended tab uses hardware-aware and route-aware scoring.
- Assignments tab edits route/backend/model mappings and aliases through confirmed save actions.
- Downloads have plan, confirm, run, progress/status, retry, and failure states.

### M4 — Runtime Manager adapters

Add runtime adapter abstraction.

Acceptance:

- LM Studio adapter: discover/list loaded/load/unload/server status via REST and/or `lms` when available.
- Ollama adapter: discover/list loaded/pull tags/generate health through Ollama API.
- MLX-LM and llama.cpp adapters wrap existing managed runtime manager.
- Generic OpenAI-compatible adapter reports limited capabilities honestly.
- UI/TUI show capability gaps instead of pretending controls work.

### M5 — TUI v1, fully wired

Add `model-router tui` using the same admin state/actions as web settings.

Acceptance:

- TUI has working tabs for Status, Models, Routing, Runtimes, Telemetry, Logs, Settings.
- No placeholder rows; empty states explain which backend state is missing.
- Mutating actions use confirmation modals.
- TUI refreshes state after actions.
- If Textual is missing, command gives exact install hint and exits cleanly.

### M6 — API compatibility expansion

Add endpoints and capability routing.

Acceptance:

- `/v1/embeddings` routes to embedding/RAG backend or model alias.
- `/v1/completions` forwards legacy completions.
- `/v1/messages` supports Anthropic-compatible clients enough for Claude-style local workflows.
- Capabilities table informs endpoint availability, tool support, structured output, vision, embeddings, max context, and streaming.

### M7 — Polish and release maturity

Move features through maturity gates.

Acceptance:

- Stable config migration path.
- Docs updated.
- Dogfood check covers decision mode and all basic router modes.
- UI screenshots or golden render snapshots are checked in for major tabs.
- Upgrade/uninstall flows are documented.

## Suggested implementation order

```text
1. M0 shared admin backend
2. M1 basic router mode
3. M2 installer v1
4. M3 model library UI
5. M5 TUI read/write
6. M4 runtime adapters in priority order
7. M6 API parity
8. M7 maturity polish
```

Basic router mode should happen early because it affects proxy config, telemetry, UI state, TUI state, and mental model. The installer and UI should launch with a clear mode selector instead of retrofitting it later.
