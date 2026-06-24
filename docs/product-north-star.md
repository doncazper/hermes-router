# Product North Star

This is the current ModelRouter product north star:

![ModelRouter product north star](assets/model-router-north-star.png)

The image is directional product truth. It defines the product shape we are
building toward, not a claim that every visible control or panel is fully
implemented today.

## Product Identity

ModelRouter is a local OpenAI-compatible proxy router and admin control center.
It is not an agent, not a webchat UI, and not a prompt transcript product.

The product should make routing observable and controllable:

- A local endpoint is visible and copyable.
- Proxy status, selected mode, and telemetry status are always obvious.
- Routing policy is expressed in plain-language modes: `Fast`, `Balanced`,
  `Quality`, `Private`, and `Safe`.
- Route classes, provider/runtime choices, latency, cost, privacy, tool needs,
  fallback paths, and rejected routes are inspectable without reading YAML.
- Local and hosted provider boundaries remain explicit.
- Safety gates and `human_confirm` behavior are visible and conservative.
- Receipts explain what happened, why, and how to label a wrong route.
- Telemetry supports dogfooding without exposing private prompt text by default.

## Intended Surface

The main settings surface should feel like a local proxy control center:

- Local-only admin/config UI started by `model-router settings`.
- Proxy status and endpoint visibility.
- Mode controls for `Fast`, `Balanced`, `Quality`, `Private`, and `Safe`.
- Request flow from incoming request to ModelRouter decision, selected engine,
  backend runtime, and response.
- Routing map/table with route classes, route ids, target descriptions,
  providers/runtimes, latency, cost, privacy, tools, and fallback behavior.
- Provider/runtime panel for llama.cpp, Ollama, LM Studio, MLX-LM,
  OpenAI-compatible providers, and related local or custom backends.
- Runtime command, model path, port, context, readiness, idle-timeout, start,
  stop, restart, and log controls where the runtime is managed.
- Route receipt panel showing selected engine, backend, model, rationale, risk,
  tools, fallback, rejected routes, confirmation state, latency, privacy, and
  receipt JSON.
- Safety panel for human-confirm gates.
- Recent requests/telemetry panel with wrong-route feedback entry points.
- Compact "Mini" monitor concept for status, recent routes, receipts,
  providers, and safety without turning the product into chat.

## Implemented Today

The current product already includes:

- Deterministic routing through `ModelRouter.route_fast(...)` and receipt-rich
  `ModelRouter.route(...)`.
- OpenAI-compatible proxy endpoints for supported request shapes.
- Local settings UI through `model-router settings`.
- Data-backed settings dashboard panels for the latest route receipt, configured
  routing map, provider/runtime config, recent telemetry, feedback labels,
  benchmark status, and proxy process controls.
- Visual proxy config editing for profile, observability, backend policy, and
  per-route backend/runtime fields with explicit Save/Apply/Restart actions.
- Response headers, receipts, and telemetry workflows that make route ids easier
  to identify and label.
- Opt-in managed local runtimes for configured llama.cpp and MLX-LM processes.

## In Progress

The north star assumes continued polish around:

- Richer route-map editing and profile/provider policy controls in the settings
  UI.
- More complete visual runtime status, readiness, logs, and managed-runtime
  controls.
- Continued dogfooding of the recent-request and wrong-route review loops before
  adding heavier review surfaces.
- Broader dogfood evidence from real local runtimes and benchmark-backed setup.

## Future Direction

Future UI work should align with the screenshot when it improves the proxy
control-center experience. It should not add:

- A chat prompt box.
- Agent behavior.
- A prompt transcript surface.
- Silent model downloads.
- Silent config or routing-policy mutation.
- Silent hosted-provider enablement.
- Raw prompt display unless prompt capture was explicitly configured and the UI
  marks the data as sensitive.

Downloads, config writes, benchmark runs, hosted-provider use, and proxy/runtime
process changes should remain explicit user actions.
