# Runtime Strategy

ModelRouter's runtime strategy is adapter-first and control-plane-first.
ModelRouter should own routing policy, local operator UX, receipts, telemetry,
safety gates, and explicit lifecycle controls where they are safe. It should
not own low-level model kernels, provider-native execution semantics, or a
custom inference engine when proven runtimes already exist.

The goal is practical LM Studio-level local workflow parity without cloning LM
Studio internals or locking users into ModelRouter-only model storage. Operators
should be able to use ModelRouter above LM Studio, alongside LM Studio, or
eventually without LM Studio for common local-model workflows.

## Runtime Modes

### `external_managed`

An `external_managed` runtime is already managed by another app, service, or
daemon. Examples include LM Studio with its local server enabled, an existing
Ollama daemon, a LocalAI service run by the operator, a vLLM deployment, or a
hosted OpenAI-compatible gateway.

ModelRouter may:

- Detect the runtime from config, local conventions, or bounded health checks.
- Health-check configured endpoints.
- Import metadata such as visible model ids, loaded-model state when exposed,
  capability flags, and disabled-operation reasons.
- Route traffic to the configured endpoint through the local proxy.
- Record receipts, telemetry, usage, catalog coverage, and outcome feedback.
- Show operator guidance for actions that remain owned by the external app or
  deployment supervisor.

ModelRouter must not:

- Pretend to own model execution.
- Silently start, stop, load, unload, pull, or install anything.
- Route based on live discovery side effects.
- Require the runtime adapter for `route_fast(...)`, `route(...)`, or ordinary
  proxy forwarding.

This mode is the default compatibility stance for LM Studio. LM Studio can keep
owning model search, downloads, loading, chat, and native runtime behavior,
while ModelRouter adds policy, routing, receipts, safety gates, telemetry, and a
stable local `/v1` switchboard above it.

### `external_cli`

An `external_cli` runtime is installed by the user and can be controlled through
a local command, API, or service interface. Examples include `llama-server`,
`ollama`, `mlx_lm.server`, LocalAI, and vLLM where the operator has installed
and configured the runtime.

ModelRouter may:

- Detect whether the expected command exists.
- Validate configured argv-style runtime commands.
- Start or stop configured managed processes when the backend explicitly opts
  into ModelRouter management.
- Call stable CLI/API actions such as list models, list loaded models, load, or
  unload only when the adapter declares support.
- Surface logs, readiness URLs, ports, context settings, and unsupported-action
  reasons.
- Require confirmation for mutating actions.

ModelRouter must not:

- Install system packages silently.
- Pull/download models silently.
- Mutate global runtime state without an explicit operator action.
- Invent provider-specific behavior outside the adapter contract.
- Convert runtime status into an implicit routing signal.

This mode is the near-term workhorse. It gives users an integrated control
center for common local operations while respecting the fact that llama.cpp,
Ollama, MLX-LM, LocalAI, and vLLM own their own execution semantics.

### `bundled_future`

A `bundled_future` runtime is a future convenience layer where ModelRouter
packages or vendors a proven runtime for easier first-run setup. The likely
order is llama.cpp first, then MLX/MLX-LM for Apple Silicon if the integration
is stable and maintainable.

ModelRouter may eventually:

- Package a known runtime binary or managed dependency with clear provenance.
- Provide a simpler first-run path for local chat/completions through that
  runtime.
- Manage runtime version metadata, compatibility warnings, logs, and lifecycle
  state.
- Keep the same adapter contract used by external runtimes.

ModelRouter still must not:

- Become a low-level inference kernel project.
- Hide runtime provenance, version, or capability limits.
- Make bundled runtime use mandatory.
- Break compatibility with LM Studio, Ollama, LocalAI, llama.cpp, MLX/MLX-LM,
  vLLM, hosted providers, or generic OpenAI-compatible servers.

Bundling should be treated as packaging and UX polish, not a new product
identity. The durable advantage remains control, policy, receipts, safety, and
telemetry across runtimes.

## Why Adapter-First Comes First

Adapter-first is the right first move because it preserves ModelRouter's core
identity while moving toward local-app parity:

- It keeps external runtimes first-class instead of forcing migration.
- It lets ModelRouter support LM Studio, Ollama, LocalAI, llama.cpp, MLX-LM,
  vLLM, hosted providers, and unknown OpenAI-compatible servers with honest
  capability boundaries.
- It keeps routing deterministic: runtime discovery and health are operator
  diagnostics, not hidden route-decision inputs.
- It allows small, testable slices: detect, health, list models, list loaded
  models, start/stop, load/unload, logs, and disabled reasons.
- It avoids committing ModelRouter to low-level GPU, quantization, kernel,
  scheduler, and platform-specific inference work before the control center is
  proven.

The adapter contract also gives the UI and CLI a stable shape: supported
actions are enabled, unsupported actions are disabled with reasons, and mutating
actions require confirmation. The target product contract is defined in
`docs/runtime-adapter-contract.md`.

## Why Bundled Runtimes Come Later

Bundled runtimes can improve first-run convenience, but they are expensive to
own operationally. They introduce binary distribution, platform differences,
GPU/CPU capability detection, signing/notarization questions, update cadence,
security patching, model-format drift, and user support load.

Those costs are worth considering only after the adapter path is solid. By
building adapters first, ModelRouter can learn which runtime controls operators
actually use, which backends are reliable, which logs and errors matter, and
which packaging gaps block adoption. Bundling can then reuse the same adapter
contract instead of creating a second execution path.

## LM Studio Parity Without Cloning LM Studio

LM Studio parity means common local-model workflows feel complete:

- Discover or import models.
- See what is installed and what is loaded.
- Start a local endpoint.
- Load or unload models where the runtime exposes that action.
- Test and route through an OpenAI-compatible API.
- Understand capability gaps.

ModelRouter should reach those workflows through its own strengths: route-aware
recommendations, provider-neutral backends, policy controls, receipts,
telemetry, safety gates, catalog coverage, and outcome feedback. It should not
copy LM Studio's internals, make superiority claims, or pretend LM Studio is an
enemy. LM Studio can remain the best surface for users who prefer its search,
chat, download, and native runtime tools.

## Above, Alongside, Or Without LM Studio

ModelRouter should support three operator choices:

- **Above LM Studio**: LM Studio manages local models and exposes a local server;
  ModelRouter routes to it and adds policy, receipts, telemetry, and safety.
- **Alongside LM Studio**: ModelRouter manages some routes through other
  runtimes such as Ollama, llama.cpp, MLX-LM, LocalAI, vLLM, or hosted
  providers, while LM Studio remains available for selected models or chat.
- **Without LM Studio**: ModelRouter uses external CLI runtimes or, later,
  bundled runtimes to provide enough local model discovery, lifecycle, endpoint,
  and routing UX for common workflows.

All three paths should share the same routing policy, model registry shape,
runtime capability model, receipts, telemetry, and safety gates.

## Non-Goals

- No custom low-level inference engine now.
- No silent install, pull, download, config mutation, hosted-provider enablement,
  or runtime lifecycle action.
- No route decisions based on live runtime discovery.
- No dependency on runtime adapters for `route_fast(...)`, `route(...)`, or
  default proxy forwarding.
- No lock-in to ModelRouter-only runtimes or model storage.
- No benchmark, cost, or performance claims without checked-in evidence and
  clear scope.

## Implementation Implications

- Runtime config should identify whether a backend is externally managed,
  controlled through an explicit CLI/API adapter, or eventually bundled.
- Adapter state should be JSON-safe and privacy-safe.
- Settings, CLI, TUI, and admin APIs should use the same capability contract.
- Read-only status can be bounded and best-effort; mutating actions require
  confirmation.
- Unsupported operations should be product-visible disabled states, not generic
  errors.
- Hot-path tests should keep proving that runtime adapter failures do not affect
  routing or proxy forwarding.
