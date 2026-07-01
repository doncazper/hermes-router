# Runtime Adapter Contract

This document defines the product-level contract for ModelRouter runtime
adapters. It aligns with `docs/runtime-strategy.md`: adapters coordinate proven
runtimes, expose honest capabilities, and keep runtime management out of the
routing hot path.

This is not the host-agent adapter contract in `docs/adapter-contract.md`.
Runtime adapters describe model server/runtime surfaces such as LM Studio,
Ollama, llama.cpp, MLX-LM, LocalAI, vLLM, and hosted OpenAI-compatible
providers. Host agents still own task execution, tools, context, delegation,
and final review.

## Contract Goals

- Support partial runtime capabilities without pretending every backend behaves
  the same.
- Make unsupported operations first-class disabled states with clear reasons.
- Keep `route_fast(...)`, `route(...)`, and default proxy forwarding free of
  runtime discovery dependencies.
- Let settings UI, CLI, TUI, admin actions, setup assistant, model registry,
  health checks, and recommendations use the same JSON-safe runtime state.
- Preserve provider/runtime neutrality and avoid ModelRouter-only lock-in.

## RuntimeAdapter Shape

The target contract is a small, bounded object around one runtime family or
provider surface:

```python
class RuntimeAdapter(Protocol):
    id: str
    display_name: str
    runtime_kind: Literal["external_managed", "external_cli", "bundled_future"]
    privacy_notes: tuple[str, ...]
    safety_notes: tuple[str, ...]

    def detect(self) -> RuntimeDetection: ...
    def install_status(self) -> RuntimeInstallStatus: ...
    def health(self, *, timeout_seconds: float = 0.25) -> RuntimeHealth: ...
    def start(self) -> RuntimeActionResult: ...
    def stop(self) -> RuntimeActionResult: ...
    def list_models(self, *, timeout_seconds: float = 0.25) -> tuple[RuntimeModel, ...]: ...
    def list_loaded_models(self, *, timeout_seconds: float = 0.25) -> tuple[RuntimeModel, ...]: ...
    def load_model(self, model_id: str) -> RuntimeActionResult: ...
    def unload_model(self, model_id: str) -> RuntimeActionResult: ...
    def endpoint(self) -> RuntimeEndpoint: ...
    def capabilities(self) -> RuntimeCapabilities: ...
    def unsupported_operation_reason(self, operation: str) -> str | None: ...
```

Implementation note: the current Python implementation uses `runtime_kind` in
some state for runtime family strings such as `lmstudio`, `ollama`, and
`llama-server`. The product contract above uses `runtime_kind` for the ownership
mode from `docs/runtime-strategy.md`. A future code cleanup should either add a
separate ownership-mode field or rename family identity to `runtime_family` in
JSON state. Until then, docs and UI should avoid treating family identity as
ownership semantics.

## Core State Types

### `RuntimeDetection`

```json
{
  "adapter_id": "lmstudio",
  "display_name": "LM Studio",
  "runtime_kind": "external_managed",
  "detected": true,
  "installed": true,
  "available": true,
  "endpoint": "http://127.0.0.1:1234/v1",
  "version": null,
  "command": ["lms"],
  "detail": "LM Studio endpoint configured; health check determines reachability."
}
```

Detection is read-only and bounded. It may inspect config, PATH, expected local
ports, or a configured health endpoint. It must not install packages, download
models, start servers, or mutate runtime state.

### `RuntimeInstallStatus`

```json
{
  "installed": true,
  "install_source": "operator",
  "version": "0.0.0-or-null",
  "update_available": null,
  "detail": "Installed externally; ModelRouter will not update it."
}
```

Install status is advisory. For `external_managed` and most `external_cli`
runtimes, ModelRouter reports what it can see and leaves installation and
updates to the operator. For `bundled_future`, this can report packaged runtime
provenance and version metadata.

### `RuntimeHealth`

```json
{
  "status": "ready",
  "reachable": true,
  "ok": true,
  "checked_url": "http://127.0.0.1:1234/v1/models",
  "status_code": 200,
  "detail": "configured model listed by runtime"
}
```

Health is best-effort. It should prefer standard OpenAI-compatible `/v1/models`
or documented local status endpoints. Failures are state, not crashes.

### `RuntimeEndpoint`

```json
{
  "base_url": "http://127.0.0.1:1234/v1",
  "auth_required": false,
  "local": true,
  "openai_compatible": true,
  "notes": ["ModelRouter forwards requests; the runtime owns execution."]
}
```

Endpoint metadata must not expose secrets. Auth presence is fine; token values
are not.

### `RuntimeActionResult`

```json
{
  "ok": false,
  "status": "unsupported",
  "message": "LM Studio native lifecycle commands are not wired yet.",
  "disabled_reason": "Stable local CLI/API lifecycle contract is not confirmed."
}
```

Unsupported operations should return `status: "unsupported"` with a
`disabled_reason`. They should not be generic exceptions in operator surfaces.
Mutating operations require confirmation before the adapter method is invoked.

## Capability Model

Each capability should be an object, not a bare boolean:

```json
{
  "status": "supported",
  "disabled_reason": null,
  "source": "openai_compatible",
  "confidence": "documented",
  "notes": []
}
```

Allowed `status` values:

- `supported`: ModelRouter knows how to expose or preserve this capability for
  the runtime.
- `partial`: ModelRouter can preserve or report the shape, but actual behavior
  depends on runtime version, loaded model, or server flags.
- `unsupported`: The runtime or adapter does not expose this operation.
- `deferred`: Product direction exists, but the adapter does not implement it.
- `unknown`: ModelRouter has not checked and should not pretend.

Required capability keys:

- `chat_completions`
- `responses`
- `embeddings`
- `streaming`
- `tool_calls`
- `structured_output`
- `vision`
- `mcp`
- `model_download`
- `model_load`
- `model_unload`
- `parallel_requests`
- `local_only`

Lifecycle/action support should also be visible:

- `detect`
- `install_status`
- `health`
- `start`
- `stop`
- `list_models`
- `list_loaded_models`
- `load_model`
- `unload_model`
- `logs`

The implementation may keep endpoint compatibility and lifecycle support in
separate structs, but admin/state surfaces should preserve both. A runtime can
support chat completions while not supporting model load/unload; that is normal
and should render as a disabled control with a reason.

## Operation Rules

- `detect()`, `install_status()`, `health()`, `endpoint()`,
  `capabilities()`, `list_models()`, `list_loaded_models()`, and `logs()` are
  read-only and bounded.
- `start()`, `stop()`, `load_model()`, and `unload_model()` are mutating and
  must require explicit operator confirmation before invocation.
- `model_download` is a capability, not permission to download. Download or pull
  actions must be separate, explicit, previewed, and confirmation-gated.
- Streaming responses must not be buffered just to discover usage or runtime
  metadata.
- Runtime discovery, health, installed state, and model lists must not change
  `route_fast(...)` or `route(...)` decisions.
- Proxy forwarding may use configured backend URLs/models and managed-process
  config, but it must not require the optional adapter contract to exist.

## Unsupported Operations

Unsupported behavior is product data:

```json
{
  "operation": "load_model",
  "supported": false,
  "disabled_reason": "OpenAI-compatible runtimes do not expose a standard load action.",
  "operator_hint": "Load the model in the runtime app, or choose a runtime adapter that supports explicit load."
}
```

Use this shape for disabled buttons, CLI output, admin API responses, and TUI
state. Avoid generic messages such as "failed" when the real state is "not
supported by this adapter."

## Privacy And Safety Notes

Every adapter should expose short notes suitable for UI/admin surfaces.

Privacy notes should cover:

- Whether the endpoint is local or hosted.
- Whether auth secrets are configured but hidden.
- Whether model discovery calls leave the machine.
- Whether logs are local and whether they may contain runtime output.

Safety notes should cover:

- Which operations mutate runtime state.
- Whether actions can start processes, unload models, or contact hosted
  providers.
- Whether model downloads/pulls are separate confirmation-gated actions.
- Whether the runtime can execute tools or MCP operations itself.

These notes are advisory and do not replace ModelRouter route receipts or
safety gates.

## Example Adapters

### LM Studio

```json
{
  "id": "lmstudio",
  "display_name": "LM Studio",
  "runtime_kind": "external_managed",
  "endpoint": {"base_url": "http://127.0.0.1:1234/v1", "local": true},
  "capabilities": {
    "chat_completions": {"status": "supported"},
    "responses": {"status": "partial"},
    "embeddings": {"status": "partial"},
    "streaming": {"status": "partial"},
    "tool_calls": {"status": "partial"},
    "structured_output": {"status": "partial"},
    "vision": {"status": "unknown"},
    "mcp": {"status": "deferred"},
    "model_download": {"status": "unsupported", "disabled_reason": "LM Studio owns downloads."},
    "model_load": {"status": "unsupported", "disabled_reason": "Stable native lifecycle contract is not wired."},
    "model_unload": {"status": "unsupported", "disabled_reason": "Stable native lifecycle contract is not wired."},
    "parallel_requests": {"status": "unknown"},
    "local_only": {"status": "supported"}
  }
}
```

LM Studio owns model execution, search, downloads, and native loading unless a
future stable local API/CLI contract is explicitly wired. ModelRouter can route
to its OpenAI-compatible endpoint and report health/model ids.

### Ollama

```json
{
  "id": "ollama",
  "display_name": "Ollama",
  "runtime_kind": "external_managed",
  "endpoint": {"base_url": "http://127.0.0.1:11434/v1", "local": true},
  "capabilities": {
    "chat_completions": {"status": "supported"},
    "responses": {"status": "partial"},
    "embeddings": {"status": "partial"},
    "streaming": {"status": "partial"},
    "tool_calls": {"status": "partial"},
    "structured_output": {"status": "partial"},
    "vision": {"status": "partial"},
    "mcp": {"status": "unsupported"},
    "model_download": {"status": "deferred", "disabled_reason": "Pull/download must be explicit and separate."},
    "model_load": {"status": "unsupported", "disabled_reason": "Ollama run/load and pull/download are separate actions."},
    "model_unload": {"status": "partial", "disabled_reason": "Requires ollama CLI for `ollama stop <model>`."},
    "parallel_requests": {"status": "unknown"},
    "local_only": {"status": "supported"}
  }
}
```

Ollama may be externally managed by its app/service or controlled through its
CLI when available. ModelRouter must not pull models or start global services
silently.

### llama.cpp Server

```json
{
  "id": "llamacpp",
  "display_name": "llama.cpp server",
  "runtime_kind": "external_cli",
  "endpoint": {"base_url": "http://127.0.0.1:8090/v1", "local": true},
  "capabilities": {
    "chat_completions": {"status": "supported"},
    "responses": {"status": "partial"},
    "embeddings": {"status": "partial"},
    "streaming": {"status": "partial"},
    "tool_calls": {"status": "partial"},
    "structured_output": {"status": "partial"},
    "vision": {"status": "partial"},
    "mcp": {"status": "unsupported"},
    "model_download": {"status": "unsupported", "disabled_reason": "Model files are managed by the operator."},
    "model_load": {"status": "partial", "disabled_reason": "Starting the configured process loads the configured model."},
    "model_unload": {"status": "partial", "disabled_reason": "Stopping the configured process unloads the model."},
    "parallel_requests": {"status": "partial"},
    "local_only": {"status": "supported"}
  }
}
```

For configured managed processes, ModelRouter may start/stop the exact argv
declared in `routing_proxy.yaml` after confirmation or proxy demand-start rules.
Operator-triggered start writes a ModelRouter-owned PID marker; stop/unload only
targets that marked process and leaves externally started runtimes alone. It
must not infer arbitrary model paths, kill unrelated processes, or download GGUF
files.

### MLX-LM

```json
{
  "id": "mlx_lm",
  "display_name": "MLX-LM",
  "runtime_kind": "external_cli",
  "endpoint": {"base_url": "http://127.0.0.1:8091/v1", "local": true},
  "capabilities": {
    "chat_completions": {"status": "supported"},
    "responses": {"status": "unsupported", "disabled_reason": "Managed MLX-LM support is chat/models-first."},
    "embeddings": {"status": "unsupported", "disabled_reason": "Managed MLX-LM support is chat/models-first."},
    "streaming": {"status": "partial"},
    "tool_calls": {"status": "partial"},
    "structured_output": {"status": "partial"},
    "vision": {"status": "unsupported"},
    "mcp": {"status": "unsupported"},
    "model_download": {"status": "deferred", "disabled_reason": "Downloads must be planned and confirmed separately."},
    "model_load": {"status": "partial", "disabled_reason": "Starting mlx_lm.server loads the configured model."},
    "model_unload": {"status": "partial", "disabled_reason": "Stopping the configured process unloads the model."},
    "parallel_requests": {"status": "unknown"},
    "local_only": {"status": "supported"}
  }
}
```

MLX-LM is a good Apple Silicon target, but it remains an external CLI runtime
until a future bundled path exists.

### Hosted OpenAI-Compatible Backend

```json
{
  "id": "hosted_openai_compatible",
  "display_name": "Hosted OpenAI-compatible backend",
  "runtime_kind": "external_managed",
  "endpoint": {"base_url": "https://provider.example/v1", "local": false},
  "capabilities": {
    "chat_completions": {"status": "supported"},
    "responses": {"status": "partial"},
    "embeddings": {"status": "partial"},
    "streaming": {"status": "partial"},
    "tool_calls": {"status": "partial"},
    "structured_output": {"status": "partial"},
    "vision": {"status": "partial"},
    "mcp": {"status": "unknown"},
    "model_download": {"status": "unsupported", "disabled_reason": "Hosted provider owns model availability."},
    "model_load": {"status": "unsupported", "disabled_reason": "Hosted provider owns model loading."},
    "model_unload": {"status": "unsupported", "disabled_reason": "Hosted provider owns model lifecycle."},
    "parallel_requests": {"status": "partial"},
    "local_only": {"status": "unsupported", "disabled_reason": "Requests leave the machine."}
  }
}
```

Hosted adapters should not probe availability by default unless explicitly
configured. They must hide secrets and make network/privacy implications visible
in `privacy_notes`.

## Current Implementation Alignment

The current runtime implementation already has the right skeleton:

- `RuntimeAdapter` protocol for endpoint, capabilities, detection, health,
  model discovery, load/unload, start/stop, and logs.
- `AdapterSupport` and `RuntimeCapabilities` for first-class disabled reasons.
- Runtime-specific adapters for generic OpenAI-compatible endpoints, LM Studio,
  Ollama, LocalAI, vLLM, and configured managed runtimes.
- Confirmed start/stop/load/unload controls for configured managed CLI runtimes
  when the declared command is available, with unsupported reasons when the
  command or capability is missing.
- Runtime detection reports include `runtime_id`, family `runtime_kind`,
  ownership `runtime_mode`, detected state, endpoint, optional version,
  health status, missing dependency, install hint, and `last_checked_at`.
- Admin/CLI/settings actions that require confirmation for mutating runtime
  actions.
- Regression tests proving runtime adapters are optional for routing/proxy hot
  paths.

Known contract gaps to close later:

- Add explicit `display_name`, `privacy_notes`, and `safety_notes`.
- Split ownership mode from runtime family identity.
- Extend capabilities from lifecycle-only support toward endpoint capability
  objects with `supported`/`partial`/`unsupported`/`deferred`/`unknown` states.
- Add `install_status()` as a read-only advisory method.
- Standardize `endpoint()` metadata beyond a bare URL.
- Add a single `unsupported_operation_reason(operation)` helper for UI/CLI/TUI.

These are contract/shape improvements. They should not change routing policy,
`route_fast(...)`, `route(...)`, or default proxy forwarding behavior.
