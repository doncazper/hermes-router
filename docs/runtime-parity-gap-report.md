# Runtime Parity Gap Report

This report summarizes where ModelRouter stands relative to common local-model
workflows, especially the workflows users expect from LM Studio. It is a
product and implementation gap report, not a benchmark claim and not a plan to
clone LM Studio internals.

The durable ModelRouter advantage is control: provider/runtime policy,
OpenAI-compatible proxy routing, route receipts, local-first telemetry,
cost/outcome/catalog reporting, safety gates, and multi-runtime coordination.
The parity goal is to make common local-model operations approachable without
giving up that control-plane identity.

## What ModelRouter Can Do Today

ModelRouter currently provides:

- A local AI control center and routing/control plane.
- OpenAI-compatible proxy routing over configured local and hosted backends.
- Deterministic `route_fast(...)` and richer `route(...)` routing APIs.
- Route receipts, reason codes, delegation suitability signals, safety gates,
  and fallback policy.
- Privacy-safe telemetry with latency, usage tokens when upstreams report them,
  manual outcome labels, pricing catalog reporting, and catalog coverage/cost
  confidence.
- Packaged and user-overridable pricing/model catalogs with explicit
  maintenance commands.
- Settings UI, compact mode, TUI direction, and admin-state surfaces for proxy,
  routing, models, runtimes, telemetry, catalog coverage, and policy status.
- Model registry support for packaged, user-declared, local-scan, and
  runtime-imported model metadata.
- Guided runtime connect/install planning for LM Studio, Ollama, and configured
  llama.cpp-style endpoints, with preview-first config writes.
- Runtime adapters that report endpoint, detection, health, model listing,
  loaded-model listing where available, capability flags, logs, and disabled
  reasons.
- Explicit runtime lifecycle actions for configured managed CLI runtimes when
  ModelRouter owns the process marker.

These surfaces are designed so routing and proxy forwarding remain usable even
when optional runtime adapters, setup guidance, or settings UI features fail.

## Adapter-Supported Today

### LM Studio

Supported shape:

- Detect configured LM Studio local-server endpoints.
- Health-check and list visible models through bounded OpenAI-compatible
  endpoints when available.
- Import visible model ids into the registry as runtime-discovered metadata.
- Route requests through the ModelRouter proxy to LM Studio's
  OpenAI-compatible server.
- Show lifecycle actions as unsupported unless a stable native LM Studio
  CLI/API contract is wired.

ModelRouter does not own LM Studio downloads, search, local chat, native model
loading, or runtime execution.

### Ollama

Supported shape:

- Detect Ollama on PATH and/or the local server endpoint.
- Use bounded local API or CLI paths for model listing where available.
- Report loaded models where the CLI/API exposes them.
- Route through Ollama's OpenAI-compatible endpoint when configured.
- Support explicit unload through `ollama stop <model>` only when the adapter
  confirms the operation is available.

ModelRouter does not silently run `ollama pull`, start global services, or
treat missing tags as a reason to download.

### llama.cpp And MLX-LM Managed CLI Runtimes

Supported shape:

- Represent configured runtime commands as `external_cli` managed processes.
- Start and stop the exact configured argv only through explicit operator
  actions or configured proxy-managed runtime behavior.
- Write PID markers for ModelRouter-owned processes.
- Stop or unload only the marked ModelRouter-owned process.
- Import configured local model paths and visible runtime model ids when
  available.
- Surface logs, readiness URLs, health, and disabled reasons.

This does not bundle binaries, infer arbitrary model paths, download model
files, or kill unrelated user processes.

### LocalAI, vLLM, Generic OpenAI-Compatible, And Hosted Backends

Supported shape:

- Treat configured endpoints as OpenAI-compatible backends.
- Report runtime/provider identity when config hints identify LocalAI or vLLM.
- Health-check and list models through bounded `/v1/models`-style discovery.
- Surface lifecycle load/unload/start/stop as unsupported unless an adapter has
  a stable tested contract.
- Preserve hosted-provider privacy and auth boundaries by hiding secrets and
  avoiding default network probes unless explicitly configured.

## Installer-Guided Only

The guided setup surface is intentionally conservative:

- LM Studio: connect instructions, local-server health check, config preview,
  and registry refresh after the operator starts LM Studio.
- Ollama: install/connect guidance, PATH/server checks, config preview, and
  separate model-pull guidance.
- llama.cpp server: endpoint/configure guidance and managed argv config preview.
- MLX-LM: Apple Silicon guidance and local Python/server setup design.
- LocalAI: connect-first guidance, container/app install references, config
  preview, and rollback notes.
- vLLM: advanced connect-first guidance, isolated environment suggestions, API
  key redaction, and server deployment warnings.

The installer does not silently install packages, write config, pull models,
start services, enable hosted providers, or change routing policy.

## Future Or Bundled Only

These remain future work:

- Packaged runtime binaries.
- Signed/notarized desktop runtime helpers.
- Runtime update channels and rollback.
- Built-in model download manager for bundled runtimes.
- Platform-specific bundled acceleration paths.
- Bundled llama.cpp distribution.
- Bundled MLX or MLX-LM path for Apple Silicon.
- Sandboxing/resource policy beyond conservative process launch.
- Native bundled-runtime installer UI.
- Service registration and startup-item management.

The future bundled path should reuse the adapter contract. It should remain
optional and should never become a routing hot-path dependency.

## What LM Studio Still Does Better

LM Studio is still ahead for a polished individual local-model app experience:

- Integrated model search and discovery.
- GUI-first model download and local storage management.
- Native local chat and playground workflows.
- Familiar model load/unload controls inside the same app that owns execution.
- Runtime and model settings tuned for LM Studio's own server.
- A cohesive desktop UX for users who mainly want to download, load, chat, and
  expose one local server.
- Native LM Studio CLI and REST surfaces for LM Studio-owned lifecycle actions.
- Local-model onboarding that does not require thinking in routing policies or
  backend config.

ModelRouter should respect that strength. If LM Studio already handles a user's
model app workflow well, ModelRouter should sit above or alongside it and add
policy, routing, receipts, telemetry, safety, and governance surfaces.

## Where ModelRouter Is Already Stronger

ModelRouter is already stronger as a provider-neutral control plane:

- One local `/v1` routing endpoint over several local and hosted runtimes.
- Transparent route receipts and reason codes.
- Safety gates and policy controls.
- Local-first telemetry, feedback labels, and cost/catalog coverage.
- Manual outcome labels without inferred success claims.
- Pricing catalog maintenance without live pricing in routing.
- Runtime/provider neutrality across LM Studio, Ollama, llama.cpp, MLX-LM,
  LocalAI, vLLM, hosted providers, and generic OpenAI-compatible endpoints.
- Host-agent friendly control-plane primitives without hidden task
  orchestration.
- Explicit non-goals around no custom inference engine, no silent mutation, no
  live pricing in hot paths, and no ModelRouter-only runtime lock-in.

## Gap Groups

### Polish

- Runtime status copy should become more concise and action-oriented.
- Settings UI should refresh runtime state after actions without implying that
  rendering the UI mutates runtime state.
- Runtime logs need a compact, privacy-aware viewing surface.
- Disabled action reasons should be consistent across CLI, settings, TUI, and
  admin JSON.
- Guided setup should expose clearer "why this next action" explanations.
- Compatibility and maturity labels should be visible but not visually heavy.

### Missing Backend Capability

- LM Studio native load/unload/start/stop remains unsupported until a stable
  local CLI/API contract is selected and tested.
- Ollama pull, run/load, serve/start, and stop/unload need careful separation so
  ModelRouter never downloads or starts services silently.
- LocalAI and vLLM lifecycle controls remain conservative because deployments
  are often containerized, remote, or supervisor-owned.
- `/v1/responses`, embeddings, tool calls, and structured output need deeper
  per-backend capability validation beyond passthrough.
- Model metadata depth varies by runtime: quantization, context length, format,
  source path, loaded state, and capabilities are not equally available.
- Hosted OpenAI-compatible backends should remain explicit about network and
  auth boundaries rather than mimicking local runtime behavior.

### Missing UI Wiring

- Runtime lifecycle controls need final compact UI affordances around confirm,
  success, unsupported, and refresh states.
- Guided runtime connect actions should be available from settings in the same
  preview-first shape as CLI.
- Model registry imports need clearer UI distinction between packaged,
  user-declared, local-scan, and runtime-imported entries.
- "Copy config patch" and "write config with confirmation" flows should be
  easy to find but not dominant in the main viewport.
- Runtime action results should link to logs and rollback guidance.
- TUI runtime controls should converge on the same admin action contract.

### Packaging And Distribution

- No bundled runtime binaries are packaged today.
- There is no runtime package manifest, checksum verification flow, update
  channel, or rollback command for bundled runtimes.
- macOS signing/notarization for helper binaries is not designed in detail.
- Windows installer, SmartScreen, firewall prompts, and service management are
  not implemented.
- Linux distribution strategy across tarballs, packages, containers, CUDA,
  ROCm, systemd, and permissions is unresolved.
- Release checks cover the Python package and routing/proxy maturity, but not a
  bundled binary matrix.

### Future Bundled Runtime

- llama.cpp is the likely first candidate, but only after external adapter and
  packaging readiness criteria are met.
- MLX-LM is a possible Apple Silicon convenience path after macOS packaging is
  proven.
- Bundled model storage, download planning, cache cleanup, and disk pressure UX
  need design and tests.
- Sandboxing, localhost binding, PID ownership, log handling, and uninstall
  behavior need to be productized before bundling.
- Bundled runtime use must remain optional and must not affect `route_fast(...)`,
  `route(...)`, or proxy forwarding unless the operator selected that backend.

## Recommended Next Implementation Order

1. Finish runtime action UX polish in CLI/settings/admin: confirmations,
   disabled reasons, post-action refresh, and log links.
2. Wire guided runtime connect flows into settings as preview-first actions.
3. Improve model registry display so imported models are clearly distinct from
   ModelRouter-owned or user-declared models.
4. Deepen real metadata import for stable adapters: context length,
   quantization, format, source, capabilities, and loaded state where available.
5. Harden endpoint capability reporting for `/v1/responses`, embeddings,
   streaming, tool calls, and structured output.
6. Run a packaging spike for one llama.cpp platform without shipping it as the
   default path.
7. Design runtime package manifests, checksum verification, rollback, and
   uninstall before any bundled runtime release.
8. Evaluate MLX-LM bundling only after the llama.cpp packaging path proves the
   process is supportable.

## Bundled Runtime Go/No-Go

Proceed toward bundled runtime only when:

- Adapter and installer-guided paths have shown a clear setup gap that bundling
  would solve.
- The bundled runtime can use the existing adapter contract.
- Packaging, signing, verification, update, rollback, and uninstall are tested
  for the target platform.
- Runtime provenance and licensing are clear.
- Hot-path tests prove no dependency from routing/proxy forwarding to bundled
  runtime maintenance.

Do not proceed when:

- The work would turn ModelRouter into a low-level inference engine.
- It would weaken external runtime support.
- It would require silent installs, model downloads, service mutation, or live
  route-decision side effects.
- The support matrix is too large to verify.

## Related Docs

- [Bundled runtime roadmap](bundled-runtime-roadmap.md)
- [Runtime strategy](runtime-strategy.md)
- [Runtime adapter contract](runtime-adapter-contract.md)
- [Runtime install flow](runtime-install-flow.md)
- [LM Studio parity roadmap](lm-studio-parity-roadmap.md)
- [Local server compatibility matrix](local-server-compatibility.md)
- [Release checklist](release-checklist.md)
