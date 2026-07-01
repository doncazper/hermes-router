# LM Studio Parity Roadmap

LM Studio is the floor, not the ceiling.

This roadmap defines how ModelRouter can replace LM Studio for common
local-model workflows while preserving its stronger identity as a local AI
control center and routing/control plane. The goal is parity by user workflow,
not cloning LM Studio internals or making superiority claims.

ModelRouter should work in three shapes:

- Standalone local control center for common local-model workflows.
- Compatibility layer above or alongside LM Studio, Ollama, LocalAI,
  llama.cpp, MLX/MLX-LM, vLLM, hosted providers, and generic
  OpenAI-compatible servers.
- Routing/control plane for host agents, which remain responsible for task
  execution, context management, delegation, tool use, and final review.

## Guardrails

- Keep routing/proxy hot paths deterministic and fast.
- Prefer adapters around proven runtimes over a custom inference engine.
- Preserve provider/runtime neutrality and avoid ModelRouter-only lock-in.
- Keep downloads, config writes, hosted-provider enablement, runtime starts,
  tool calls, and benchmark runs explicit.
- Do not fetch live pricing during `route_fast(...)`, `route(...)`, proxy
  forwarding, or default routing paths.
- Do not claim benchmark parity, cost reductions, frontier performance, or
  superiority over LM Studio without checked-in evidence and clear scope.
- Do not add hidden planner/worker orchestration. Fusion-like or other host
  harnesses may use ModelRouter for policy, receipts, telemetry, and safety
  gates, but they own orchestration.

## Baseline References

Use these as workflow references, not as UI-copying targets or performance
claims:

- LM Studio app docs (`https://lmstudio.ai/docs/app`): local chat,
  Hugging Face model search/download, MCP, OpenAI-like local endpoints,
  model/config management, offline document chat, and headless operation.
- LM Studio OpenAI compatibility docs
  (`https://lmstudio.ai/docs/developer/openai-compat`): `/v1/models`,
  `/v1/responses`, `/v1/chat/completions`, `/v1/embeddings`, and
  `/v1/completions`.
- LM Studio REST API docs (`https://lmstudio.ai/docs/developer/rest`): native
  chat, model list/load/unload/download, download status, stateful chats,
  authentication, MCP via API, and idle TTL.
- LM Studio CLI docs (`https://lmstudio.ai/docs/cli`): `lms chat`, `lms get`,
  `lms ls`, `lms ps`, `lms load`, `lms unload`, server start/stop/status,
  runtime management, and daemon commands.

## Priority Key

- **P0**: required for credible common-workflow parity.
- **P1**: important parity or differentiator after the core loop is usable.
- **P2**: useful expansion once the control center is stable.
- **Later**: primarily enterprise or ecosystem scale.

## Roadmap Matrix

### 1. Local Chat App / Test Workbench

- **LM Studio baseline**: desktop chat, terminal chat, chat history, local model
  testing, and document chat workflows.
- **ModelRouter current state**: proxy, settings UI, receipts, telemetry, and
  route decisions exist; the product intentionally does not center on chat.
- **Parity requirement**: add a lightweight workbench for testing a selected
  route/backend/model, streaming a response when supported, showing selected
  route, receipt summary, latency, and usage without becoming a consumer chat
  transcript product.
- **ModelRouter advantage opportunity**: make every test request explain policy,
  rejected routes, safety gates, backend/model selection, usage, catalog
  coverage, and feedback labels.
- **Priority**: P1.
- **Owner surface**: settings UI, proxy, CLI.

### 2. Model Discovery And Download

- **LM Studio baseline**: searchable model discovery, Hugging Face downloads,
  local imports, and visible download state.
- **ModelRouter current state**: local model scans, a JSON-safe known-model
  registry, curated discovery candidates, route-aware recommendations,
  hardware-aware download plans, and explicit download prompts.
- **Parity requirement**: make model discovery, local imports, download
  planning, download progress, installed-state detection, and route assignment
  feel like one coherent library.
- **ModelRouter advantage opportunity**: recommend models by route fit,
  hardware, runtime compatibility, policy, privacy mode, and catalog evidence
  rather than only by popularity or search.
- **Priority**: P0.
- **Owner surface**: settings UI, CLI, runtime adapter.

### 3. Model Lifecycle Management

- **LM Studio baseline**: list local models, list loaded models, load/unload,
  per-model defaults, context settings, runtime selection, idle behavior, and
  model identifiers.
- **ModelRouter current state**: configured backend/model ids, known-model
  records, model assignments, route maps, managed llama.cpp/MLX process
  support, runtime status, and explicit proxy restart flows.
- **Parity requirement**: track installed/available/loaded/unavailable states,
  support load/unload where runtimes expose it, show unsupported actions with
  reasons, and keep model ids aligned with the selected runtime.
- **ModelRouter advantage opportunity**: bind lifecycle state to route policy,
  fallback behavior, safety gates, telemetry, and cost/outcome reporting.
- **Priority**: P0.
- **Owner surface**: settings UI, CLI, runtime adapter, proxy.

### 4. Local Inference Server Compatibility

- **LM Studio baseline**: OpenAI-compatible local endpoints including models,
  responses, chat completions, embeddings, and completions, plus native and
  Anthropic-compatible API surfaces.
- **ModelRouter current state**: OpenAI-compatible proxy for supported request
  shapes, route receipts, response headers, health, and telemetry.
- **Parity requirement**: harden common OpenAI-compatible flows such as
  `/v1/models`, `/v1/chat/completions`, `/v1/responses`, streaming, tool calls,
  structured output, embeddings where supported, auth behavior, and consistent
  error shapes.
- **ModelRouter advantage opportunity**: provide one stable `/v1` endpoint over
  multiple local and hosted backends with policy, receipts, fallback, usage
  telemetry, and privacy-safe review.
- **Priority**: P0.
- **Owner surface**: proxy, runtime adapter, settings UI.

### 5. Headless Daemon / Server Operation

- **LM Studio baseline**: headless operation, daemon workflows, server
  start/stop/status, logs, and startup/service guidance.
- **ModelRouter current state**: CLI proxy, settings server, health endpoint,
  background-service docs, proxy process controls, and explicit runtime process
  starts for configured managed runtimes.
- **Parity requirement**: make daemon/server operation predictable through
  status, logs, restart guidance, install/startup instructions, safe shutdown,
  config validation, and no-GUI operation.
- **ModelRouter advantage opportunity**: run as the policy and telemetry layer
  in front of several runtimes, not just one inference server.
- **Priority**: P1.
- **Owner surface**: CLI, proxy, settings UI.

### 6. SDKs And Public API

- **LM Studio baseline**: REST API plus Python and TypeScript SDK workflows for
  local model loading, generation, embeddings, and agent/plugin use cases.
- **ModelRouter current state**: Python routing API, CLI, proxy endpoint,
  receipts, telemetry logs, pricing catalog reports, and admin/state concepts.
- **Parity requirement**: document stable public APIs for route decisions,
  receipts, proxy/admin state, model registry, runtime status, telemetry
  summaries, and pricing/catalog coverage.
- **ModelRouter advantage opportunity**: expose control-plane primitives that
  SDKs and host agents can use without coupling to a single inference runtime.
- **Priority**: P1.
- **Owner surface**: SDK, CLI, proxy, core router.

### 7. Tool Use And Structured Output

- **LM Studio baseline**: OpenAI-compatible tool use and structured output in
  compatible endpoints, plus native API differences by endpoint.
- **ModelRouter current state**: route classes and receipts can account for
  tool needs and safety gates; proxy compatibility depends on supported
  upstream shapes.
- **Parity requirement**: preserve tool-call and structured-output request
  shapes through the proxy where the selected backend supports them, reject or
  explain unsupported combinations clearly, and keep safety gates visible.
- **ModelRouter advantage opportunity**: route tool-heavy or structured-output
  workloads to capable backends and make unsupported routes auditable.
- **Priority**: P1.
- **Owner surface**: proxy, core router, receipts, settings UI.

### 8. MCP Connection And Safety-Gated Tool Surface

- **LM Studio baseline**: MCP host/client workflows in the app and MCP-related
  API support where available.
- **ModelRouter current state**: safety gates and routing policy exist, but
  ModelRouter is not an MCP host or autonomous tool runner.
- **Parity requirement**: design an explicit MCP connection surface that can
  inventory, gate, and route tool-capable model requests without silently
  running tools or becoming an agent harness.
- **ModelRouter advantage opportunity**: centralize provider/model/MCP gating,
  policy denials, receipts, audit events, and manual confirmation for risky
  tool surfaces across agents and runtimes.
- **Priority**: P2.
- **Owner surface**: future plugin, settings UI, proxy, core router.

### 9. Offline Document Chat / RAG

- **LM Studio baseline**: offline document chat/RAG inside the local app.
- **ModelRouter current state**: can route to embeddings or RAG-capable
  backends when configured, but does not own a document index or chat workflow.
- **Parity requirement**: provide a design for local document ingestion,
  embeddings, index status, privacy controls, and routeable RAG backends while
  keeping raw prompt/document exposure explicit.
- **ModelRouter advantage opportunity**: make RAG a transparent, policy-gated
  backend class with receipts, telemetry, catalog coverage, and clear ownership
  of local indexes.
- **Priority**: P2.
- **Owner surface**: future plugin, settings UI, proxy, runtime adapter.

### 10. Runtime Management For llama.cpp, MLX, Ollama, LM Studio, LocalAI

- **LM Studio baseline**: manages its own runtimes, supports llama.cpp/GGUF and
  MLX on Apple Silicon, exposes runtime management through app and CLI.
- **ModelRouter current state**: integrates with LM Studio, Ollama, LocalAI,
  llama.cpp, MLX/MLX-LM, vLLM, hosted providers, and generic OpenAI-compatible
  services as backends; managed-process support exists for configured
  llama.cpp and MLX-LM processes.
- **Parity requirement**: formalize runtime adapters that expose detect,
  health, list models, list loaded models, start, stop, load, unload, endpoint,
  capabilities, and unsupported-operation reasons where each runtime allows it.
- **ModelRouter advantage opportunity**: coordinate several proven runtimes from
  one control center without forcing users to abandon the runtime app that
  already works for them.
- **Priority**: P0.
- **Owner surface**: runtime adapter, settings UI, CLI, proxy.

### 11. Enterprise / Team Controls

- **LM Studio baseline**: individual/local workflows with developer and
  integration surfaces; team governance needs vary by deployment.
- **ModelRouter current state**: local-first receipts, telemetry, feedback
  labels, pricing catalog reporting, policy controls, safety gates, and
  business-model guidance for future governance.
- **Parity requirement**: keep single-user local use strong, then add
  enterprise controls only where they help teams operate the same control plane:
  SSO/OIDC/SAML, RBAC, central policy, provider/model/MCP gating, audit exports,
  shared team telemetry, deployment packaging, and support.
- **ModelRouter advantage opportunity**: make governance the paid/team layer
  while keeping local routing, proxy, receipts, telemetry, catalog, and basic
  control center inspectable and useful.
- **Priority**: Later.
- **Owner surface**: settings UI, CLI, proxy, SDK, future plugin.

## Sequencing

1. **P0 parity loop**: model discovery/download, model lifecycle, runtime
   adapters, and OpenAI-compatible server coverage.
2. **P1 operator polish**: lightweight workbench, headless/server flows, SDK/API
   clarity, structured-output and tool-call passthrough.
3. **P2 expansion**: MCP gating and offline document/RAG surfaces as explicit
   plugins or modules, not hidden orchestration.
4. **Later enterprise**: central policy, identity, audit, shared telemetry, and
   deployment packaging.

At every step, keep the distinction visible: parity features make ModelRouter a
credible local-model app replacement for common workflows; differentiators make
it more than that by adding routing policy, receipts, local-first telemetry,
safety gates, cost/outcome reporting, catalog coverage, and multi-runtime
coordination.
