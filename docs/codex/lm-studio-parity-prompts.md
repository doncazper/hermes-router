# ModelRouter LM Studio Parity And Beyond Prompt Pack

Goal: make ModelRouter capable of replacing LM Studio for common local-model workflows while preserving its stronger identity as a provider-neutral routing, policy, telemetry, and runtime control center.

Core principle:

> LM Studio is the floor, not the ceiling.

Build parity by user workflow, not by cloning LM Studio internals. ModelRouter should work as a standalone local AI control center, and it should also work above or alongside LM Studio, Ollama, llama.cpp, MLX, LocalAI, hosted APIs, and future runtimes.

Guardrails:

- Do not turn ModelRouter into a hidden autonomous planner/worker orchestrator.
- Do not build a custom inference engine unless explicitly justified later.
- Prefer adapters over lock-in.
- Keep routing/proxy hot paths fast and deterministic.
- Do not fetch live pricing during routing or proxy forwarding.
- Do not claim benchmark parity, cost reductions, or frontier performance without evidence.
- Preserve open provider/runtime compatibility.

## Starter Prompt: Save And Execute The First Roadmap Step

```text
Read docs/codex/lm-studio-parity-prompts.md.

Execute Prompt 1 only: Product Positioning And Parity Roadmap.

Do not implement feature code yet. Focus on product positioning and the LM Studio parity roadmap. Preserve the existing ModelRouter identity as a routing/control plane, but make clear that ModelRouter should also become capable of replacing LM Studio for common local-model workflows. LM Studio is the floor, not the ceiling.

After the docs update, run relevant checks, summarize changes, and stop.
```

## Prompt 1: Product Positioning And Parity Roadmap

```text
Create a durable product roadmap for LM Studio parity and beyond.

Scope:
- Review existing positioning docs, north-star docs, UI docs, README, and roadmap docs.
- Create docs/lm-studio-parity-roadmap.md.
- Update existing positioning docs where needed.

The roadmap must include these capability areas:
1. Local chat app / test workbench
2. Model discovery and download
3. Model lifecycle management
4. Local inference server compatibility
5. Headless daemon/server operation
6. SDKs and public API
7. Tool use and structured output
8. MCP connection and safety-gated tool surface
9. Offline document chat / RAG
10. Runtime management for llama.cpp, MLX, Ollama, LM Studio, LocalAI
11. Enterprise/team controls where appropriate

For each area, document:
- LM Studio baseline
- ModelRouter current state
- Parity requirement
- ModelRouter advantage opportunity
- Priority
- Owner surface: core router, proxy, settings UI, CLI, runtime adapter, SDK, or future plugin

Explicitly state:
- ModelRouter should be able to replace LM Studio for common local workflows.
- ModelRouter should also work above and alongside LM Studio.
- ModelRouter should not lock users into one runtime/provider.
- ModelRouter should not become a hidden autonomous orchestration harness.
- LM Studio-level capability is the baseline, not the ambition ceiling.

Acceptance:
- Docs clearly say "LM Studio is the floor, not the ceiling."
- Roadmap distinguishes parity features from ModelRouter differentiators.
- Existing routing/control-plane positioning remains intact.
- No code changes unless needed for doc links.
```

## Prompt 2: Business Model And Open-Core Positioning

```text
Design the commercial/open-source packaging direction for ModelRouter.

Scope:
- Review docs/business-model.md, docs/product-north-star.md, docs/product-boundaries.md, docs/roadmap.md, README.md, and any docs touching local-first/open-source positioning.
- Add or update a document such as docs/commercial-packaging.md or docs/business-model.md.
- Recommend what stays free/open-source and what could become paid later.

Suggested free/open core:
- local routing
- local proxy
- model registry
- local runtime adapters
- basic telemetry
- local pricing catalog
- single-user settings/control center
- compact/full local control-center modes

Suggested paid team/enterprise layer:
- SSO
- centralized policy/model gating
- shared team presets
- audit/export controls
- private model catalogs
- admin dashboards
- license management
- collaboration and approval workflows
- managed enterprise packaging/support

Acceptance:
- Does not overpromise revenue.
- Does not weaken open-source trust.
- Explains why users would choose ModelRouter over LM Studio:
  provider neutrality, routing policy, telemetry, receipts, cost/outcome visibility, safety gates, runtime control, and multi-runtime coordination.
- Keeps local-first users respected.
- Makes clear that free/local use is not a crippled demo.
```

## Prompt 3: Model Library And Registry

```text
Build the foundation for LM Studio-style model ownership.

Scope:
- Inspect current model/backend/provider config, existing model recommendation/download code, runtime templates, settings UI, and tests.
- Design or implement a model registry that can represent local and hosted models.

The registry should track:
- provider
- runtime
- model id/name
- source
- local path when applicable
- format: GGUF, MLX, safetensors, API, etc.
- context length if known
- quantization if known
- size if known
- license if known
- install/download state
- health/load state
- tags/capabilities
- routing eligibility

Support import/discovery from:
- existing ModelRouter config
- LM Studio model folders if detectable
- Ollama model list if available
- local folders
- user-declared hosted models

Acceptance:
- ModelRouter has a durable concept of "known models."
- Registry is JSON-safe and local-first.
- Missing metadata is tolerated.
- No routing hot path slowdown.
- Tests cover local model, hosted model, missing metadata, and imported model cases.
- Docs explain how the registry helps ModelRouter reach LM Studio-level model library capability without becoming a locked-in runtime.
```

## Prompt 4: Runtime Manager

```text
Add runtime management as adapters, not a custom inference engine.

Scope:
- Inspect existing backend adapter code, setup assistant, runtime templates, CLI commands, settings UI, proxy health checks, and tests.
- Design or implement runtime adapters for:
  - LM Studio
  - Ollama
  - llama.cpp server
  - MLX / MLX-LM
  - LocalAI
  - hosted OpenAI-compatible backends

Each adapter should expose:
- detect installed/available
- health check
- list models if supported
- list loaded models if supported
- start server if supported
- stop server if safe/supported
- load model if supported
- unload model if supported
- endpoint URL
- capabilities
- unsupported operation reason

Acceptance:
- Runtime manager is optional and adapter-based.
- Unsupported operations are reported clearly, not treated as generic errors.
- ModelRouter can work with externally managed runtimes.
- No runtime adapter is required for route_fast.
- Tests cover at least one mocked runtime with full support and one with partial support.
- Docs make clear ModelRouter is coordinating proven runtimes, not trying to become a low-level inference engine by default.
```

## Prompt 5: Local Server Compatibility

```text
Harden ModelRouter as the single endpoint users can point apps at.

Scope:
- Audit OpenAI-compatible endpoint coverage in the proxy/server.
- Review proxy tests, routing log tests, telemetry tests, and docs.
- Identify gaps against common LM Studio workflows:
  - /v1/models
  - /v1/chat/completions
  - streaming
  - tool calls
  - structured output
  - embeddings if supported
  - auth/token behavior
  - error shape consistency
- Consider Anthropic-compatible endpoint support as a future design if not already present.

Acceptance:
- Add or update a compatibility matrix doc.
- Add focused tests for the most important OpenAI-compatible flows.
- Do not fake unsupported capabilities.
- Do not route based on live pricing or telemetry side effects.
- Preserve privacy-safe telemetry behavior.
- route_fast remains unchanged unless a direct bug is discovered and justified.
```

## Prompt 6: Lightweight Chat/Test Workbench

```text
Build a practical local-model test surface, not a giant consumer chat product.

Scope:
- Inspect settings/control-center UI, compact mode, admin views, proxy state, receipts, and telemetry surfaces.
- Add or design a compact Workbench/Test view.

It should support:
- select route/model/backend
- send a test prompt
- stream response if available
- show selected route
- show receipt summary
- show latency/token usage when available
- test structured output if supported
- test tool-call eligibility if supported
- save no prompts by default unless the user explicitly enables it

Acceptance:
- Workbench helps users verify models and routes.
- It does not dominate the settings/control-center.
- It does not leak prompts into telemetry.
- It works with local and hosted backends where configured.
- It clearly distinguishes test/chat UX from autonomous agent orchestration.
- If any UI is left unwired, document exactly what is static and why.
```

## Prompt 7: MCP And Tool Surface

```text
Design MCP support in a way that strengthens ModelRouter's control-plane identity.

Scope:
- Add or update a design doc such as docs/mcp-tool-surface.md.
- Review current safety, routing, receipts, session-aware routing, and host-agent extension docs.
- Define how ModelRouter should connect to MCP servers or expose MCP-aware routing metadata.

ModelRouter may:
- register MCP servers
- show available tools
- safety-gate tool availability
- route requests based on tool-use capability
- log privacy-safe tool metadata
- expose policy decisions to host agents

ModelRouter must not:
- become a hidden autonomous agent harness
- spawn workers without host/user control
- silently execute tools outside explicit request flow
- infer outcome labels from tool results

Acceptance:
- Design distinguishes MCP connection, routing, safety gating, and execution ownership.
- Host agents remain responsible for task execution and context management.
- Includes example flows for local chat, external agent, and enterprise policy use.
- Preserves ModelRouter's existing non-goals.
```

## Prompt 8: Offline Documents / RAG

```text
Design local document chat without compromising ModelRouter's core.

Scope:
- Add or update docs/offline-documents-rag.md.
- Review telemetry/privacy docs, model registry direction, runtime adapter direction, and settings UI.
- Define a local-first RAG capability.

Include:
- document ingestion
- local embeddings
- vector index location
- privacy model
- citation behavior
- model/backend routing
- telemetry boundaries
- prompt/content retention controls
- opt-in indexing
- deletion/rebuild flow

Acceptance:
- No document text is written to route telemetry.
- RAG is clearly optional.
- Works with local models where possible.
- Hosted-provider use requires explicit user awareness.
- Design leaves room for future implementation without blocking current routing work.
- Docs explain how this reaches LM Studio-style offline document chat while preserving ModelRouter's transparency.
```

## Prompt 9: SDK And Public API

```text
Design a first-class ModelRouter developer surface.

Scope:
- Add or update docs/sdk-api-roadmap.md.
- Review existing CLI, proxy, route APIs, receipts, telemetry, pricing catalog, model registry plans, and runtime manager plans.
- Define future Python and TypeScript SDKs.

SDKs should expose:
- route request
- route receipt
- model registry
- runtime status
- telemetry summary
- pricing catalog status
- feedback/outcome labels
- policy/config inspection
- optional workbench/test helpers

Acceptance:
- SDKs are not required for core CLI use.
- Public API avoids leaking implementation internals.
- API supports local-first and enterprise/team future use.
- Includes examples for Python and TypeScript.
- Does not turn SDK into an agent framework by default.
- Explains how SDKs help ModelRouter compete with LM Studio while keeping ModelRouter's provider-neutral control-plane identity.
```

## Prompt 10: Headless / Daemon Mode

```text
Design or complete headless operation.

Scope:
- Inspect existing CLI/proxy/settings server behavior, service scripts, background services docs, and tests.
- Add or update docs/headless-daemon.md.

Headless mode should support:
- start ModelRouter service
- stop service
- status
- logs
- health
- active endpoint
- runtime adapter status
- model registry refresh
- pricing catalog status
- telemetry summary

Acceptance:
- Works without opening the UI.
- Does not require dev-tree imports.
- Suitable for launch agents/system services later.
- Keeps secrets out of logs.
- Includes smoke-test commands.
- Makes clear which pieces are implemented now and which are future-facing.
```

## Prompt 11: UI Integration And Unwired Surface Report

```text
Hardwire the UI to real app state and report what remains unwired.

Scope:
- Inspect settings/control-center UI, compact mode, admin views, pricing catalog panels, telemetry panels, runtime/model panels, workbench surfaces, and tests.
- Replace mock/static UI state with real app state wherever practical.
- Preserve compact/full mode distinction:
  - full control center/main window
  - compact standalone minimal control panel/windowed mode

After implementation, write a report:
- wired surfaces
- partially wired surfaces
- unwired surfaces
- why each remains unwired
- required backend/API work
- suggested next implementation order

Acceptance:
- UI does not pretend unavailable features work.
- Empty states are honest and useful.
- No oversized hero cards for routine operational information.
- Report is committed under docs or generated as a clear markdown artifact.
- Tests or smoke checks cover the key UI state paths where practical.
```

## Prompt 12: Release/Readiness Pass

```text
Prepare the completed batch for review and release.

Scope:
- Run relevant tests.
- Run installed-package smoke if release-worthy.
- Verify CLI commands.
- Verify settings/control-center starts.
- Verify / and /compact if applicable.
- Check docs for overclaims.
- Confirm worktree status.

Acceptance:
- Worktree is clean after commit.
- Commit message clearly describes the batch.
- Release notes include user-facing changes only.
- Explicitly state if no PyPI release is intended.
- Any remaining gaps are documented, not hidden.
```

## Recommended Execution Rhythm

Run the prompts one at a time. For each prompt:

1. Inspect the relevant files first.
2. Keep the implementation commit-sized.
3. Preserve route_fast and routing hot-path guarantees.
4. Add focused tests for any behavior change.
5. Update docs when product semantics shift.
6. Commit cleanly before moving to the next prompt.

For large implementation prompts, ask Codex to stop after proposing the smallest useful slice if the scope is too wide for one batch.
