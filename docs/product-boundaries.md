# Product Boundaries

ModelRouter is a local AI control center and routing/control plane. It can be
the one local surface for common model operations, but it must stay transparent
about what it owns, what it delegates to proven runtimes, and what remains the
host agent's job.

The durable advantage is not owning every layer of AI execution. It is control:
clear model/provider policy, explicit safety gates, route receipts, local-first
telemetry, cost/outcome reporting, and an OpenAI-compatible endpoint that can
sit above many runtimes without locking users into one.

The runtime mode taxonomy lives in `docs/runtime-strategy.md`: externally
managed runtimes, external CLI/API-controlled runtimes, and future bundled
runtime packaging.

## Roles

### ModelRouter As Local AI Control Center

For common local-model workflows, ModelRouter can replace a separate
local-model app as the daily operator surface. The control center should help
users discover installed models, review route-aware recommendations, plan and
run confirmed downloads, start and stop configured local runtimes, inspect
runtime health, expose a local `/v1` endpoint, route requests, and review
privacy-safe telemetry.

This does not mean ModelRouter owns every runtime. It means ModelRouter can
coordinate the local workflow when that is simpler for the operator.

### ModelRouter As Routing/Control Plane

ModelRouter's core routing role is policy and decision control. It selects a
route/backend/model, applies provider and backend constraints, records
receipts, exposes safety gates, and emits telemetry. `route_fast(...)` remains
the small production decision path; richer receipts and reporting belong to
diagnostic, proxy, and admin surfaces.

Runtime discovery, health checks, model lists, and lifecycle capability reports
are advisory operator state. They must not become hidden inputs to
`route_fast(...)`, `route(...)`, or default proxy forwarding. A configured
backend can be routed to even when its runtime adapter is unavailable.

Fusion-like or other multi-agent harnesses can use ModelRouter as their
routing/control layer. They remain responsible for task execution, context
management, delegation, monitoring, synthesis, and final review.

### External Runtimes As Supported Backends

External runtimes such as LM Studio, Ollama, LocalAI, llama.cpp servers,
MLX/MLX-LM, vLLM, generic OpenAI-compatible servers, and hosted providers are
supported backends. They own model execution when they are the selected runtime.
ModelRouter should integrate with them through explicit adapters, config, proxy
forwarding, runtime health checks, and managed-process boundaries where
appropriate.

When a runtime exposes load/unload/list/status APIs, ModelRouter can surface
those controls. When it does not, ModelRouter should show the capability gap
instead of pretending the action exists.

Lifecycle actions for external runtimes are explicit maintenance actions. They
require operator confirmation when mutating, and they must never run as a side
effect of rendering settings, deciding a route, or forwarding a request.

### Host Agents As Executors And Orchestrators

Host agents own the task. They decide how to plan, call tools, manage context,
spawn or supervise workers, verify results, and present final answers.
ModelRouter can help them choose models and enforce policy at routing
boundaries, but it should not hide a planner/worker system inside the proxy.

## What ModelRouter Should Own

- Model discovery from local scan paths, runtime adapters, known model stores,
  and configured catalogs.
- A local known-model registry that can represent installed, configured,
  hosted, and runtime-discovered models without making routing decisions depend
  on live discovery.
- Model recommendation using hardware, route fit, runtime compatibility,
  operator config, and explicit benchmark evidence when available.
- Local runtime lifecycle for configured managed processes, including safe
  start/stop/restart, readiness, logs, and idle behavior.
- Runtime adapter capability reporting for external runtimes, including
  endpoint, detection, health, model listing, loaded-model listing, lifecycle
  support, load/unload support, logs, and unsupported-operation reasons.
- Route, provider, and backend policy, including profiles, allowlists,
  denylists, local-only controls, fallbacks, and human-confirm gates.
- A local OpenAI-compatible proxy endpoint for clients and agents that want one
  stable `/v1` surface.
- Route receipts that explain selected routes, rejected options, policy
  constraints, fallback behavior, safety state, and wrong-route next actions.
- Telemetry, cost, and outcome reporting from local logs, upstream usage fields,
  manual feedback labels, and local versioned pricing catalogs.
- Safety gates for destructive, sending, purchasing, deployment, high-impact,
  or ambiguous requests.

## What ModelRouter Should Not Own

- Hidden planner/worker orchestration, synthesis, or final agent review.
- A custom inference engine built from scratch when proven runtimes can be used
  through adapters or proxy boundaries.
- Lock-in to ModelRouter-only runtimes or model storage.
- Runtime adapter discovery as an implicit routing signal.
- Live pricing fetches during `route_fast(...)`, `route(...)`, proxy
  forwarding, or default routing paths.
- Benchmark, cost-reduction, or performance-superiority claims without
  checked-in evidence and clear scope.
- Silent model downloads, hosted-provider enablement, config writes, benchmark
  runs, runtime starts, verifier calls, or prompt logging.
- A chat UI, prompt transcript product, or hidden prompt workspace.

## LM Studio Compatibility, Not Hostility

ModelRouter can replace a local-model app for operators who want one integrated
workflow for discovery, recommendations, downloads, runtime controls, proxy
routing, receipts, telemetry, and feedback. That is a product convenience, not
a hostility stance toward LM Studio or similar tools.

LM Studio, Ollama, LocalAI, llama.cpp, MLX/MLX-LM, vLLM, and hosted providers
remain valuable runtimes and UIs. ModelRouter should work above or alongside
them. If a user prefers LM Studio for model search, chat, or runtime management,
ModelRouter should still provide policy, routing, safety gates, receipts, and
telemetry around the local endpoint LM Studio exposes.

The design goal is compatibility and control without lock-in: operators can
choose the integrated ModelRouter control center, keep their existing runtime
apps, or mix both.

The parity roadmap is intentionally more ambitious than compatibility alone:
LM Studio is the floor, not the ceiling. ModelRouter should reach common
local-workflow parity where that helps adoption, then differentiate through
provider-neutral routing policy, receipts, local-first telemetry, safety gates,
cost/outcome reporting, catalog coverage, and multi-runtime coordination.

## Evidence And Reporting Boundaries

Telemetry and reports should separate route policy, actual usage, estimated
cost, and outcome labels. Routing uses configured metadata such as `cost_tier`
and provider policy. Usage comes from upstream responses when already present.
Cost estimates come from local versioned pricing catalogs. Outcome labels come
from explicit operator/user feedback.

ModelRouter must not infer success from latency, token usage, verifier status,
or cost. It must not fetch live prices while routing. It should report unknown
coverage, missing catalog matches, and placeholder pricing as gaps rather than
inventing precision.
