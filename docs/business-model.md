# Business Model

ModelRouter should use an open-core strategy that keeps the local developer
experience inspectable and useful while monetizing team governance and
operations.

This note is product strategy, not a license change, pricing plan, entitlement
system, or implementation commitment. It describes a realistic packaging
direction without promising revenue or changing the current local-first product
boundary.

## Free And Open Core

The core should stay free, local-first, and inspectable:

- Routing engine, including `route_fast(...)`, `route(...)`, profiles, policy
  constraints, and fail-closed behavior.
- Local OpenAI-compatible proxy.
- Model registry and local model-library metadata.
- Local runtime integrations and adapters for common runtimes.
- Route receipts, explanations, and reason codes.
- Local telemetry, feedback labels, replay, and workflow benchmarks.
- Local versioned pricing catalog and reporting estimates.
- Single-user settings/control center for individual operators.
- Compact and full local control-center modes.

This is the trust-building layer. Users and embedding applications need to see
how routing decisions are made, verify that prompts and secrets stay local by
default, inspect policy behavior, and adapt the product to their own runtimes.
If individual/local use is blocked too early, adoption and trust both suffer.
Free local use should be a real product, not a crippled demo: a single operator
should be able to route requests, run the proxy, inspect receipts, manage local
runtime integrations, review local telemetry, use the pricing catalog, and
operate the basic control center without needing an enterprise license.

## Paid Or Enterprise Surface

Paid features should focus on organization operations, not on blocking
individual local workflows:

- SSO through SAML/OIDC.
- RBAC for settings, policy changes, runtime actions, and audit access.
- Centralized policy management for routes, providers, backends, profiles,
  safety gates, and model/provider/MCP gating.
- Shared team presets for routing profiles, backend pools, model assignments,
  safety defaults, and catalog choices.
- Private model catalogs for approved local, hosted, and internal models.
- Admin dashboards for policy state, fleet health, catalog coverage, outcome
  labels, and usage aggregates.
- Audit and export controls for route decisions, policy denials, settings
  changes, feedback labels, catalog coverage, runtime actions, and governance
  events.
- Shared team telemetry with privacy controls and prompt-redaction defaults.
- License management for team/org packaging when a commercial edition exists.
- Collaboration and approval workflows for policy changes, provider enablement,
  MCP/tool exposure, runtime changes, and audit review.
- Managed enterprise packaging for desktops, servers, internal platforms, and
  controlled environments.
- Support, implementation help, and SLA-backed releases.

These are the places where teams have budget and operational pain: proving what
models were allowed, controlling hosted-provider exposure, distributing policy,
reviewing audit trails, and supporting many operators without hand-edited local
configs.

## Why Users Choose ModelRouter With Or Instead Of LM Studio

ModelRouter should respect LM Studio as a useful local-model app and runtime
surface. The commercial story should not depend on competitive dunking or on
forcing users away from tools that already work.

Users choose ModelRouter when they need control across more than one local app
or provider:

- Provider neutrality across LM Studio, Ollama, LocalAI, llama.cpp,
  MLX/MLX-LM, vLLM, generic OpenAI-compatible runtimes, hosted providers, and
  future adapters.
- Routing policy that can select or constrain backend/model choices by task
  shape, privacy mode, safety rules, route profile, and operator policy.
- Receipts that explain selected routes, rejected options, fallback behavior,
  safety state, and feedback actions.
- Privacy-safe telemetry, manual outcome labels, usage capture, catalog
  coverage, and cost/outcome visibility without raw prompt exposure by default.
- Safety gates for risky actions and provider/tool exposure.
- Runtime control and model-library workflows that can replace common
  local-model app loops when one integrated control center is simpler.
- Multi-runtime coordination when a user wants LM Studio for some models,
  Ollama or llama.cpp for others, and hosted providers only under explicit
  policy.

That makes ModelRouter complementary to LM Studio for users who like LM
Studio's model search, chat, or runtime management, and a possible replacement
for common local workflows when users want one provider-neutral control center.
The differentiator is governance-quality control, evidence, and coordination,
not a claim that ModelRouter is a better inference engine or chat app.

## Why Not Monetize By Blocking Local Use

ModelRouter's durable advantage is control, policy, receipts, telemetry, and
safety across many runtimes. That advantage depends on trust. The routing core,
local proxy, local runtime integrations, receipts, and basic control center
should remain inspectable so developers can understand and verify the product
before adopting it broadly.

Monetization should not require locking users into ModelRouter-only runtimes,
hiding routing behavior, limiting individual local workflows, or putting live
pricing/network fetches into routing. The free core should be strong enough that
individuals can use ModelRouter with or instead of local model apps. The paid
surface should make the same control plane manageable for teams and
organizations.

In practical terms, do not put these behind a paywall for individual/local use:
local routing, local proxying, local runtime adapters, model registry,
receipts, local telemetry, local pricing catalog, and basic settings/control
center. Monetization should start where coordination, governance, identity,
auditability, packaging, and support become organization problems.

## Alignment With Product Boundaries

The business model follows `docs/product-boundaries.md`:

- ModelRouter owns routing/control-plane policy, receipts, local telemetry,
  cost/outcome reporting, safety gates, and local control-center workflows.
- External runtimes own model execution when selected.
- Host agents own task execution, planning, delegation, and final review.
- Enterprise monetization adds governance around those boundaries; it should
  not turn ModelRouter into hidden agent orchestration or a custom inference
  engine.

## Practical Path To Profit

1. Grow adoption with a useful open local control center that works with LM
   Studio, Ollama, LocalAI, llama.cpp, MLX/MLX-LM, vLLM, generic
   OpenAI-compatible runtimes, hosted providers, and agent tools.
2. Make receipts, local telemetry, pricing catalog coverage, and outcome labels
   reliable enough that teams want the same evidence across many users.
3. Add enterprise governance where coordination becomes hard: central policy,
   SSO/RBAC, audit exports, shared telemetry, deployment packaging, and support.
4. Keep the open core credible so enterprise buyers can evaluate the decision
   path rather than taking a black box on faith.

The result is a realistic revenue path that does not undermine adoption:
individuals get the local control center, teams pay for governance and
operational scale.
