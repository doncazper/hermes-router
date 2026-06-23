# Host Adapter Contract

## Purpose

ModelRouter stays the fast decision layer. It should not load every model,
keep a large model pool hot, or execute user actions by itself. Adapter work is
optional and lazy: the router decides, a dispatch plan explains what would be
called, and future adapters can run only after the caller chooses to dispatch.

## Current Milestone

The current dispatch surface is dry-run only:

```bash
model-router dispatch-plan "fix the repo and run tests"
model-router dispatch-plan --json "rewrite this text"
model-router dispatch-plan --include-alternatives --json "rewrite this text"
```

Programmatic use:

```python
from model_router import build_dispatch_plan

plan = build_dispatch_plan("rewrite this text")
full_plan = build_dispatch_plan("rewrite this text", include_alternatives=True)
print(plan.selected_engine, plan.adapter, plan.can_dispatch)
```

The dry-run plan never calls a provider, starts a local server, loads model
weights, runs a shell command, sends a message, or performs an external action.
It skips ranked alternatives by default for speed; request them explicitly when
callers need a full diagnostic receipt.

The optional `model-router-proxy` runtime adapter is the first supported
execution boundary. It exposes one local OpenAI-compatible endpoint and forwards
chat completions and Responses API requests to configured OpenAI-compatible
upstreams. It remains outside the router hot path and is installed only with
the `proxy` extra.

The `hermes/plugins/model_router` package path is legacy Python packaging
history, not a host-app adapter contract. New integrations should import
`model_router`, call `ModelRouter.route_fast(...)` or `route(...)`, or use the
OpenAI-compatible proxy endpoint. Host-specific plugin manifests should live in
the embedding application, not in this legacy namespace.

Routing profiles and provider policies are input constraints on the router
decision. They do not change adapter execution semantics, start runtimes, call
providers, or bypass confirmation gates. Proxy backend policy is enforced at
the OpenAI-compatible proxy boundary because backend names are proxy-local
configuration.

The optional verifier is also proxy/runtime behavior. It runs after forwarding
when explicitly configured, never inside `route_fast(...)`, and never bypasses
`human_confirm`.

Offline workflow benchmarks are router diagnostics, not adapter execution. They
route sanitized fixture prompts, compare expected engines, and emit prompt
hashes and receipt explanations without calling providers, starting runtimes,
or invoking verifiers.

Catalog updates are also configuration maintenance, not execution. The
packaged-only workflow can preview and apply router catalog defaults after
confirmation, with backup and migration logging, but it does not download
models, enable hosted providers, start runtimes, or dispatch adapters.

## Runtime Principles

- Load the YAML catalog once through `ModelRouter`.
- Route prompts in memory.
- Load or start a model runtime only if a future caller explicitly dispatches.
- Enforce provider policy before dispatch and preserve backend policy in proxy
  forwarding/fallback code.
- Keep at most one heavy local model active by default.
- Allow a tiny fast local model to stay warm only when the user opts in.
- Prefer hosted/API or agent adapters when local memory is constrained.
- Block `human_confirm` and high-risk decisions until explicit confirmation
  exists outside the router, unless a versioned safety override deliberately
  narrows that requirement.

## Adapter Shape

Future adapters should be small wrappers around a single runtime family:

```text
adapter name -> runtime owner
local_chat -> Ollama, LM Studio, llama-server, or another local chat runtime
local_reasoning -> same as local_chat, but selected for heavier prompts
local_code -> local code model or code-agent bridge
web_research -> web/RAG research service
local_vision -> vision/OCR runtime
local_image_generation -> diffusion/image runtime
claude_code / codex -> command or agent tool bridge
openai_chat / anthropic_chat -> hosted API bridge
confirmation_gate -> human confirmation UI
```

No runtime is mandatory. Ollama, LM Studio, llama-server, hosted APIs, Codex,
Claude Code, and image servers are all optional choices declared in YAML.

## Future Execution Boundary

If execution is added later, keep it outside the scoring policy:

```python
decision = router.route(prompt)
plan = build_dispatch_plan(prompt, router=router)

if plan.requires_confirmation:
    return ask_user_to_confirm(plan)

adapter = adapters[plan.adapter]
return adapter.run(prompt, decision)
```

Adapters should own lazy loading, idle unload, memory budgets, and runtime health
checks. The router should continue to emit receipts and stay usable without any
provider installed.
