# Production Readiness

ModelRouter is the open switchboard for AI model routing: one local
OpenAI-compatible endpoint that routes each request to the right model, with
receipts, safety gates, and full provider control.

Its production surface is intentionally small: initialize `ModelRouter` once,
keep it in memory, and call `route_fast(prompt)` for live traffic. The local
proxy, optional verifier, workflow benchmarks, model evals, catalog
maintenance, pricing reports, model discovery/import, and runtime adapters sit
outside that hot path.

## API Contract

- `ModelRouter.route_fast(prompt, hints=None) -> str` is the production routing
  API. It returns only the selected engine name.
- `ModelRouter.route(prompt, hints=None, include_alternatives=True)` is the
  diagnostic and audit API. It returns a `RoutingDecision` with scores, reasons,
  feature flags, rejected engines, fallback data, and optional alternatives.
- `route_prompt(...)` is a compatibility helper for one-off scripts. It creates
  a router for the call and is not a high-QPS runtime boundary.
- The CLI is for humans, diagnostics, CI, and scripts. Production services
  should not spawn the CLI per prompt.

`route_fast(...)` is intentionally not receipt-compatible. Callers that need
receipts, explanations, audit logs, or ranked alternatives should use
`route(...)`, usually with `include_alternatives=False` unless alternatives are
needed.

## Routing Profiles

Routing profiles are a thin constraint layer over the existing hint and
requirements model:

- `fast`: low-latency, low-cost preference.
- `balanced`: default deterministic routing.
- `quality`: stronger configured fallbacks are allowed.
- `private`: local-only provider policy; hosted API providers are rejected and
  listed in receipts.
- `safe`: stricter confirmation for ambiguous or sensitive requests.

Profiles do not add planner-worker behavior, classifier calls, downloads,
benchmarks, verifier calls, or automatic hosted-provider enablement. They also
do not bypass `human_confirm`; forced engines remain subject to confirmation
and profile constraints.

`route_fast(prompt)` without hints stays on the precompiled default hot path.
When a caller supplies profile hints, those constraints are applied inside the
existing fast fallback resolver. The local proxy passes its
`proxy.routing_profile` as a hint for each request and logs the diagnostic
receipt when observability is enabled.

## Provider And Backend Policies

Provider policy is versioned in `model_router.yaml` under `provider_policy`.
It reuses the existing provider names, cost tiers, latency tiers, route targets,
and fallback chains:

```yaml
provider_policy:
  version: 1
  provider_allowlist: []
  provider_denylist: []
  local_only: false
  hosted_allowed: true
  max_cost_tier: null
  max_latency_tier: null
  route_pools:
    simple:
      local_only: true
```

The router compiles provider policy into `RoutingRequirements` alongside
profile hints. `human_confirm` remains allowed even under restrictive
allowlists/denylists, and fallback resolution rejects denied providers instead
of jumping around the policy. `route_fast(...)` honors configured provider
policy through its precompiled target engines and per-request hint path without
adding classifier calls or runtime behavior.
Caller hints can narrow provider policy, but they cannot loosen configured
allowlists, denylists, local-only mode, or max tier caps.

Proxy backend policy is versioned in `routing_proxy.yaml` under
`backend_policy`:

```yaml
backend_policy:
  version: 1
  backend_allowlist: []
  backend_denylist: []
```

Backend policy is enforced only in proxy space, because backend names are proxy
configuration details. The proxy rejects denied selected backends before any
upstream request, filters denied fallback backends, reports
`backend_policy_rejected`, and exposes the redacted policy state in health and
settings diagnostics.

## Productized Receipts

`route(...)` and `model-router decide --json` preserve the original receipt
fields and add deterministic product fields for normal operator review:

- `summary`
- `reason_codes`
- `delegation_suitability`
- `selected_route_explanation`
- `policy_explanation`
- `rejection_explanation`
- `fallback_explanation`
- `safety_explanation`
- `privacy_explanation`
- `wrong_route_next_action`

Reason codes are additive and stable enough for tests, dashboards, and
release-note comparisons. Existing reason strings remain present for
compatibility. Use `model-router decide --explain` for a concise human-readable
view, and use JSON receipts for scripts.

`delegation_suitability` is diagnostic only. It exposes deterministic signals
such as mechanical work, judgment-heavy work, verification-heavy work,
repo-wide scope, risky/external actions, and ambiguity sensitivity so host
agents can make their own delegation choices. ModelRouter does not spawn
workers or delegate tasks.

Receipts are diagnostic artifacts, not the production hot path.
`route_fast(...)` still returns only an engine string and does not build
receipts, call classifiers, summarize telemetry, load pricing catalogs, scan or
import models, start runtimes, run benchmarks/evals, log prompts, scan the
filesystem, make network requests, or perform provider calls. `route(...)` may
build diagnostic scores and route details, but it must not run live pricing
refresh, runtime lifecycle actions, workflow benchmarks, model evals, or
provider/runtime network calls.

## Explicit Verification Boundary

Verification is optional proxy/runtime behavior and is disabled by default:

```yaml
verifier:
  version: 1
  mode: "off"
  backend: null
  sample_rate: 0.0
  route_codes: []
  timeout_seconds: 10
  failure_behavior: log_only
  include_response_preview: false
  max_response_preview_chars: 500
```

Supported modes are `off`, `receipt-only`, `sampled`, and
`always-for-risky-output`. The verifier runs after the selected backend returns
a non-streaming response. It never runs for `human_confirm`, never changes
router scoring, and never participates in `route_fast(...)`.

Verifier telemetry includes mode, status, backend, status code, latency, and a
short error class when applicable. Streaming requests are marked
`skipped_streaming` instead of being buffered. Failures are log-only unless
`failure_behavior: fail_closed` is explicitly configured.

## SLOs

These SLOs apply to an initialized router in a long-running Python process, not
to Python interpreter startup or CLI subprocess calls.

| Surface | Target |
| --- | ---: |
| `route_fast(...)` ordinary mixed prompts, best sample | <= 25 us/route |
| `route_fast(...)` ordinary mixed prompts, mean sample | <= 50 us/route |
| `route(...)` without alternatives, ordinary mixed prompts | <= 250 us/route |
| `route(...)` with alternatives, ordinary mixed prompts | <= 350 us/route |

Local development on Apple Silicon is normally much faster than these budgets.
The CI guard intentionally uses portable thresholds to catch regressions without
being flaky on shared runners.

## Benchmark Guard

Run the production hot-path guard:

```bash
python scripts/check_route_fast_latency.py --json
```

For stricter local checks:

```bash
python scripts/check_route_fast_latency.py \
  --iterations 100000 \
  --repeat 5 \
  --max-best-us 10 \
  --max-mean-us 20 \
  --json
```

The script exits non-zero when either budget is exceeded. CI runs it after lint
and tests.

## Workflow Benchmarks

Run the offline correctness suite before release notes or routing-threshold
changes:

```bash
model-router workflow-benchmark --json --fail-on-mismatch
```

This benchmark uses sanitized fixture prompts for simple, balanced, coding,
research, vision, image generation, safety, private-profile, quality-profile,
and sidekick-delegation-shaped workflows. It measures route correctness, profile
behavior, confirmation behavior, route changes, diagnostic receipt fields, and
expected delegation suitability signals. It does not call providers, local model
servers, optional verifiers, downloads, benchmarks, or hosted APIs. Reports
serialize prompt hashes rather than prompt bodies.

## Catalog Update Workflow

Catalog maintenance is explicit and packaged-only by default:

```bash
model-router catalog status --config ~/.model-router/model_router.yaml
model-router catalog diff --config ~/.model-router/model_router.yaml
model-router catalog apply --config ~/.model-router/model_router.yaml --yes
```

`status` and `diff` never write and perform no remote checks. `apply` requires
confirmation, backs up an existing local config before writing packaged
defaults, and appends a JSONL entry to `catalog-migrations.jsonl`. The workflow
does not download models, enable hosted providers, change provider policy
silently, or mutate proxy backend policy.

## Observability

The router does not perform built-in hot-path logging. This avoids adding
per-request formatting, allocation, IO, or lock contention to
`route_fast(...)`.

Production services that need metrics should wrap calls at the service boundary.
The optional proxy does this through an `observability` config block that writes
JSONL events with selected engine, route scores, feature flags, backend, fallback
status, caller-owned request id, and latency. Default prompt retention is a hash
plus a redacted preview. Full prompt capture requires `prompt_capture: full` or
`MODEL_ROUTER_LOG_PROMPTS=1` and should be used only for deliberate calibration
runs.

Use `model-router feedback` to label bad routes and
`scripts/replay_routing_log.py` to replay labeled traffic against a new router
implementation before changing routing thresholds.

Use `model-router telemetry summary` during dogfood runs to track event
coverage, replayable events, unlabeled request ids, skipped private events, and
`expected -> actual` mismatch groups. Use `model-router telemetry feedback` to
inspect labels. Both commands avoid printing raw prompt text by default; notes
are hidden unless `--include-notes` is passed. See
`docs/telemetry-dogfood.md`.

The proxy adds route-identification headers to routed chat and Responses API
responses so operators can label a bad route while it is fresh:
`X-ModelRouter-Request-ID`, `X-ModelRouter-Engine`,
`X-ModelRouter-Profile`, `X-ModelRouter-Backend`,
`X-ModelRouter-Fallback`, and `X-ModelRouter-Route-API`. These headers are
metadata-only and must not include raw prompts, request bodies, API keys, or
secrets. On shutdown, the proxy prints a best-effort session summary with route
counts and the telemetry summary command for follow-up review.

Cost and outcome telemetry must stay outside routing decisions. `route_fast(...)`
and `route(...)` use configured route/backend metadata such as `cost_tier` and
provider policy; they must not fetch live prices, scrape provider pages, or call
pricing APIs. The proxy can record actual upstream usage fields when a response
already includes them, and reporting paths can estimate cost only from a local
versioned pricing catalog. Outcome labels are explicit user/operator feedback,
not inferred success claims.

Future cost/outcome telemetry fields should distinguish route identity
(`selected_engine`, `routing_profile`, `selected_backend`, `selected_model`,
`backend`, `backend_model`), latency (`route_latency_ms`,
`diagnostic_latency_ms`, `upstream_latency_ms`, `total_latency_ms`), usage
(`usage_prompt_tokens`, `usage_completion_tokens`, `usage_total_tokens`,
`usage_cached_input_tokens`, `upstream_model`), configured cost metadata
(`configured_cost_tier`, `configured_latency_tier`), estimated cost
(`estimated_input_cost`, `estimated_output_cost`,
`estimated_cached_input_cost`, `estimated_total_cost`,
`estimated_cost_currency`, `pricing_catalog_version`, `pricing_catalog_source`,
`pricing_source`, `pricing_effective_date`, `pricing_match_status`), and feedback
(`outcome_label`, `feedback_label`, `expected_engine`). Missing usage or pricing
catalog matches should produce no exact estimate rather than an invented value.

Pricing maintenance is explicit and local:

```bash
model-router pricing status --override ~/.model-router/pricing_catalog.yaml
model-router pricing diff --override ~/.model-router/pricing_catalog.yaml
model-router pricing apply --override ~/.model-router/pricing_catalog.yaml --yes
```

These commands operate on packaged metadata and a local override file only. They
must not fetch pricing during `route_fast(...)`, `route(...)`, proxy forwarding,
verification, or telemetry rendering.

Optional classifier-based routing is not part of the production path. The
Milestone 7 audit found no labeled replay mismatches that justify it. Revisit
that decision only with recurring labeled replay failures, no deterministic fix
without regressions, and proof that the default `route_fast(...)` latency guard
still passes. See `docs/advanced-routing.md`.

## Safety Configuration

Human confirmation is default-on for destructive, sending/publishing,
purchase/payment, high-impact external actions, and ambiguous sensitive-domain
prompts. Production configs may opt into narrow escape hatches under
`safety.confirmation_overrides`, but the router does not learn from approvals or
relax rules implicitly.

Keep escape hatches visible in versioned config and pair them with application
tests. Invalid config, undefined routes, unavailable engines without compatible
fallbacks, and fallback cycles still fail closed to `human_confirm`.

## Startup Checks

Production processes should validate config during startup:

```python
from model_router import ModelRouter

router = ModelRouter.from_config("configs/model_router.yaml")
```

Startup validation loads YAML, validates static config, and caches availability.
If startup must proceed without external availability checks, use
`validate_availability=False` deliberately and document that choice in the
deploying service.

The default catalog ships as package data. Explicit `--config` paths and
`ModelRouter.from_config(path)` still override it for local or application-
specific catalogs.

ModelRouter does not currently include any host-specific plugin manifest or
adapter. Embeddings should use the stable Python API unless and until a target
application's actual integration contract is implemented.

## Proxy Operations

For local agents, the production-like smoke path is:

```bash
model-router init --auto --yes
model-router doctor --config ~/.model-router/routing_proxy.yaml
model-router-proxy --config ~/.model-router/routing_proxy.yaml --log-level info
```

For local admin/config work, `model-router settings --config-dir ~/.model-router`
runs a localhost-only settings UI on `127.0.0.1:8099` by default. It can edit
proxy config fields, run doctor, start/stop/restart the proxy as a child
process, show the latest actual route receipt, render the routing map from the
configured engine/backend/fallback policy, inspect model discovery and
recommendations, plan explicit downloads, inspect recent telemetry, show local
and workflow benchmark status, run confirmed local backend benchmarks, and write
feedback labels. It is not a chat UI, does not submit user prompts, and does not
display literal API keys or raw prompt text.

The product north star in `docs/product-north-star.md` defines the intended UI
shape: a local AI control center and proxy routing plane for model discovery,
recommendations, explicit downloads, routing policy, runtime status, route
receipts, safety gates, telemetry, and feedback labeling. ModelRouter can be the
integrated local surface for common workflows or sit above external runtimes
such as LM Studio, Ollama, LocalAI, llama.cpp, MLX/MLX-LM, vLLM, and hosted
providers. It is directional product truth, not a guarantee that every pictured
control is implemented in the current release.

Setup recommendations treat RAM as a fit/load gate, then use CPU architecture,
CPU core count, Apple Silicon/Metal, CUDA/ROCm hints, runtime format, model
size, quantization, and optional benchmark results for usability scoring. Run
`model-router setup benchmark --config ~/.model-router/routing_proxy.yaml` for a
dry plan, and add `--execute --yes` only when you want local synthetic benchmark
requests. Results are stored under `~/.model-router/benchmarks.json` without
prompt bodies, request bodies, API keys, or secrets.

Use the matching OpenAI-compatible agent settings:

```text
Base URL: http://127.0.0.1:8082/v1
Model: model-router
API key: leave blank unless proxy.api_key or proxy.api_key_env is configured
Chat endpoint: /v1/chat/completions
Responses endpoint: /v1/responses
```

`doctor` should be part of background-service setup. It verifies proxy YAML,
router YAML, backend reachability, and advertised backend model ids when the
upstream exposes `/v1/models`. It should fail clearly when LM Studio, Ollama,
or another local upstream is not running, instead of leaving the agent to debug
a generic connection error.

Readable `doctor` output includes the agent base URL, telemetry log path, and
next-step remediation such as starting Ollama, starting the LM Studio local
server, pulling missing Ollama models, or editing LM Studio model ids.

Managed local runtimes are opt-in per backend. When `runtime.enabled: true`, the
proxy starts only the configured argv command, waits for the configured
readiness URL, captures stdout/stderr to the configured log file, keeps the
process warm while it is active, stops it after the idle timeout, and stops all
managed child processes during proxy shutdown. It does not use a shell, infer
commands, download models, or restart crashing runtimes in a loop.

For `mlx-lm` runtimes, the first supported upstream shape is
`/v1/chat/completions` plus `/v1/models`. `/v1/responses` is not translated for
MLX-LM; use an upstream that supports Responses API when clients need that
endpoint.

## Real Proxy Dogfood

Use the dogfood harness before a release or after changing proxy behavior:

```bash
model-router dogfood proxy --config ~/.model-router/routing_proxy.yaml
```

The default command is a plan only. It performs no HTTP requests and does not
start a proxy, start runtimes, enable hosted providers, download models, or run
verifiers. When local runtimes are deliberately available, opt into live checks:

```bash
model-router dogfood proxy \
  --config ~/.model-router/routing_proxy.yaml \
  --execute
```

The live harness covers `/health`, `/v1/models`, `/v1/chat/completions`,
streaming chat, `/v1/responses` when the configured backend supports it,
fallback visibility, backend policy rejection, `human_confirm`, and verifier
mode visibility. Missing runtimes and unsupported endpoints skip clearly unless
`--require-running` is set. Smoke prompts are fixed and sanitized, and report
output does not serialize prompt bodies.

## Regression Coverage

Production readiness is guarded by:

- CI on pushes and pull requests.
- A benchmark regression check for initialized `route_fast(...)`.
- Offline workflow benchmarks for common route outcomes, profile behavior,
  human confirmation, and route-change evidence.
- Catalog status/diff/apply tests that cover no-network defaults, confirmation
  requirements, backups, migration logs, and no-op behavior.
- Proxy dogfood harness tests that keep live runtime checks opt-in and verify
  skip/fail behavior without requiring live providers in CI.
- API contract tests that keep `route_fast` as the string-only production API
  and `route` as the diagnostic/audit API.
- Adversarial and fuzz tests for deterministic behavior and fail-closed handling
  of destructive, sending, purchasing, deployment, and other external actions.
- Proxy streaming tests that cover mocked upstream interruption, generator
  cleanup on close, and a live uvicorn/raw-socket client disconnect against a
  controlled ASGI upstream. The live disconnect test verifies upstream stream
  finalization and metadata-only logging with prompt capture disabled.

Release evidence should include:

```bash
python -m ruff check .
python -m pytest
python scripts/check_route_fast_latency.py --json
model-router workflow-benchmark --json --fail-on-mismatch
model-router dogfood proxy --config ~/.model-router/routing_proxy.yaml
```

Run `model-router dogfood proxy --execute` only as a manual local-runtime check
against the runtimes you intend to support for that release.
