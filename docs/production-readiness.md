# Production Readiness

ModelRouter's production surface is intentionally small: initialize
`ModelRouter` once, keep it in memory, and call `route_fast(prompt)` for live
traffic.

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

## Regression Coverage

Production readiness is guarded by:

- CI on pushes and pull requests.
- A benchmark regression check for initialized `route_fast(...)`.
- API contract tests that keep `route_fast` as the string-only production API
  and `route` as the diagnostic/audit API.
- Adversarial and fuzz tests for deterministic behavior and fail-closed handling
  of destructive, sending, purchasing, deployment, and other external actions.
