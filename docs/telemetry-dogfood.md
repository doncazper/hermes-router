# Routing Telemetry Dogfood Loop

Use this loop when dogfooding ModelRouter through the local proxy. The goal is
to collect enough real, labeled routing evidence before changing deterministic
rules or revisiting optional classifier-based routing.

## Safe Defaults

Enable proxy observability with redacted prompt previews:

```yaml
observability:
  enabled: true
  log_path: ~/.model-router/logs/routing-events.jsonl
  prompt_capture: redacted_preview
```

The proxy logs request metadata, selected engine, routing profile, scores,
feature flags, backend, fallback status, latency, prompt hash, prompt length,
estimated tokens, and a redacted preview. It does not log raw prompts unless
`prompt_capture: full` or `MODEL_ROUTER_LOG_PROMPTS=1` is explicitly enabled.

Routing decisions use configured route/backend metadata such as `cost_tier` and
`latency_tier`. They do not fetch live pricing, and future cost telemetry must
not change `route_fast(...)`, `route(...)`, or proxy forwarding decisions.

When provider policy rejects an engine, the diagnostic receipt includes the
policy reason and rejected engine without raw prompt text. When proxy backend
policy rejects forwarding, the event status is `backend_policy_rejected`, the
proxy returns a structured error, and no upstream backend is called.

When proxy observability has a diagnostic decision, routing events also include
privacy-safe receipt fields such as `receipt_summary`, `reason_codes`,
`policy_explanation`, `fallback_explanation`, `safety_explanation`,
`privacy_explanation`, and `wrong_route_next_action`. These fields are derived
from routing metadata and do not include raw prompts.

If the optional verifier is enabled, routing events include verifier metadata:
`verification_mode`, `verification_status`, `verification_backend`,
`verification_status_code`, `verification_latency_ms`, and
`verification_error`. Streaming requests report `skipped_streaming`; default
configs keep verification `off`.

Telemetry inspection commands do not print raw prompt text. Feedback notes are
hidden by default because they may contain private context.

## Cost And Outcome Direction

Cost and outcome telemetry should separate four concepts:

- **Cost tiers**: configured metadata such as `cost_tier`, `latency_tier`,
  selected engine, selected backend, selected model, backend model, routing
  profile, and policy constraints. These are stable enough for routing policy.
- **Actual usage**: upstream response metadata when a provider already returns
  it, such as prompt tokens, completion tokens, total tokens, cached input
  tokens, and upstream model id. The proxy should record these when available
  without buffering streaming responses or making provider-specific follow-up
  calls.
- **Estimated cost**: best-effort cost estimates only when the local versioned
  pricing catalog has a matching provider/model entry and usage counts are
  present. Missing price or missing usage means no estimate.
- **Outcome labels**: explicit user/operator feedback such as `success`,
  `partial`, `failed`, `wrong_route`, `too_expensive`, or `too_slow`. The router
  must not infer task success from status codes, route choice, or verifier
  status.

Reporting fields:

| Group | Fields |
| --- | --- |
| Route identity | `selected_engine`, `routing_profile`, `selected_backend`, `selected_model`, `backend`, `backend_model`, `status`, `status_code`, `fallback_used` |
| Latency | `route_latency_ms`, `diagnostic_latency_ms`, `upstream_latency_ms`, `total_latency_ms` |
| Usage | `usage_prompt_tokens`, `usage_completion_tokens`, `usage_total_tokens`, `usage_cached_input_tokens`, `upstream_model` |
| Cost tier | `configured_cost_tier`, `configured_latency_tier`, `cost_estimate_available` |
| Estimated cost | `estimated_input_cost`, `estimated_output_cost`, `estimated_cached_input_cost`, `estimated_total_cost`, `estimated_cost_currency`, `pricing_catalog_version`, `pricing_catalog_source`, `pricing_source`, `pricing_effective_date`, `pricing_match_status` |
| Outcomes | `outcome_label`, `feedback_label`, `expected_engine`, `feedback_notes_present` |

Pricing catalog metadata is local and versioned. Maintaining it is an explicit
operator action:

```bash
model-router pricing status
model-router pricing diff
model-router pricing apply --yes
```

These commands preview local packaged metadata, require confirmation for writes,
and never run during routing or proxy forwarding. They do not fetch live pricing
or scrape provider pages.

## Identify Routes Live

Routed `/v1/chat/completions` and `/v1/responses` responses include
privacy-safe headers:

```text
X-ModelRouter-Request-ID
X-ModelRouter-Engine
X-ModelRouter-Profile
X-ModelRouter-Backend
X-ModelRouter-Fallback
X-ModelRouter-Route-API
```

These headers never include raw prompts, request bodies, API keys, or upstream
secrets. When a route feels wrong, copy `X-ModelRouter-Request-ID` immediately
and label it later with `model-router feedback`.

When `model-router-proxy` exits, it prints a concise session summary with event
count, engine/backend/status counts, fallback/interruption/error counts, and the
`model-router telemetry summary ...` command for the configured log paths. This
summary is best-effort and does not read prompts or call upstream services.

## Inspect Coverage

For a local visual workflow, run:

```bash
model-router settings --config-dir ~/.model-router
```

The settings UI shows the latest safe route receipt, recent request ids, route
headers/session telemetry, and a review form that writes the same JSONL labels
as `model-router feedback`. It does not show raw prompts or API keys and it has
no chat prompt box.

Summarize collected events, labels, replayability, mismatch groups, private
events that were skipped, and unlabeled replayable request ids:

```bash
model-router telemetry summary \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl
```

Use JSON output for scripts:

```bash
model-router telemetry summary \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl \
  --json
```

Inspect labels without exposing notes:

```bash
model-router telemetry feedback \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl
```

Add `--include-notes` only when the output target is safe for any private
context you put in feedback notes.

Review unlabeled route events as a local triage queue:

```bash
model-router telemetry review \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl
```

The review queue lists request ids, selected engine, status, backend, routing
profile, receipt summaries, reason codes, replayability, and a suggested
`model-router feedback` command. It omits raw prompts, prompt previews, request
bodies, feedback notes, API keys, and secrets by default. Private/no-prompt
events can still appear as non-replayable metadata rows so operators can label
them by request id.

## Label Wrong Routes

When a route is wrong, copy `X-ModelRouter-Request-ID` from the proxy response,
or copy the matching `request_id` from the routing event, and label the intended
engine:

```bash
model-router feedback req-123 code_agent \
  --notes "repo prompt routed too small"
```

For a one-off explanation without reading JSON:

```bash
model-router decide --explain "fix the repo and run tests"
```

Then replay labeled traffic against the current router:

```bash
python scripts/replay_routing_log.py \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl \
  --json \
  --fail-on-regression
```

Rows without full prompts are skipped for replay, but they still count toward
event coverage and skipped-private counts. If a private event needs replay,
reproduce the case intentionally with `prompt_capture: full`, or create a
sanitized fixture prompt that preserves the routing behavior without private
content.

## Promote Regressions

Promote recurring wrong-route clusters into checked-in fixtures or parametrized
tests only after replay shows the pattern. Keep the fixture prompt sanitized,
rerun replay before and after any scoring change, and keep the regression test
with the fix.

Use the offline workflow benchmark for broad release evidence:

```bash
model-router workflow-benchmark --json --fail-on-mismatch
```

The workflow report has the same privacy posture as telemetry summaries:
prompt hashes, expected/selected routes, receipt summaries, reason codes,
delegation suitability signals, policy/fallback/safety explanations, and
route-change counts, but no prompt bodies. Promote stable wrong-route clusters
into workflow fixtures only after sanitizing the prompt enough to preserve
routing behavior without private content.

When proxy responses include upstream usage metadata, telemetry summaries may
show aggregate prompt, completion, total, and cached-input token counts by
route, backend, and model. These fields are usage telemetry only: they do not
include prompt or response text and do not imply task success. When a local
pricing catalog has a matching model entry, summaries may also show estimated
input, output, cached-input, and total cost. Missing catalog entries are shown
as pricing match statuses rather than invented prices.

To use a local override for CLI reporting:

```bash
model-router telemetry summary \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl \
  --pricing-catalog ~/.model-router/pricing_catalog.yaml
```

To draft local override entries from missing catalog coverage, add
`--pricing-override-skeleton` to telemetry summary or review. The generated YAML
uses provider/model/source fields with zero placeholder prices, so operators can
copy it into `~/.model-router/pricing_catalog.yaml` and replace the placeholders
with verified pricing. The settings UI shows the same action as **Copy override
skeleton** when coverage gaps exist. This is a local remediation helper only; it
does not fetch prices, apply files, or affect routing.

## Maintain Catalogs

Use the packaged catalog workflow when updating local recommendations or
checking whether a user config has drifted from packaged defaults:

```bash
model-router catalog status --config ~/.model-router/model_router.yaml
model-router catalog diff --config ~/.model-router/model_router.yaml
```

These commands perform no remote checks and write nothing. Apply packaged
defaults only after reviewing the diff:

```bash
model-router catalog apply --config ~/.model-router/model_router.yaml --yes
```

Apply backs up the existing config and writes a migration log entry. Treat the
result as a maintenance action, not as telemetry-driven auto-tuning.

Maintain pricing metadata separately:

```bash
model-router pricing status --override ~/.model-router/pricing_catalog.yaml
model-router pricing diff --override ~/.model-router/pricing_catalog.yaml
model-router pricing apply --override ~/.model-router/pricing_catalog.yaml --yes
```

The packaged pricing catalog contains local/control-plane defaults and
non-authoritative examples. Add operator-verified provider prices through the
override file before using estimates for spend review.

## Advanced Routing Threshold

Do not reopen optional advanced routing for one-off mistakes. Revisit a
second-pass classifier only when dogfood data has roughly 20-30 labeled wrong
routes with repeated patterns that deterministic scoring cannot fix without
regressing golden, parity, adversarial, or replay fixtures.
