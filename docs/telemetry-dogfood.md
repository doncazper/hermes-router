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

The proxy logs request metadata, selected engine, scores, feature flags,
backend, fallback status, latency, prompt hash, prompt length, estimated tokens,
and a redacted preview. It does not log raw prompts unless
`prompt_capture: full` or `MODEL_ROUTER_LOG_PROMPTS=1` is explicitly enabled.

Telemetry inspection commands do not print raw prompt text. Feedback notes are
hidden by default because they may contain private context.

## Identify Routes Live

Routed `/v1/chat/completions` and `/v1/responses` responses include
privacy-safe headers:

```text
X-ModelRouter-Request-ID
X-ModelRouter-Engine
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

## Label Wrong Routes

When a route is wrong, copy `X-ModelRouter-Request-ID` from the proxy response,
or copy the matching `request_id` from the routing event, and label the intended
engine:

```bash
model-router feedback req-123 code_agent \
  --notes "repo prompt routed too small"
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

An interactive `model-router telemetry review` queue is deferred until
dogfooding shows that response headers, telemetry summaries, and shutdown
session summaries are still too clunky.

## Promote Regressions

Promote recurring wrong-route clusters into checked-in fixtures or parametrized
tests only after replay shows the pattern. Keep the fixture prompt sanitized,
rerun replay before and after any scoring change, and keep the regression test
with the fix.

## Advanced Routing Threshold

Do not reopen optional advanced routing for one-off mistakes. Revisit a
second-pass classifier only when dogfood data has roughly 20-30 labeled wrong
routes with repeated patterns that deterministic scoring cannot fix without
regressing golden, parity, adversarial, or replay fixtures.
