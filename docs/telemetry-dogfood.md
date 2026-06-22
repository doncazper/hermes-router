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

When a route is wrong, copy the `request_id` from the routing event and label
the intended engine:

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
