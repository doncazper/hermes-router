# Optional Advanced Routing Decision

Milestone 7 evaluated whether ModelRouter should add an optional second-pass
classifier for uncertain prompts. The decision is to defer it until replay data
shows recurring labeled errors that deterministic scoring cannot fix cleanly.

## Evidence Reviewed

- Checked-in replay fixtures:
  - `v0_5_proxy_events.jsonl` plus `v0_5_feedback.jsonl`: 4 replayed events,
    4 feedback labels, 0 route changes, 0 expected mismatches.
  - `short_prompt_calibration_events.jsonl` plus
    `short_prompt_calibration_feedback.jsonl`: 7 replayed events, 7 feedback
    labels, 0 route changes, 0 expected mismatches.
- Curated golden, parity, and replay audit: 39 cases, 0 expected mismatches.
- Low-confidence audit: only `fix this` fell below 60 confidence, and it already
  routes upward to `reasoning_local` as intended.
- No local `~/.model-router` JSONL replay or feedback logs were present during
  the audit.
- No matching GitHub issues were found for wrong-route, classifier, replay, or
  feedback problems during the audit.

## Decision

Do not add an optional classifier yet.

The current labeled data does not show unresolved wrong-route patterns. Adding a
classifier now would add product and test surface without evidence of improved
accuracy. Deterministic routing remains the default and only production routing
path.

## Revisit Criteria

Reopen optional advanced routing only when all of these are true:

- Replay logs contain recurring, labeled wrong-route cases.
- The cases cannot be fixed with deterministic scoring rules without causing
  regressions in golden, parity, adversarial, or replay fixtures.
- A candidate classifier reduces expected mismatches on labeled replay data.
- It introduces no new safety regressions, especially around `human_confirm`.
- It is disabled by default, loaded lazily, and isolated outside
  `route_fast(...)`.
- `python scripts/check_route_fast_latency.py --json` still passes with the
  default configuration.

## Residual Risk

The current corpus is intentionally small and privacy-safe. It proves there is
no known labeled need for a classifier, not that future users will never find
ambiguous prompts. The mitigation is the feedback-to-replay workflow: capture
full prompts only during deliberate calibration, label wrong routes with
`model-router feedback`, replay before changing routing, and add regression
tests for any accepted fix.
