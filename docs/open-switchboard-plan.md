# Open Switchboard Robustness Plan

## Positioning

ModelRouter should not become a hidden multi-agent orchestrator. Its stronger
lane is the open switchboard: one local OpenAI-compatible endpoint, transparent
routing, explicit safety gates, user-owned providers, and replayable evidence
for every routing change.

That lane can still support Fusion-like products. A host harness can use
ModelRouter as its routing and policy control plane, then keep task execution,
context management, sidekick delegation, monitoring, and final review in the
host application where those responsibilities are visible and testable.

The goal of this plan is to take the best product lessons from hosted
orchestration systems without giving up ModelRouter's core promises:

- Deterministic default routing.
- Local-first operation.
- User-owned model and provider policy.
- Receipts and telemetry instead of black-box decisions.
- Clear separation between routing policy and agent orchestration.
- Optional execution features outside `route_fast(...)`.
- Human confirmation for high-risk actions.

The canonical local UI direction is captured in
`docs/product-north-star.md`. Future settings work should move toward that
local proxy control-center shape: route map, runtime status, receipts, safety
gates, telemetry, and feedback labeling, without adding chat, agent behavior, or
silent config changes.

A future session-aware routing extension is sketched in
`docs/session-aware-routing.md`. It keeps phase-boundary re-routing as an
optional host-agent control-plane API, not as hidden orchestration in the proxy
or the `route_fast(...)` hot path.

## Non-Goals

- Do not add hidden planner-worker-synthesizer behavior to the default proxy.
- Do not make ModelRouter responsible for task execution, context management,
  worker delegation, or final agent review.
- Do not add LLM classification to the production path without the replay gates
  in `docs/advanced-routing.md`.
- Do not silently enable hosted providers, downloads, benchmarks, or verifier
  calls.
- Do not make the settings UI a chat app or agent surface.
- Do not weaken `human_confirm` defaults for destructive, sending, purchasing,
  deployment, or ambiguous high-impact requests.

## Track 1: Routing Profiles

Status: Track 1 base implementation is in place for CLI hints, proxy defaults,
settings UI defaults, receipts, and local-only private routing. Per-request
proxy metadata remains deferred until there is a clear trust boundary and
configuration model for client-supplied profile changes.

Goal: give users plain-language modes before engine names.

Ship named profiles that compile down to existing hints and config constraints:

- `fast`: prefer low latency and local/free backends.
- `balanced`: current default behavior.
- `quality`: allow stronger reasoning and hosted fallbacks when configured.
- `private`: local-only, no hosted API backends.
- `safe`: strict confirmation and conservative fallback behavior.

Implementation notes:

- Add a profile model separate from engine categories.
- Make profiles usable from CLI, proxy config, and settings UI.
- Keep direct engine names available for advanced users.
- Compile profiles into `RoutingHints` or proxy-side constraints so the router
  core remains small.
- Add tests proving profiles do not bypass `human_confirm`.

Done when:

- `model-router decide --profile private "research this"` produces an auditable
  decision that excludes hosted backends.
- The proxy can set a default profile without trusting arbitrary client input by
  default.
- The settings UI lets users pick a default profile without editing YAML.
- `route_fast(...)` latency guard remains green.

## Track 2: Provider Pools And Policy Controls

Status: Track 2 base implementation is in place. Router provider policy is
versioned in `model_router.yaml`; proxy backend policy is versioned in
`routing_proxy.yaml`; CLI hints, receipts, fallback resolution, proxy forwarding,
health/settings visibility, and tests cover the constraint path. Richer policy
editing in the settings UI can build on this foundation.

Goal: make provider control a first-class product surface.

Add policy controls for:

- Provider allowlists and denylists.
- Backend allowlists and denylists.
- Local-only and hosted-allowed modes.
- Max cost tier and max latency tier presets.
- Optional per-route provider pools, such as local-only for simple work and
  hosted-allowed for reasoning.

Implementation notes:

- Reuse existing provider, cost tier, latency tier, availability, and fallback
  concepts before adding new abstractions.
- Keep provider and backend policy versioned in YAML and visible in the
  settings UI.
- Make rejected providers/backends visible in receipts, diagnostics, health, or
  proxy error responses depending on where the policy applies.
- Ensure fallback resolution never jumps into a denied provider or backend.

Done when:

- A user can configure "never use hosted APIs" once and see that policy honored
  in CLI, proxy diagnostics, settings, and receipts.
- Receipts explain provider-policy rejections.
- Tests cover fallback chains, unavailable backends, forced engines, and
  high-risk prompts under restrictive pools.

## Track 3: Productized Receipts

Status: Track 3 base implementation is in place. Receipts now preserve the
existing JSON fields and add a concise summary, stable reason codes, selected
route, policy, rejection, fallback, safety, privacy, and wrong-route next-action
explanations. `model-router decide --explain`, proxy telemetry summaries/codes,
and the settings receipt panel expose the same privacy-safe product language.

Goal: make transparency a feature, not just diagnostics.

Improve route receipts so a normal user can understand:

- Why this route was selected.
- Which engines were rejected and why.
- Which policy constraints mattered.
- Which fallback was used.
- Whether the request needs tools, freshness, vision, image generation, or
  confirmation.
- What to change when the route was wrong.

Implementation notes:

- Add a concise human-readable receipt summary alongside the current JSON-safe
  fields.
- Add stable reason codes while preserving existing reason strings for
  compatibility.
- Add a `model-router explain` or `model-router decide --explain` view optimized
  for humans.
- Surface the same explanation in settings UI telemetry detail.
- Keep raw prompt handling privacy-safe.

Done when:

- A route can be explained without reading source code or raw JSON.
- Wrong-route feedback can be started from a receipt/request id.
- Existing receipt serialization tests still pass, with focused tests for new
  reason codes and summaries.

## Track 4: Explicit Verification Boundary

Status: Track 4 base implementation is in place. The proxy now has a versioned
`verifier` config with `off`, `receipt-only`, `sampled`, and
`always-for-risky-output` modes. Verification is disabled by default, runs only
after proxy forwarding, skips streaming instead of buffering, logs privacy-safe
metadata, and can either log-only or fail closed when explicitly configured.

Goal: offer reliability checks without hidden orchestration.

Add an optional verifier boundary that is explicit, observable, and disabled by
default:

```text
route -> forward to selected backend -> optional verifier -> final response
```

Possible verifier modes:

- `off`: default.
- `receipt-only`: record whether the request would qualify for verification.
- `sampled`: verify a configured percentage of low-risk requests.
- `always-for-risky-output`: verify selected route classes, but never bypass
  human confirmation for risky actions.

Implementation notes:

- Keep verifier configuration outside router scoring.
- Treat verifier calls as proxy/runtime behavior, not `route_fast(...)` work.
- Make verifier backend, prompt template, latency budget, and failure behavior
  explicit in config.
- Log verifier metadata without raw prompts unless full prompt capture is
  deliberately enabled.
- Never let a verifier perform external actions.

Done when:

- Operators can opt into verification and see verifier route ids, backend,
  latency, and outcome in telemetry.
- Verification failures have clear proxy behavior.
- Tests cover disabled default, sampled behavior with deterministic test hooks,
  streaming compatibility, and privacy-safe logs.

## Track 5: Workflow Benchmarks

Goal: measure practical routing outcomes, not only router speed.

Status: implemented for offline workflow correctness benchmarks. The
`model-router workflow-benchmark` command exercises sanitized fixtures for
simple, balanced, coding, research, vision, image generation, safety,
private-profile, quality-profile, and sidekick-delegation-shaped routes; emits
readable or JSON reports; serializes prompt hashes instead of prompt bodies; and
performs no backend, verifier, download, or hosted API calls.

Add benchmark suites for common workflows:

- Simple rewrite routes to fast local.
- Ordinary summary routes to balanced local.
- Repo/test work routes to code.
- Current-information requests route to research.
- Image/screenshot requests route to vision.
- Image generation requests route to image generation.
- Risky external actions route to `human_confirm`.
- Private profile excludes hosted APIs.
- Quality profile can use stronger configured backends.
- Sidekick-delegation task shapes expose expected mechanical, verification-heavy,
  judgment-heavy, repo-wide, and risky/external-action receipt signals.

Implementation notes:

- Keep the existing microsecond latency guard as a hard contract.
- Add a correctness benchmark based on sanitized prompts and expected engines.
- Add profile-specific benchmark fixtures.
- Add a readable benchmark report that can be pasted into releases.
- Keep any live backend benchmark opt-in and synthetic.

Done when:

- CI can run offline routing correctness benchmarks.
- Release notes can report route correctness, route changes, and latency.
- Replay logs and benchmark fixtures share enough shape to promote real
  wrong-route clusters into tests.

## Track 5A: Cost And Outcome Telemetry

Goal: measure actual usage and user-labeled outcomes without turning pricing
into a routing dependency.

Status: design direction only. Routing still uses configured cost tiers and
provider policy. `route_fast(...)`, `route(...)`, and default proxy forwarding
must not fetch live pricing or scrape provider pages.

Add telemetry support for:

- Upstream usage fields when a provider already returns them.
- Configured cost and latency tiers for the selected route/backend/model.
- Optional exact cost estimates only from a future local versioned pricing
  catalog.
- Explicit outcome labels supplied by users/operators, not inferred task
  success.

Implementation notes:

- Keep pricing lookup outside `route_fast(...)` and `route(...)`.
- Do not buffer streaming responses just to find usage.
- Treat missing usage or missing catalog entries as "estimate unavailable."
- Make pricing catalog refresh an explicit status/diff/apply workflow later.
- Keep old JSONL telemetry readers tolerant of missing cost/outcome fields.

Done when:

- Proxy telemetry can record usage from mocked OpenAI-compatible responses.
- Feedback/outcome labels are optional and privacy-safe.
- Cost estimates are clearly marked unavailable until a local catalog exists.
- Documentation explains cost tiers, actual usage, estimates, and outcomes as
  separate concepts.

## Track 6: Catalog Update Workflow

Goal: make model and preset updates feel maintained without silently changing
users' routing policy.

Status: implemented for packaged catalog maintenance. The
`model-router catalog status|diff|apply` commands check packaged metadata,
preview local config changes, and apply packaged router catalog defaults only
after confirmation. Apply backs up an existing config, records a JSONL
migration entry, performs no remote checks, and does not download models or
change hosted/provider policy silently. The settings UI state and dashboard
surface catalog status.

Add a catalog workflow for:

- Checking the installed catalog version.
- Listing newer packaged or remote catalog versions.
- Previewing model, preset, route, and recommendation changes.
- Applying updates only after confirmation.
- Recording what changed in local config comments or a separate migration log.

Implementation notes:

- Start with packaged catalog metadata before remote update delivery.
- Treat remote fetching as a separate, explicit feature with clear provenance.
- Never overwrite user-edited local configs without backup and diff.
- Keep setup recommendations and runtime benchmarks advisory.

Done when:

- `model-router catalog status` explains current catalog and local overrides.
- `model-router catalog diff` shows candidate changes without applying them.
- The settings UI can show available catalog updates and require confirmation.
- Tests cover config preservation, backups, and no-network default behavior.

## Track 7: Product Language And Onboarding

Goal: make the product promise obvious.

Status: implemented for docs and onboarding language. The README opens with the
open-switchboard promise, the first-run flow explains profile/provider/receipt
steps, and production docs preserve the boundary between routing, proxy
forwarding, optional verification, workflow benchmarks, catalog maintenance,
and future adapter execution.

Adopt this message in docs and release copy:

```text
ModelRouter is the open switchboard for AI model routing: one local
OpenAI-compatible endpoint that routes each request to the right model, with
receipts, safety gates, and full provider control.
```

Implementation notes:

- Keep the README proxy-first.
- Explain profiles before engine internals.
- Put receipts and provider control near the top-level value proposition.
- Avoid claiming ModelRouter is a general multi-agent system.
- Show how ModelRouter can route to local models, hosted APIs, code agents, web
  research, vision, image generation, and future custom backends.

Done when:

- A new user can explain the product in one sentence.
- Setup docs show "choose a profile, choose providers, inspect receipts" as the
  normal path.
- The docs preserve the distinction between routing, proxy forwarding, optional
  verification, and future adapter execution.

## Suggested Implementation Order

1. Routing profiles.
2. Provider pools and policy controls.
3. Productized receipts.
4. Workflow benchmarks.
5. Catalog update workflow.
6. Explicit verification boundary.
7. Product language and onboarding polish throughout the docs.

This order keeps user-facing clarity and safety policy ahead of more complex
runtime behavior. Verification comes after receipts and benchmarks so there is
evidence for where it helps.

## Global Acceptance Gates

- `python -m ruff check .`
- `python -m pytest`
- `python scripts/check_route_fast_latency.py --json`
- New tests for every profile, provider-policy, receipt, benchmark, catalog, and
  verification behavior.
- No new default network calls.
- No raw prompt logging unless existing explicit prompt-capture controls are
  enabled.
- No regression that allows high-risk requests to bypass `human_confirm`.

## Ready-To-Paste Coordination Prompt

```text
Please implement the next unfinished track in docs/open-switchboard-plan.md.

Keep ModelRouter in its open-switchboard lane: deterministic default routing,
transparent receipts, local-first provider control, and optional execution
features outside route_fast.

For the selected track:
- Read the relevant existing code and docs before editing.
- Keep changes narrowly scoped and compatible with existing config where possible.
- Add focused tests and docs.
- Run ruff, pytest, and route_fast latency checks.
- Report any migration risks, safety implications, and follow-up tracks.
```
