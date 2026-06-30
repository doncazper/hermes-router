# Session-Aware Routing Design

Status: future optional design. This document describes a possible extension
point for host agents. It is not implemented today and must not change the
default `route_fast(...)` contract.

## Purpose

ModelRouter's default job is to route one request to an engine category with
deterministic policy, receipts, and safety gates. Fusion-like multi-agent
harnesses need another shape: they may want to reconsider model and provider
choice when a task reaches a meaningful phase boundary.

The session-aware extension would give host agents a way to ask:

```text
Given the current task phase and policy constraints, should this work stay on
the current route, move to a stronger route, move to a cheaper route, or require
human confirmation?
```

ModelRouter would still only decide. It would not run tools, manage context,
spawn workers, delegate subtasks, monitor a sidekick, compact transcripts, or
review code. The host harness owns all execution behavior.

## Non-Goals

- Do not change `ModelRouter.route_fast(prompt, hints=None) -> str`.
- Do not add hidden planner/worker orchestration to the proxy.
- Do not store or reconstruct session context inside ModelRouter.
- Do not call a model classifier on the default routing path.
- Do not perform provider calls, tool calls, shell commands, Git operations, or
  verifier calls from this API.
- Do not accept raw prompts or transcripts by default when summaries and
  structured metadata are enough.

## Phase Boundaries

The first version should use a closed enum so host integrations and receipts can
be tested deterministically:

| Phase | Host-agent meaning | Routing question |
| --- | --- | --- |
| `initial_prompt` | User request has arrived. | What route should start the task? |
| `plan_ready` | The host agent has produced a plan or task decomposition. | Does the plan reveal higher/lower complexity or stricter policy needs? |
| `context_compaction` | The host is about to compact or summarize context. | If cache/context cost is already being paid, should the next route change? |
| `delegation_candidate` | The host is considering assigning work to a worker. | Is the candidate route suitable under cost, risk, and provider policy? |
| `verification` | The host is about to verify output or inspect results. | Should verification use the same route, a cheaper route, or a stricter route? |
| `final_review` | The host is preparing final review or handoff. | Should final judgment use a stronger or safer route or human confirmation? |

Unknown phases should be rejected as invalid input rather than treated as
free-form prompts.

## Future API Shape

The API should be separate from `route_fast(...)` and the existing diagnostic
`route(...)` call. A possible Python shape:

```python
decision = router.route_session_phase(
    phase="context_compaction",
    task_summary="Migrate the repository from deprecated tracing APIs.",
    current_engine="code_agent",
    prior_receipt_id="req_123",
    phase_summary="Plan is stable; remaining work is mechanical edits plus tests.",
    hints={
        "profile": "balanced",
        "max_cost_tier": "medium",
        "hosted_allowed": True,
    },
    context={
        "estimated_context_tokens": 48000,
        "compaction_expected": True,
        "files_touched_estimate": 24,
        "host_confidence": "medium",
        "candidate_action": "keep_or_downgrade",
    },
)
```

The exact names can change during implementation, but the boundary should stay
stable:

- Inputs are summaries, metrics, current route state, phase name, and routing
  hints.
- Outputs are route recommendations and receipts.
- No hidden session state is required; callers pass the context they want
  considered on each call.

## Expected Inputs

Required inputs:

- `phase`: one of the six phase boundaries above.
- `task_summary`: short host-provided summary of the user request or current
  task. Raw prompt text should be optional and marked sensitive if supplied.
- `current_engine`: selected engine currently in charge, or `null` at
  `initial_prompt`.
- `hints`: existing routing hints/profile/provider constraints.

Optional inputs:

- `prior_receipt_id`: last ModelRouter receipt or proxy request id.
- `phase_summary`: host-provided summary of what changed since the last route.
- `current_backend` and `current_model`: proxy/runtime details when available.
- `context.estimated_context_tokens`: host-estimated session size.
- `context.compaction_expected`: whether this call is at a natural cache miss or
  compaction boundary.
- `context.files_touched_estimate`: approximate repo breadth.
- `context.test_or_verification_cost`: low, medium, high, or unknown.
- `context.delegation_candidate`: worker route or engine being considered.
- `context.host_confidence`: low, medium, high, or unknown.
- `context.outcome_so_far`: not_started, progressing, blocked, failed,
  partial, or succeeded.
- `context.risk_notes`: structured host flags for destructive, external,
  credential, deployment, or data-sensitive concerns.

All optional fields are advisory. Configured provider policy, safety policy, and
fallback rules remain authoritative.

## Expected Outputs

A future `SessionRoutingDecision` should be JSON-safe and receipt-first:

- `phase`: normalized phase.
- `selected_engine`: recommended route for the next phase.
- `previous_engine`: current engine supplied by the host.
- `route_action`: `start`, `keep_current`, `switch_engine`,
  `tighten_policy`, `require_human_confirm`, or `reject_invalid`.
- `routing_profile`: effective profile.
- `fallback_engine`: compatible fallback, if configured.
- `requires_confirmation`: whether the host must stop for confirmation.
- `policy_constraints`: provider/cost/latency constraints applied.
- `switch_reason`: concise reason for keeping or changing route.
- `receipt`: embedded diagnostic receipt fields.
- `host_next_action`: plain-language guidance such as "continue on current
  engine", "re-route next call", or "ask for human confirmation".

The output is advisory except for safety failures. Host agents decide whether to
continue, start a worker, compact context, run tests, or review output.

## Receipt Fields

Session-aware receipts should preserve existing route receipt fields and add a
small phase block:

```json
{
  "session_phase": "context_compaction",
  "previous_engine": "code_agent",
  "route_action": "keep_current",
  "switch_reason": "Mechanical repo work remains, but provider policy and cost tier still allow code_agent.",
  "compaction_boundary": true,
  "delegation_candidate": null,
  "host_context_summary_present": true,
  "host_context_stored": false,
  "orchestration_boundary": "host_agent_owns_execution"
}
```

Suggested reason code prefixes:

- `session.phase.initial_prompt`
- `session.phase.plan_ready`
- `session.phase.context_compaction`
- `session.phase.delegation_candidate`
- `session.phase.verification`
- `session.phase.final_review`
- `session.action.keep_current`
- `session.action.switch_engine`
- `session.action.require_human_confirm`
- `session.boundary.host_owns_execution`

Raw prompts and transcripts should remain absent from receipts unless explicit
prompt capture is already configured.

## Failure And Safety Behavior

The extension must fail closed and stay deterministic:

- Invalid phase, invalid hints, invalid context schema, or unavailable selected
  route returns `route_action: reject_invalid` with `selected_engine:
  human_confirm` where a routing decision is still emitted.
- High-risk external actions, destructive actions, sending, purchasing,
  deployment, credential exposure, or ambiguous high-impact work must route to
  `human_confirm` unless the existing safety config explicitly permits the
  narrow action.
- Caller hints may narrow provider policy, but they cannot loosen configured
  provider allowlists, denylists, local-only mode, hosted-provider policy, max
  cost tier, or max latency tier.
- If the host proposes a delegation candidate that violates safety or provider
  policy, ModelRouter should reject that candidate in the receipt; it should not
  choose or launch a different worker.
- If current route metadata is missing, the router can still make a decision
  from `task_summary` and hints, but the receipt should mark previous route
  state as unknown.

## Difference From `route_fast(...)`

`route_fast(...)` remains the production hot path:

- It routes one prompt.
- It returns only an engine string.
- It does not build receipts.
- It does not inspect session phase.
- It does not log or store prompt/session state.
- It stays under the existing latency guard.

The session-aware extension is a future diagnostic/control-plane API:

- It runs only when a host agent deliberately calls it.
- It may build receipts and reason codes.
- It can accept structured session metadata supplied by the host.
- It can recommend keeping or changing route at a phase boundary.
- It remains outside default proxy forwarding and outside `route_fast(...)`.

This separation lets ModelRouter support Fusion-like harnesses without turning
the local proxy into an agent workspace.

## Example Host-Agent Flow

```text
1. initial_prompt
   Host receives: "Remove deprecated tracing across the repo and open a PR."
   Host calls route_session_phase with task_summary and default profile.
   ModelRouter recommends code_agent and records coding/safety policy.

2. plan_ready
   Host planner identifies mostly mechanical removals across 40 files and a
   slow test suite.
   Host calls route_session_phase with phase_summary and files_touched_estimate.
   ModelRouter keeps code_agent but marks the work as broad repo work; no worker
   is launched by ModelRouter.

3. delegation_candidate
   Host considers assigning mechanical edits to a cheaper worker.
   Host calls route_session_phase with delegation_candidate="fast_local" or a
   configured worker route.
   ModelRouter returns whether that candidate fits provider/cost/safety policy.
   Host decides whether to delegate.

4. context_compaction
   Before compacting, host calls route_session_phase with
   compaction_expected=true and outcome_so_far=progressing.
   ModelRouter may recommend keep_current, switch_engine, or tighten policy.
   Host performs compaction and routes its next model call accordingly.

5. verification
   Host is ready to run or inspect tests.
   ModelRouter recommends the verification route or confirms the current route
   under configured policy. Host owns the test command and result parsing.

6. final_review
   Host asks whether final judgment should remain on the current route.
   ModelRouter may recommend a stronger reasoning/code route or human_confirm
   for risky external actions such as opening a PR.
   Host performs the review and any external action only after its own controls.
```

## Implementation Gates

Only implement this after the simpler delegation-suitability and telemetry work
has enough evidence to justify the API surface. The first implementation should
meet these gates:

- Disabled by default and absent from `route_fast(...)`.
- Deterministic with no model classifier or provider call.
- JSON-schema-compatible request and receipt models.
- Backward-compatible receipts and telemetry readers.
- Tests for all six phase values, invalid phase rejection, provider-policy
  rejection, `human_confirm` safety behavior, and no raw prompt persistence by
  default.
- `python scripts/check_route_fast_latency.py --json` remains unchanged.
