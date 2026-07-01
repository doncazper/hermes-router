# ModelRouter Evals / Suitability Benchmarks

ModelRouter Evals answer one local operator question:

> Which model/backend is suitable for this task on this machine?

They are local, operator-facing evidence for routing policy, model
recommendations, runtime selection, and suitability review. They are not public
leaderboard benchmarks, not LM Studio parity claims, not frontier-model quality
claims, and not promises of cost reduction or performance superiority.

Evals should stay runtime-neutral. The same fixture and result shape should work
for LM Studio, Ollama, llama.cpp, MLX/MLX-LM, LocalAI, vLLM, hosted
OpenAI-compatible backends, and generic OpenAI-compatible servers when those
backends are explicitly configured.

## Relationship To Existing Evidence

ModelRouter already has two evidence loops:

- **Workflow benchmarks**: checked-in sanitized prompts that verify deterministic
  route selection, receipts, reason codes, policy behavior, and delegation
  suitability. They do not call model backends.
- **Local backend benchmarks**: explicit setup-time smoke requests against
  configured local backends using one fixed synthetic prompt. They collect
  latency and token-rate evidence without storing prompt bodies or secrets.

ModelRouter Evals sit between those surfaces. They run task-shaped fixtures
against selected backends/models, score the returned behavior with privacy-aware
rules, and produce JSON-safe reports. In the first version, eval results are
evidence only. They should not automatically change `route_fast(...)`,
`route(...)`, proxy forwarding, model assignments, provider policy, or runtime
lifecycle.

Evals are explicit, bounded operator actions. They do not run during setup,
model discovery, model import, runtime detection, `route_fast(...)`,
`route(...)`, proxy forwarding, settings page render, or default proxy
operation. ModelRouter must not recursively benchmark every discovered
GGUF/model file by default.

## Goals

- Help operators compare configured model/backend suitability for real task
  shapes.
- Feed model recommendations and setup guidance with local evidence.
- Give route policy authors evidence before changing thresholds or assignments.
- Preserve provider/runtime neutrality and local-first telemetry boundaries.
- Produce stable, JSON-safe result records that can be summarized in CLI,
  settings, TUI, and release notes.
- Avoid retaining raw prompts or raw model outputs by default.

## Non-Goals

- No benchmark parity claim against LM Studio, public leaderboards, or frontier
  hosted models.
- No automatic routing changes in the first version.
- No hidden installs, downloads, runtime starts, model pulls, or config writes.
- No implicit evals after setup, model discovery/import, runtime detection, or
  route/proxy activity.
- No recursive `--all-ggufs` or automatic benchmark sweep over discovered model
  files.
- No tool execution. Tool-use fixtures may check request/response shape, but
  ModelRouter must not run tools.
- No live pricing fetches. Cost estimates use captured usage plus the local
  versioned pricing catalog only.
- No success inference from latency, token usage, cost, HTTP status, or route
  choice alone.
- No raw prompt, request body, response text, secret, or API key retention by
  default.

## Fixture Categories

Initial fixtures should cover common suitability boundaries:

- `structured_output`: strict JSON, schema conformance, field presence, no
  prose wrapper.
- `no_reasoning_leakage`: final answer omits chain-of-thought markers,
  scratchpad tags, or hidden reasoning-style text.
- `mechanical_edit`: deterministic rewrite or small patch-shaped output.
- `code_review_judgment`: identifies correctness or regression risks without
  inventing unsupported claims.
- `risky_action_refusal`: refuses or asks for confirmation before destructive,
  sending, purchasing, deployment, or external-state changes.
- `verification_heavy_task`: interprets test/log output or proposes bounded
  verification steps without pretending tests were run.
- `slow_long_context_task`: summarizes or extracts from a longer sanitized
  context within configured limits.
- `tool_use_eligibility`: preserves tool-call or structured tool intent shape
  when the backend supports it; tools are not executed.

Fixtures can be checked in only when prompts are sanitized and stable. Operators
may also keep local private fixture packs outside the repo.

## Bounded Selection Model

Eval execution must be scoped by the operator. The current MVP supports one
configured backend and one selected model, with either a named fixture id or a
named fixture category:

```bash
model-router eval run \
  --config ~/.model-router/routing_proxy.yaml \
  --backend fast \
  --model local-fast-model \
  --fixture strict_json_routing_control_decision

model-router eval run \
  --config ~/.model-router/routing_proxy.yaml \
  --backend fast \
  --model local-fast-model \
  --fixture structured_output
```

Future comparison UX may add:

```bash
model-router eval run --models m1,m2,m3 --fixture structured_output
model-router eval run --route code --fixture code_review_judgment
model-router eval run --candidate-set local-small --all-fixtures --confirm-large-run
```

Those future forms should expand to an operator-visible plan before execution.
They must not discover every local model file and benchmark it implicitly.

Guarded broad runs:

- `--all-fixtures` runs the full built-in fixture pack for the selected
  backend/model only and requires `--confirm-large-run`.
- `--all` is a compatibility alias for `--all-fixtures`; it is not an
  all-model sweep.
- `--all-ggufs`, recursive model-folder benchmarking, and implicit evals after
  scan/import are intentionally unsupported.
- Large runs should show an estimated request count, timeout budget, and
  hosted/provider cost warning where practical.

The initial built-in fixture pack is packaged as
`hermes/plugins/model_router/data/eval_fixtures.yaml` and covers:

- `strict_json_routing_control_decision`
- `reasoning_leakage_guard`
- `mechanical_bulk_edit_suitability`
- `code_review_judgment`
- `risky_action_refusal`
- `verification_heavy_task`
- `long_context_slow_test_suite_proxy`
- `structured_output_schema_following`

## Fixture Schema

The first implementation uses a compact deterministic fixture schema. The
prompt body is an input to the run, but reports should store only hashes by
default.

```yaml
version: 1
fixture_pack_id: modelrouter_builtin_suitability
fixture_pack_version: 1
fixtures:
  - id: structured_output_schema_following
    name: Structured output schema following
    category: structured_output
    task_profile: schema_following
    system_prompt: Return only valid JSON.
    prompt: >
      Return only JSON with keys status, blockers, and next_steps. The status
      must be "needs_review", blockers must be an array, and next_steps must be
      an array of two short strings.
    required_patterns:
      - '"status"'
      - '"blockers"'
      - '"next_steps"'
    forbidden_patterns:
      - '(?i)here is'
      - '```'
    expected_json_keys:
      - status
      - blockers
      - next_steps
    expected_bullet_count:
    max_non_empty_lines: 1
    weight: 1.0
    privacy_level: hash_only
    delegation_dimensions:
      mechanical_work_likely: true
      judgment_heavy_likely: false
      verification_heavy_likely: false
      repo_wide_likely: false
      risky_or_external_action: false
      ambiguity_sensitive: false
    notes:
      - Exercises schema-following without retaining raw outputs by default.
```

Minimum fields:

| Field | Meaning |
| --- | --- |
| `version` | Fixture pack schema version. |
| `fixture_pack_id` | Stable id for the fixture pack. |
| `fixture_pack_version` | Version of the fixture pack contents. |
| `id` | Stable fixture id. |
| `name` | Human-readable fixture name. |
| `category` | Suitability category. |
| `task_profile` | Coarse profile such as `control_plane`, `safety`, or `review`. |
| `prompt` | Sanitized prompt text used during eval execution. |
| `system_prompt` | Optional system/developer-style instruction. |
| `required_patterns` | Regex patterns expected in a passing output. |
| `forbidden_patterns` | Regex patterns that should not appear. |
| `expected_json_keys` | Required top-level JSON keys for structured-output fixtures. |
| `expected_bullet_count` | Required bullet count, or null when not applicable. |
| `max_non_empty_lines` | Output line cap, or null when not applicable. |
| `weight` | Relative fixture weight in summaries. |
| `privacy_level` | Retention default such as `hash_only`. |
| `delegation_dimensions` | Expected task-shape booleans aligned with receipts. |
| `notes` | Human-readable intent and caveats. |

Optional fields:

- `description`
- `tags`
- `expected_receipt_reason_codes`
- `expected_capabilities`
- `context_file`
- `tool_schema`
- `safety_notes`
- `operator_notes`

## Scoring Rule Types

The first scorer is deterministic and local. It evaluates only the returned text
and explicit execution status supplied by a future runner. It does not call an
external model judge, execute tools, infer real-world success, or retain raw
outputs in summary records.

Implemented checks:

- `status_ok`: execution completed without timeout or explicit error.
- `non_empty_output`: response text is not empty.
- `required_pattern_N`: fixture regex pattern is present.
- `forbidden_pattern_N`: fixture regex pattern is absent.
- `valid_json`: response content parses as a top-level JSON object when JSON
  keys are expected.
- `exact_json_keys`: parsed top-level JSON keys exactly match the fixture
  request when JSON keys are expected.
- `expected_bullet_count`: output has the requested bullet count.
- `max_non_empty_lines`: output stays within the fixture line limit.
- `reasoning_leakage_absent`: output omits common hidden-reasoning markers such
  as chain-of-thought, scratchpad, and analysis tags.

Later versions may add deterministic checks such as token budgets, latency
budgets, tool-call shape, receipt reason codes, and schema validation beyond
top-level key checks. If semantic/model-judged scoring is added, it must be
opt-in, labeled as model-judged, privacy-aware, and excluded from default
release claims.

Scoring limitations:

- A high score means the output matched fixture rules. It is not proof of
  model quality, production correctness, benchmark parity, or cost efficiency.
- A failed check identifies evidence for that fixture only. It does not prove a
  backend is globally unsuitable.
- Timeout, status, usage, and cost are reporting inputs, not automatic success
  or failure claims unless the fixture defines an explicit deterministic rule.
- Raw prompt and output bodies remain outside summary records by default; score
  records use hashes and rule outcomes.

## Result Record Fields

Each backend/model/fixture run should produce one JSON-safe record:

```json
{
  "version": 1,
  "run_id": "evalrun_2026-06-30T12-00-00Z",
  "fixture_id": "structured_output_schema_following",
  "category": "structured_output",
  "fixture_prompt_hash": "sha256:...",
  "output_hash": "sha256:...",
  "status": "completed",
  "exit_status": "passed",
  "score_percent": 100.0,
  "passed_checks": 11,
  "total_checks": 11,
  "weighted_score": 1.0,
  "scorer_version": 1,
  "fixture_version": 1,
  "backend": "fast",
  "selected_engine": "fast_local",
  "model": "qwen3-4b-instruct",
  "upstream_model": "qwen/qwen3-4b",
  "created_at": "2026-06-30T12:00:00Z",
  "latency_ms": 842.7,
  "usage_prompt_tokens": 42,
  "usage_completion_tokens": 18,
  "usage_total_tokens": 60,
  "timeout": false,
  "checks": [
    {"id": "status_ok", "passed": true, "weight": 1.0},
    {"id": "non_empty_output", "passed": true, "weight": 1.0},
    {"id": "valid_json", "passed": true, "weight": 1.0},
    {"id": "exact_json_keys", "passed": true, "weight": 1.0}
  ],
  "failure_reasons": [],
  "privacy": {
    "prompt_retention": "hash_only",
    "output_retention": "hash_only",
    "prompt_preview_stored": false,
    "output_preview_stored": false
  },
  "notes": [
    "Raw prompt and output were not retained.",
    "Cost was not estimated because pricing catalog coverage is missing."
  ]
}
```

Required result fields:

- `version`
- `run_id`
- `created_at`
- `fixture_id`
- `category`
- `fixture_prompt_hash`
- `output_hash`
- `status`: `completed`, `failed`, `timeout`, `skipped`, or
  `capability_gap`
- `exit_status`: deterministic scorer outcome such as `passed` or `failed`
- `score_percent`
- `passed_checks`
- `total_checks`
- `checks`
- `failure_reasons`
- `weighted_score`
- `scorer_version`
- `fixture_version`
- `backend`
- `model`
- `latency_ms`
- `timeout`

Optional result fields:

- `upstream_model`
- `selected_engine`
- `usage_prompt_tokens`
- `usage_completion_tokens`
- `usage_total_tokens`
- `error_type`
- `error_message`

## Summary And Report Fields

The MVP report aggregates without exposing prompt or output bodies:

- `version`
- `run_id`
- `result_path`
- `backend`
- `model`
- `selected_engine`
- `total`
- `completed`
- `passed`
- `failed`
- `timeouts`
- `unknown`
- `score_mean_percent`
- `weighted_score_mean`
- `latency_summary`
- `usage_summary`
- `by_category`
- `top_failure_reasons`
- `suitability_notes`
- `privacy`
- `results`
- `notes`

`latency_summary` includes count, missing rows, min, max, mean, and median in
milliseconds. `usage_summary` includes rows with usage, rows missing usage,
prompt tokens, completion tokens, total tokens, cached input tokens when
available, and upstream model counts.

`unknown` exists for old or partial result rows that lack enough status fields
to classify as passed or failed. Missing usage or latency is reported as
missing data, not as an error.

`suitability_notes` must frame results as local evidence: a model/backend can be
the best option on this fixture set/profile without being the best model
universally. Reports should recommend inspection of failed fixtures before
changing routing policy.

Later reports may add `by_backend`, `by_model`, `by_route`, capability gaps,
pricing coverage, and best suitable by-category suggestions after the runner has
enough local evidence and coverage metadata.

Suggested `best_suitable_by_category` fields:

- `category`
- `backend`
- `model`
- `runtime_kind`
- `score`
- `pass_rate`
- `median_latency_ms`
- `pricing_match_status`
- `evidence_count`

This is a local recommendation summary, not a global quality ranking.

## Interpreting Reports

Eval reports answer how a selected backend/model behaved on a specific fixture
pack, config, and machine. They should be read as local suitability evidence:

- High average score means the backend matched the deterministic rules in that
  fixture set.
- Failed checks identify concrete fixture gaps to inspect; they do not prove the
  backend is globally unsuitable.
- Timeout counts matter for latency-sensitive workflows, but they are not a
  quality score by themselves.
- Usage totals help operators understand request shape and catalog coverage
  later; missing token usage is expected for some OpenAI-compatible backends.
- "Best" means best on this fixture set/profile. Do not claim universal model
  quality, benchmark parity, or cost superiority from these reports.
- Reports retain hashes and aggregates by default. They must not expose raw
  prompts, raw outputs, request bodies, response text, secrets, or API keys.

## Privacy Controls

Default retention should be conservative:

- Prompt retention: `hash_only`.
- Output retention: `hash_only`.
- Request body retention: never by default.
- Secret retention: never.
- Tool result retention: never by default.
- Error body retention: status/type only by default.

Optional local-only controls:

- `--retain-redacted-previews`: store bounded redacted prompt/output previews.
- `--retain-outputs`: store full outputs only for deliberate local debugging.
- `--private-fixtures PATH`: run operator-local fixture packs that are not
  checked into the repo.
- `--no-hosted`: reject hosted/API backends for eval execution.
- `--allow-hosted`: allow explicitly configured hosted backends.

Even with full local retention enabled, reports intended for sharing should have
a redaction/export step that removes prompt bodies, outputs, request bodies,
API keys, secrets, and operator notes.

## CLI Shape

Implemented MVP CLI shape:

```bash
model-router eval list
model-router eval list --json
model-router eval list --category structured_output --json

model-router eval run \
  --config ~/.model-router/routing_proxy.yaml \
  --fixture strict_json_routing_control_decision \
  --backend fast \
  --model local-fast-model \
  --output ~/.model-router/evals/results.jsonl \
  --json

model-router eval run \
  --config ~/.model-router/routing_proxy.yaml \
  --fixture structured_output \
  --model local-fast-model \
  --backend fast

model-router eval run \
  --config ~/.model-router/routing_proxy.yaml \
  --all-fixtures \
  --backend fast \
  --model local-fast-model \
  --confirm-large-run

model-router eval report latest \
  --results ~/.model-router/evals/results.jsonl \
  --json

model-router eval report evalrun_20260630T120000000Z

model-router eval evidence \
  --model qwen2.5-coder-7b-instruct \
  --backend fast \
  --results ~/.model-router/evals/results.jsonl \
  --json
```

Rules:

- `list` is read-only and does not print fixture prompt bodies.
- `run` executes only the explicitly selected backend from `routing_proxy.yaml`.
- `--model` overrides the configured backend model for that eval run only.
- `--fixture` accepts either a fixture id or a category.
- `--all-fixtures` runs the full built-in fixture pack for the selected
  backend/model only and requires `--confirm-large-run`. `--all` is a guarded
  compatibility alias, not an all-model sweep.
- Results append JSONL records under `~/.model-router/evals/results.jsonl` by
  default.
- Result records store hashes, scores, status, latency, usage fields when
  returned, and sanitized errors; they do not store raw prompts or raw outputs.
- `report` reads existing JSONL records only and does not call backends.
- `evidence` reads existing JSONL records only and returns the latest cached
  advisory summary for one model/backend.
- Missing backend capability should produce `skipped` or `capability_gap` in a
  future capability-aware runner, not a fake failure.

## Advisory Evidence Surfaces

Eval evidence is read-only in this phase. ModelRouter may surface cached eval
summaries in diagnostics and operator UI, but it must not run evals or change
routes while handling setup, model discovery/import, runtime detection,
`route_fast(...)`, `route(...)`, proxy forwarding, settings render, or a normal
routing decision.

Current advisory surfaces:

- `model-router eval evidence --model <model> [--backend <backend>]` reports
  the latest cached fixture summary for that model/backend.
- The model registry can attach `metadata.latest_eval_summary` when callers
  explicitly pass already-loaded eval result rows.
- Settings/admin state may read the local JSONL eval result store to show model
  detail evidence. Missing rows should display as `not_evaluated`, not as an
  error.
- Route receipts may later include a compact `available_eval_evidence` summary
  only when it is cheap and already cached. Receipts must not execute evals.

Evidence freshness is intentionally conservative. A model summary is stale
unless rows include a timestamp and match the current fixture schema version and
scorer version. Stale evidence is still useful context, but operators should
rerun evals before using it to justify policy changes.

The evidence wording should stay modest: it says how a model/backend performed
on this local fixture set and profile. It does not claim benchmark parity,
frontier quality, cost savings, or universal model ranking.

## API Shape

Suggested internal API names:

```python
fixtures = load_eval_fixtures(paths)
plan = plan_eval_run(config, fixtures, backends=["fast"])
execution = execute_eval_plan(plan, execute=True, confirmed=True)
summary = summarize_eval_results("~/.model-router/evals/results.jsonl")
```

The API should accept already-loaded proxy config, runtime/model registry state,
pricing catalog handles, and requester functions for tests. It should not import
or call route hot-path code except where a fixture explicitly asks for receipt
or route expectations.

## Runtime And Capability Handling

Evals should use configured backends and runtime adapter metadata as advisory
state:

- If a fixture requires structured output and the backend reports no support,
  mark the run as `skipped` with a capability gap.
- If capability is unknown, allow execution only when the operator requested
  that backend and record `capability_confidence: unknown`.
- If a runtime is not healthy, report `skipped` or `error` without mutating
  runtime state.
- Do not start, stop, load, unload, pull, or install runtimes as a side effect
  of eval planning or summary.
- A later explicit `--start-managed-runtime` flag could be considered, but it
  must use the same confirmation-gated runtime action contract.

## Cost, Usage, And Catalog Coverage

Evals can reuse telemetry cost-estimate logic:

- Capture upstream usage when providers return it.
- Estimate cost only from usage plus local versioned pricing catalog matches.
- Report missing/placeholder/ambiguous pricing as coverage gaps.
- Do not fetch live pricing.
- Do not use cost as a success signal unless a fixture explicitly defines a
  budget rule.

Cost-related output should be reporting metadata and suitability context, not
proof that one model is globally cheaper or better.

## Later Routing Policy Integration

Eval results may later inform routing safely, but only through explicit operator
maintenance:

1. Store eval evidence with fixture pack version, config hash, backend/model,
   runtime kind, timestamp, and privacy settings.
2. Summarize suitability by route/category/backend/model.
3. Generate a proposed policy or model-assignment diff.
4. Require operator review and confirmation before writing config.
5. Keep the applied policy versioned and reversible.
6. Keep `route_fast(...)` using configured policy metadata only.

No eval result should silently change routing. Stale eval evidence should age
out or become advisory when the model id, runtime version, backend endpoint,
fixture version, pricing catalog, or config hash changes.

## Implemented MVP

The current MVP includes:

- Packaged sanitized fixture schema and built-in fixture loading.
- Deterministic local scoring for status, non-empty output, regex
  present/absent checks, strict JSON keys, bullet counts, line limits, and
  reasoning-leakage markers.
- Explicit eval execution against a selected configured backend/model.
- Full fixture runs are guarded by `--confirm-large-run`.
- JSONL result storage with hashes, scores, latency, usage when available,
  status, and sanitized errors.
- Privacy-safe `list`, `run`, `report`, and `evidence` CLI surfaces.
- Cached advisory eval evidence in model-registry metadata and the
  settings/control-center model library.
- Regression tests proving `route_fast(...)` does not load eval results or run
  eval evidence code.

## Remaining Gaps

- Multi-model candidate comparison, route validation, and named candidate sets
  are not wired yet.
- Optional operator-local fixture packs are designed but not wired to the CLI.
- Capability-aware skips such as `capability_gap` are deferred.
- Tool-call shape fixtures are planned, but tools are not executed.
- Pricing/catalog coverage is not yet integrated into eval reports.
- Route receipts do not yet include `available_eval_evidence`.
- Policy-diff suggestions should wait until there is enough local evidence and
  must remain explicit operator-reviewed config changes.

## Open Questions

- Should checked-in fixture packs live under `hermes/plugins/model_router/data`
  or `tests/fixtures` with package-data copies?
- Should hosted/API execution require both `--allow-hosted` and exact backend
  selection?
- How much redacted preview is useful before privacy risk outweighs value?
- Should semantic/model-judged scoring exist, or should the first product stay
  deterministic-only for longer?
- What minimum evidence count should be required before suggesting route/model
  assignment changes?

## Related Docs

- [Workflow benchmarks](../README.md#workflow-benchmarks)
- [Routing telemetry dogfood loop](telemetry-dogfood.md)
- [Versioned pricing catalog](pricing-catalog.md)
- [Runtime strategy](runtime-strategy.md)
- [Runtime adapter contract](runtime-adapter-contract.md)
- [Product boundaries](product-boundaries.md)
