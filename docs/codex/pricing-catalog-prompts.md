# Pricing catalog prompt pack

Use these prompts to move from usage telemetry and manual outcome labels to local,
versioned cost reporting. The sequence is intentionally split so each step can be
reviewed and committed independently, while the starter prompt can run the whole
path end to end.

## Starter prompt - execute the sequence end to end

```text
You are working in the ModelRouter repo.

Goal: implement the next pricing/cost-reporting track end to end without adding
live pricing fetches to routing or proxy forwarding.

Run the prompts in `docs/codex/pricing-catalog-prompts.md` in order:
P0 cleanup, P1 catalog design, P2 packaged/user-overridable pricing catalog,
P3 telemetry cost estimates, P4 optional pricing maintenance command, and P5
docs/verification.

Rules:
- Preserve route_fast(...) behavior and latency.
- Do not fetch live pricing during route_fast(...), route(...), proxy forwarding,
  or telemetry summary rendering.
- Do not scrape provider pages.
- Use only local packaged metadata plus explicit operator commands.
- Treat cost estimates as reporting metadata, not routing policy.
- Keep old telemetry and feedback JSONL rows readable.
- Avoid claiming cost reductions, benchmark parity, or frontier performance.
- Add tests for every new behavior and run the relevant suite after each step.
- If a step is unsafe, ambiguous, or blocked by missing credentials/network,
  stop and explain the blocker instead of inventing data.

Expected end state:
- Merged remote branches are cleaned up when they are already merged and safe to
  delete.
- A local versioned pricing catalog exists with packaged defaults and an
  operator override path.
- Telemetry/admin/CLI/TUI reporting can estimate cost from captured usage and
  catalog metadata.
- Pricing maintenance commands preview and apply catalog updates explicitly.
- The worktree is clean after intentional commits, unless a blocker is explained.
```

## P0 - cleanup merged remote branches

```text
Clean up merged branches before starting pricing feature work.

Scope:
- Inspect `git status --short --branch`, `git branch -r --merged main`,
  `git branch -r --no-merged main`, and open PR state.
- Delete only remote branches that are already merged into `main` and clearly no
  longer needed, such as `origin/docs/codex-installer-tui-router-mode` if PR #7
  is merged.
- Do not delete protected branches, release branches, or any branch that is not
  merged into `main`.
- Do not rewrite history.

Acceptance:
- Worktree remains clean.
- Deleted branches are listed in the final note.
- Any branch intentionally left alone is explained.
```

## P1 - design the versioned pricing catalog

```text
Design the local versioned pricing catalog before implementation.

Scope:
- Review README.md, docs/telemetry-dogfood.md, docs/production-readiness.md,
  docs/open-switchboard-plan.md, routing_log.py, telemetry.py, proxy.py, cli.py,
  settings_ui.py, and existing catalog/config patterns.
- Add or update docs describing the pricing model:
  - Routing decisions still use configured `cost_tier` metadata.
  - `route_fast(...)`, `route(...)`, proxy forwarding, and default telemetry
    rendering must never fetch live pricing.
  - Exact price estimates come from a local versioned catalog only.
  - Operators may override the packaged catalog locally.
  - Cost estimates are reporting fields, not success claims and not routing
    decisions.
- Define the catalog schema:
  - catalog_version
  - updated_at
  - entries[]
  - provider
  - model
  - input_per_1m
  - output_per_1m
  - cached_input_per_1m
  - currency
  - effective_date
  - source
  - notes
- Define lookup semantics for provider/model/backend_model/upstream_model and
  missing entries.

Acceptance:
- Docs clearly separate cost tiers, captured usage, local price metadata, and
  estimated cost.
- The design explicitly forbids live pricing fetches in routing/proxy hot paths.
- No code path changes `route_fast(...)`.
```

## P2 - implement packaged and user-overridable pricing catalog

```text
Implement the local versioned pricing catalog.

Scope:
- Inspect existing packaged metadata/config loading patterns.
- Add a packaged pricing catalog file in an appropriate project location.
- Add a small pricing catalog loader module with:
  - packaged default loading
  - optional user override loading
  - schema validation
  - deterministic lookup by provider/model
  - JSON-safe serialization
- Include metadata fields:
  - catalog_version
  - updated_at
  - provider
  - model
  - input_per_1m
  - output_per_1m
  - cached_input_per_1m
  - currency
  - effective_date
  - source
  - notes
- Do not make network calls.
- Do not wire estimates into routing decisions.
- Keep values conservative and clearly sourced. If exact current values are not
  known locally, use placeholder test fixtures or a tiny example catalog that is
  clearly marked non-authoritative, then document how operators override it.

Acceptance:
- Loader tests cover valid catalog, invalid catalog, missing override, override
  precedence, missing price entry, and JSON-safe output.
- No routing or proxy hot path imports the pricing loader unless explicitly
  needed for reporting.
- Existing tests still pass.
```

## P3 - estimate costs from telemetry in reporting paths only

```text
Add estimated cost reporting from captured usage and the local pricing catalog.

Scope:
- Inspect telemetry.py, routing_log.py, cli.py, settings_ui.py, tui.py, and
  existing telemetry tests.
- Use captured usage fields only:
  - usage_prompt_tokens
  - usage_completion_tokens
  - usage_total_tokens
  - usage_cached_input_tokens
  - upstream_model
  - backend_model
  - backend
  - selected_engine
- Match usage to catalog entries using provider/backend/model metadata available
  in telemetry. Prefer upstream_model when present, then backend_model, then
  selected_model where available.
- Add JSON-safe reporting fields such as:
  - estimated_input_cost
  - estimated_output_cost
  - estimated_cached_input_cost
  - estimated_total_cost
  - estimated_cost_currency
  - pricing_catalog_version
  - pricing_catalog_source
  - pricing_match_status
- Add aggregate totals by route/backend/model where telemetry already aggregates
  usage.
- Missing catalog entries must produce `pricing_match_status: missing_price`
  or equivalent, not errors.
- Do not infer success, quality, or outcome from cost.
- Do not alter routing decisions.

Acceptance:
- CLI telemetry summary/review can show cost estimates without printing prompts,
  request bodies, secrets, or response text.
- Settings UI and TUI show compact cost aggregates when present.
- Old JSONL rows without usage fields still work.
- Tests cover matched price, missing price, cached input tokens, old logs, and
  privacy-safe output.
- `route_fast(...)` remains unchanged.
```

## P4 - add explicit pricing maintenance commands

```text
Add explicit pricing catalog maintenance commands.

Scope:
- Add a CLI group such as:
  - `model-router pricing status`
  - `model-router pricing diff`
  - `model-router pricing apply`
- These commands operate only on local packaged catalog files and local operator
  override files.
- `status` reports packaged version, override path, active version/source, entry
  counts, and validation issues.
- `diff` previews changes between packaged defaults and local override/active
  catalog.
- `apply` writes or updates the local override only with explicit confirmation
  such as `--yes`.
- Do not fetch network pricing.
- Do not scrape provider pages.
- Do not run during `route_fast(...)`, `route(...)`, proxy forwarding, or default
  telemetry reads.

Acceptance:
- CLI tests cover status, diff, apply without confirmation, apply with
  confirmation, invalid catalog, and no-network behavior.
- Commands print clear local paths and catalog versions.
- Existing routing/proxy tests still pass.
```

## P5 - docs, review, and final verification

```text
Finish the pricing/cost-reporting track with docs and verification.

Scope:
- Update README.md and docs/telemetry-dogfood.md with operator-facing guidance.
- Update docs/production-readiness.md with the no-live-pricing invariant.
- Add a short note in docs/open-switchboard-plan.md if useful.
- Review tests and implementation for these invariants:
  - cost_tier remains routing policy metadata
  - pricing catalog powers reporting estimates only
  - no live pricing fetches in hot paths
  - old JSONL telemetry and feedback rows remain readable
  - prompts/request bodies/secrets/response text are not exposed in summaries
- Run:
  - `git diff --check`
  - focused pricing/telemetry tests
  - full test suite

Acceptance:
- Docs tell a coherent product story: route by policy/tier, measure actual usage,
  estimate reporting costs from local versioned catalog metadata.
- Tests pass or failures are fixed/explained.
- Final summary lists changed files, verification commands, and any intentionally
  deferred pricing-refresh work.
```
