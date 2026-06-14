# Hermes Router Performance Report

Date: 2026-06-14

## Summary

Hermes Router is already fast for embedded use. On this machine, initialized
`route_fast(...)` takes about 2 us for ordinary prompts, while rich
`route(...)` takes about 57 us with alternatives enabled. The main performance
risks are not normal hot-path routing; they are long prompts, CLI process
startup, JSON receipt serialization, and repeated work in the rich scorer and
alternative-ranking path.

The most promising improvements are:

1. Short-circuit long prompts earlier in `route_fast(...)`.
2. Precompile or otherwise cache the remaining inline regex checks in
   `score_prompt(...)`.
3. Cache merged scoring weights on the initialized router/config.
4. Add a no-alternatives or top-N alternatives mode for dispatch/receipt paths.
5. Treat the CLI as a startup-bound interface and avoid using it as a high-QPS
   runtime boundary.

## Benchmark Environment

- Python: 3.11.15
- Executable: `/tmp/hermes-router-venv/bin/python`
- Working tree: `/Users/sambehdjou/Documents/GitHub/hermes-router`
- Branch: `main`
- Benchmark style: warmup, GC disabled during timed loops, 5 repeated samples
  for in-process calls, 8 repeated samples for CLI subprocess calls.
- Prompt set: 9 ordinary prompts plus 1 long-context prompt. The long prompt was
  `"Analyze the rollout risks and migration plan. " * 130`.

## Top-Level Results

### Initialization

| Operation | Best | Mean | Notes |
| --- | ---: | ---: | --- |
| `ModelRouter.from_config(validate_availability=False)` | 7.28 ms | 7.34 ms | YAML load and dataclass construction dominate |
| `ModelRouter.from_config(validate_availability=True)` | 7.29 ms | 7.34 ms | Availability checks are cheap for the default catalog |

Initialization is not appropriate per prompt. The initialized API should be
kept and reused, as the README already recommends.

### In-Process Runtime

| Operation | Best | Mean | Notes |
| --- | ---: | ---: | --- |
| `route_fast`, ordinary mixed prompts, availability off | 1.97 us | 1.98 us | Very fast hot path |
| `route_fast`, ordinary mixed prompts, availability on | 1.98 us | 1.99 us | Cached availability has negligible cost |
| `route_fast`, all prompts including long-context | 26.79 us | 26.84 us | Long prompt dominates the average |
| `score_prompt`, ordinary mixed prompts | 40.47 us | 40.54 us | Regex and object construction dominate |
| `score_prompt`, all prompts including long-context | 394.66 us | 396.73 us | Regex scans scale with prompt length |
| `route(..., include_alternatives=False)` | 46.48 us | 46.82 us | Rich decision without alternative ranking |
| `route(..., include_alternatives=True)` | 56.64 us | 56.84 us | Alternatives add about 10 us |
| `decision_to_receipt(...)` | 1.80 us | 1.81 us | Cheap |
| `receipt_to_json(...)` | 33.35 us | 33.42 us | JSON serialization is comparable to scoring |
| `build_dispatch_plan(...)` | 60.57 us | 60.93 us | Mostly rich route plus receipt wrapping |

### Per-Prompt `route_fast(...)`

| Prompt class | Best | Mean |
| --- | ---: | ---: |
| Confirmation | 0.29 us | 0.29 us |
| Simple | 0.46 us | 0.46 us |
| Coding | 0.72 us | 0.72 us |
| Image generation | 1.04 us | 1.04 us |
| Vision | 1.66 us | 1.66 us |
| Ambiguous | 2.18 us | 2.20 us |
| Research | 2.55 us | 2.57 us |
| Reasoning | 3.67 us | 3.68 us |
| Balanced | 4.13 us | 4.27 us |
| Long context | 249.79 us | 250.07 us |

The ordinary route-fast path is comfortably sub-5 us. Long prompts are the
outlier at about 250 us because the fast path performs several full-string
marker scans before reaching the length-based reasoning fallback.

### Per-Prompt `route(...)` With Alternatives

| Prompt class | Best | Mean |
| --- | ---: | ---: |
| Ambiguous | 34.04 us | 34.22 us |
| Simple | 38.09 us | 38.21 us |
| Confirmation | 47.02 us | 47.25 us |
| Coding | 47.35 us | 47.54 us |
| Image generation | 52.32 us | 52.56 us |
| Research | 58.53 us | 58.90 us |
| Vision | 59.83 us | 59.97 us |
| Balanced | 64.99 us | 65.12 us |
| Reasoning | 80.05 us | 80.50 us |
| Long context | 3.59 ms | 3.61 ms |

Rich route cost scales much more sharply with prompt length because
`score_prompt(...)` runs many regex searches across the normalized prompt.

### CLI Runtime

| Command | Best | Mean |
| --- | ---: | ---: |
| `decide "rewrite this text"` | 65.33 ms | 66.13 ms |
| `decide --json "fix the repo and run tests"` | 64.96 ms | 65.91 ms |
| `dispatch-plan --json "fix the repo and run tests"` | 65.38 ms | 66.17 ms |
| `validate-config --json` | 64.71 ms | 65.93 ms |

The CLI is dominated by Python interpreter/import/startup overhead. The router
itself is microseconds; a CLI command is milliseconds. Do not use the CLI as a
per-request boundary for latency-sensitive systems.

## Profiling Hotspots

### `route_fast(...)`

Profile: 30,000 ordinary mixed prompt routes, 0.189 seconds total.

Top internal-time costs:

- `_fast_has_any(...)`: 0.053 s
- `_fast_has_confirmation_word(...)`: 0.053 s
- `_fast_target_route_index(...)`: 0.036 s
- string `strip(...)`: 0.010 s
- `_fast_is_ambiguous(...)`: 0.005 s

Relevant code:

- `hermes/plugins/model_router/policy.py:599`
- `hermes/plugins/model_router/policy.py:638`
- `hermes/plugins/model_router/policy.py:669`

Interpretation: normal `route_fast(...)` time is mostly repeated marker scans
and token cleanup. This is fine for short prompts, but the same approach is
expensive for long prompts.

### `route(...)`

Profile: 3,000 ordinary mixed prompt routes, 0.336 seconds total.

Top internal-time costs:

- `re.Pattern.search`: 0.096 s
- `score_prompt(...)`: 0.045 s internal, 0.214 s cumulative
- `_rank_engine(...)`: 0.031 s internal, 0.046 s cumulative
- `_rank_alternatives(...)`: 0.022 s internal, 0.091 s cumulative
- `ModelRouter.route(...)`: 0.015 s internal, 0.362 s cumulative

Relevant code:

- `hermes/plugins/model_router/scorer.py:176`
- `hermes/plugins/model_router/policy.py:1014`
- `hermes/plugins/model_router/policy.py:1052`

Interpretation: rich routing spends most of its time in scoring and alternative
ranking. Disabling alternatives saves about 10 us per ordinary prompt.

### `score_prompt(...)`

Profile: 6,000 ordinary mixed prompt scores, 0.387 seconds total.

Top internal-time costs:

- `re.Pattern.search`: 0.185 s
- `score_prompt(...)`: 0.080 s internal, 0.396 s cumulative
- `PromptFeatures` dataclass construction: 0.019 s
- `_merged_weights(...)`: 0.011 s
- Python `re` compile/cache lookup under `_matches(...)`: 0.010 s
- `_complexity_signals(...)`: 0.009 s
- `_matches(...)`: 0.046 s cumulative

Relevant code:

- `hermes/plugins/model_router/scorer.py:188`
- `hermes/plugins/model_router/scorer.py:203`
- `hermes/plugins/model_router/scorer.py:219`
- `hermes/plugins/model_router/scorer.py:227`
- `hermes/plugins/model_router/scorer.py:420`

Interpretation: the recent precompile work helped the stable patterns, but the
remaining inline `_matches(...)` calls still pay function and `re` cache lookup
cost. Merged scoring weights are also recomputed for every prompt.

## Improvement Opportunities

### 1. Short-Circuit Long Prompts Earlier In `route_fast(...)`

Current behavior in `policy.py:599` checks confirmation, simple/coding special
cases, image markers, vision markers, coding markers, research markers, and
only then checks `prompt_length >= 4000` at `policy.py:629`.

For a long-context prompt, `route_fast(...)` measured about 250 us versus
sub-5 us for ordinary prompts.

Recommended change:

- Keep the high-risk confirmation scan first.
- Move the `prompt_length >= 4000` reasoning fallback immediately after the
  safety-critical confirmation and direct coding/simple prefix checks.
- Avoid image/vision/research marker tuple scans for prompts that already exceed
  the long-context threshold.

Expected impact:

- Large improvement for long prompts in `route_fast(...)`.
- Minimal risk if confirmation remains first.

### 2. Precompile The Remaining Inline Scorer Patterns

The scorer intentionally left `image_generation_intent`, `pii_risk`, and
`purchase_action` inline during conflict resolution. Now that the scorer
precision changes have landed, these can be safely promoted to module-level
compiled regex constants.

Candidates:

- `scorer.py:203` image generation pattern
- `scorer.py:219` PII pattern
- `scorer.py:227` purchase pattern
- `scorer.py:416` order word check

Expected impact:

- Small but measurable reduction in `score_prompt(...)`.
- Removes `re` cache lookup overhead visible in the profile.

### 3. Cache Merged Scoring Weights

`score_prompt(...)` calls `_merged_weights(config)` on every prompt at
`scorer.py:182`. The profile shows this at about 0.020 s cumulative for 6,000
scores, roughly 3.3 us per score.

Recommended change:

- Merge weights once in `ModelRouter.__init__`.
- Pass the merged weights into scoring, or add a cached field/method on
  `ScoringConfig`.
- Keep the existing public `score_prompt(prompt, scoring_config=...)` behavior
  for compatibility.

Expected impact:

- Up to several microseconds off rich `route(...)`.
- No impact on `route_fast(...)`.

### 4. Add A Lean Rich Route For Dispatch Paths

`build_dispatch_plan(...)` currently calls `route(...)` with default
`include_alternatives=True` in `dispatch.py:44`. The benchmark shows:

- Rich route with alternatives: 56.84 us mean.
- Rich route without alternatives: 46.82 us mean.
- Dispatch plan: 60.93 us mean.

Recommended change:

- Add an `include_alternatives` parameter to `build_dispatch_plan(...)`, or make
  dispatch plans default to `include_alternatives=False` unless callers ask for a
  full receipt.

Expected impact:

- About 10 us saved per dispatch plan for ordinary prompts.
- Smaller receipts when alternatives are not needed.

### 5. Limit Or Cache Alternative Ranking

`_rank_alternatives(...)` and `_rank_engine(...)` account for a meaningful slice
of rich route cost. This is small with the current catalog, but it scales
linearly with the number of engines.

Recommended change options:

- Add `top_n_alternatives` and avoid materializing every compatible alternative.
- Precompute static engine ranking components for common ranking modes.
- Precompute enabled non-routing candidates and skip repeated filtering of
  `human_confirm` and `intent_router`.

Expected impact:

- Low value for the current small catalog.
- Higher value if the YAML catalog grows to dozens of engines.

### 6. Treat CLI Latency As Startup-Bound

Every measured CLI command took about 66 ms. Since in-process routing is
microseconds, CLI latency is almost entirely interpreter startup, imports, YAML
load, and command setup.

Recommended guidance:

- Keep CLI for humans, scripts, and diagnostics.
- For services, instantiate `ModelRouter` once and call `route_fast(...)` or
  `route(...)` in process.
- If a command boundary is needed, consider a small long-running daemon or
  service wrapper instead of spawning Python per prompt.

Expected impact:

- Orders-of-magnitude lower latency for any high-volume caller.

### 7. Be Careful With JSON Serialization

`receipt_to_json(...)` measured about 33 us, comparable to the scorer itself for
ordinary prompts. This is fine for diagnostics, but it is too expensive to do
unnecessarily in a hot path.

Recommended guidance:

- Return `route_fast(...)` strings for hot loops.
- Return `RoutingDecision` objects when callers do not need serialized JSON.
- Use compact JSON (`indent=None`) for machine-to-machine paths if readable JSON
  is not needed.

## Suggested Priority

1. Move the long-context `route_fast(...)` check earlier after confirmation.
2. Precompile the remaining inline scorer regexes now that the conflict window
   has passed.
3. Cache merged scoring weights.
4. Add `include_alternatives=False` support to dispatch-plan callers.
5. Add a benchmark regression script that covers ordinary prompts and the
   long-context outlier.

## Useful Baselines For Regression Tests

These are not hard pass/fail thresholds because local hardware varies, but they
are useful ratios:

- Ordinary `route_fast(...)` should remain in the low single-digit microseconds.
- Ordinary `route(...)` should remain under roughly 100 us on this machine.
- Long-context `route_fast(...)` should be much closer to ordinary routing after
  the early-length short-circuit.
- CLI commands should not be used as runtime latency baselines.

## Follow-Up Implementation Results

After the first report, the following changes were implemented and remeasured:

- Long-prompt `route_fast(...)` now short-circuits to reasoning before the
  non-safety marker scans.
- The remaining scorer regexes for image generation, PII, purchase intent, and
  `order` detection are precompiled.
- Initialized `ModelRouter` instances cache merged scoring weights for rich
  routing.
- Dispatch plans skip ranked alternatives by default and expose
  `include_alternatives=True` / `--include-alternatives` for full receipts.
- README/docs now call out that the CLI is for humans, diagnostics, and scripts,
  not high-QPS runtime dispatch.

Final focused benchmark:

| Operation | Baseline best | Final best | Result |
| --- | ---: | ---: | --- |
| `route_fast`, ordinary mixed prompts | 1.97 us | 2.04 us | roughly flat; slight branch overhead/noise |
| `route_fast`, long prompt | 252.24 us | 98.67 us | 2.6x faster |
| `score_prompt`, ordinary direct call | 40.86 us | 39.97 us | modestly faster |
| `score_prompt`, cached weights | n/a | 38.71 us | fastest scorer path |
| `route(..., include_alternatives=False)` | 46.99 us | 44.07 us | 6% faster |
| `route(..., include_alternatives=True)` | 57.95 us | 55.27 us | 5% faster |
| `build_dispatch_plan(...)`, default | 61.87 us | 47.77 us | 23% faster |
| `build_dispatch_plan(..., include_alternatives=True)` | n/a | 58.35 us | full receipt remains available |

An attempted regex prefilter for `route_fast(...)` confirmation detection was
tested and reverted. It increased ordinary route-fast latency and made the long
prompt case slower, so the simpler token scanner remains the better tradeoff.

Verification after the changes:

```bash
/tmp/hermes-router-venv/bin/python -m pytest tests/test_model_router*.py
```

Result: 175 passed.
