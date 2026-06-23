# ModelRouter Roadmap

This roadmap is meant to be worked one milestone at a time. Each milestone has
a ready-to-paste prompt that asks Codex to implement or coordinate the work,
then end with a checkpoint: either fix issues in a small incremental release or
move to the next milestone.

## Milestone 1: v0.5 Release Polish

Goal: make the current local proxy beta cleanly releasable.

Tasks:

- Update stale docs before release:
  - `README.md` still says "after v0.5 is published."
  - `README.md` still points GitHub install examples at `v0.4.1`.
  - `docs/model-router.md` still has "Future Gateway Mode," but the proxy now
    exists.
- Tag `v0.5.0`.
- Create GitHub release notes with benchmark and test output.
- Confirm PyPI trusted publishing is configured.
- Publish package.

Done when:

- `pip install "hermes-router[proxy]"` works from PyPI.

Prompt:

```text
Please handle Milestone 1: v0.5 Release Polish.

Goal: make the current local proxy beta cleanly releasable.

Tasks:
- Update stale docs before release:
  - README still says "after v0.5 is published"; rewrite it for the actual v0.5 release.
  - README still points GitHub install examples at v0.4.1; update them to v0.5.0.
  - docs/model-router.md still has "Future Gateway Mode," but the proxy now exists; rewrite that section to describe the current optional OpenAI-compatible proxy and any truly future adapter work separately.
- Run release verification:
  - python -m ruff check .
  - python -m pytest
  - python scripts/check_route_fast_latency.py --json
  - python -m build
  - python -m twine check dist/*
- Record the benchmark and test output for release notes.
- Tag v0.5.0 and push the tag if verification passes.
- Create GitHub release notes with benchmark/test output.
- Confirm whether PyPI trusted publishing is configured. If it is configured, publish the package through the release workflow. If it is not configured, stop and tell me the exact missing setup.
- After publishing, verify that pip install "hermes-router[proxy]" works from PyPI in a fresh environment if possible.

At the end, tell me whether we should do a small v0.5.1 fix release for any release/publishing issues, or move on to Milestone 2: v0.5.1 Dogfood Stability Pass.
```

## Milestone 2: v0.5.1 Dogfood Stability Pass

Goal: make sure the proxy works smoothly in real agent usage.

Tasks:

- Run a real LM Studio, Ollama, or local OpenAI-compatible endpoint smoke test.
- Test `/v1/chat/completions`, `/v1/models`, `/health`, streaming, fallback,
  and `human_confirm`.
- Add any failed real prompts to the routing corpus.
- Improve error messages for common first-run failures.
- Verify `model-router doctor` catches bad model names, closed ports, missing
  env vars, and auth mistakes.

Done when:

- A fresh user can install, init, run proxy, point an agent at
  `http://127.0.0.1:8082/v1`, and diagnose failures without reading code.

Prompt:

```text
Please handle Milestone 2: v0.5.1 Dogfood Stability Pass.

Goal: make sure the proxy works smoothly in real agent usage.

Tasks:
- Run a real local endpoint smoke test using whichever backend is available on this machine, preferably LM Studio or Ollama.
- Test /v1/chat/completions, /v1/models, /health, streaming, fallback, and human_confirm behavior.
- Add any failed real prompts to the routing corpus as regression fixtures.
- Improve error messages for common first-run failures.
- Verify model-router doctor catches bad model names, closed ports, missing env vars, and auth mistakes.
- Run ruff, pytest, and route_fast latency checks after changes.
- Commit and push a v0.5.1-ready fix set if changes are needed.

At the end, tell me whether we should tag/publish v0.5.1 for stability fixes, or move on to Milestone 3: Routing Accuracy Calibration.
```

## Milestone 3: Routing Accuracy Calibration

Goal: make routing smarter from real traffic without slowing `route_fast`.

Tasks:

- Dogfood with observability enabled.
- Label bad routes with `model-router feedback`.
- Replay logs before changing scoring.
- Add golden tests for common wrong-route cases.
- Tune deterministic marker groups, precedence, ambiguity rules, and thresholds.
- Keep the benchmark guard unchanged.

Done when:

- Real-world misroutes decrease and `route_fast` remains within the documented
  SLOs.

Prompt:

```text
Please handle Milestone 3: Routing Accuracy Calibration.

Goal: make routing smarter from real traffic without slowing route_fast.

Tasks:
- Review routing logs, feedback labels, replay results, golden tests, and real-world failed prompts.
- Identify the highest-impact wrong-route patterns.
- Tune deterministic marker groups, precedence, ambiguity rules, and thresholds.
- Add or update golden/adversarial tests for every fixed pattern.
- Run replay before and after the scoring changes and report route changes, mismatch counts, and latency deltas.
- Run ruff, pytest, and route_fast latency checks.
- Do not add LLM classification or semantic routing unless the replay data proves deterministic rules cannot hit the target.

At the end, tell me whether we should ship these calibration fixes as an incremental v0.5.x release, or move on to Milestone 4: Proxy Hardening.
```

## Milestone 4: Proxy Hardening

Goal: make the proxy robust enough for daily background use.

Status: shipped as the v0.5.3 proxy hardening release. The deferred
socket-level disconnect risk is covered by a live uvicorn/raw-socket streaming
test that disconnects before the full response is consumed and verifies upstream
stream cleanup plus privacy-safe logging. Move next to Milestone 5.

Tasks:

- Improve graceful shutdown and resource cleanup tests.
- Add stronger health detail for backend reachability without exposing secrets.
- Validate streaming edge cases.
- Improve log rotation behavior under long-running usage.
- Add or refine macOS/Linux background-run smoke docs or scripts.

Done when:

- Users can run the proxy all day without manual babysitting.

Prompt:

```text
Please handle Milestone 4: Proxy Hardening.

Goal: make the proxy robust enough for daily background use.

Tasks:
- Review the proxy lifecycle, HTTP client cleanup, streaming behavior, log writer behavior, and health checks.
- Improve graceful shutdown/resource cleanup tests.
- Add stronger /health detail for backend reachability while ensuring no secrets, raw prompts, request bodies, or API keys are exposed.
- Validate streaming edge cases with mocked upstreams.
- Stress or unit test log rotation for long-running usage.
- Refine macOS launchd and Linux systemd docs if needed.
- Run ruff, pytest, and route_fast latency checks.

At the end, tell me whether we should ship these hardening fixes as an incremental v0.5.x release, or move on to Milestone 5: Docs And Product UX Pass.
```

## Milestone 5: Docs And Product UX Pass

Goal: make the project instantly understandable.

Status: completed in the post-v0.5.3 docs pass. The README and supporting docs
now keep the proxy path first, include LM Studio and Ollama setup examples,
show sample `init`, `doctor`, proxy startup, and generic agent configuration
transcripts, document the wrong-route to feedback/replay/regression-test loop,
and clarify that `hermes/plugins/...` is only a legacy Python namespace.

Tasks:

- Keep README proxy-first.
- Add known-good local setup examples for LM Studio and Ollama.
- Add sample terminal transcripts for `init`, `doctor`, and agent config.
- Add a "wrong route to fixed regression test" walkthrough.
- Clarify that `hermes/plugins/...` is only a legacy namespace, not host-app
  plugin integration.

Done when:

- A new user can get from zero to routed requests in under 10 minutes.

Prompt:

```text
Please handle Milestone 5: Docs And Product UX Pass.

Goal: make the project instantly understandable.

Tasks:
- Review README, docs/model-router.md, docs/production-readiness.md, docs/adapter-contract.md, and docs/background-services.md for user friction.
- Keep the README proxy-first.
- Add known-good local setup examples for LM Studio and Ollama.
- Add sample terminal transcripts for model-router init, model-router doctor, model-router-proxy, and generic agent configuration.
- Add a walkthrough showing how a wrong route becomes a feedback label, replay case, and regression test.
- Clarify anywhere needed that hermes/plugins/... is only a legacy Python namespace, not host-app plugin integration.
- Run docs-adjacent checks and the normal test/latency suite if code changes are made.

At the end, tell me whether we should ship these docs/product fixes as an incremental v0.5.x release, or move on to Milestone 6: v0.6 Compatibility Expansion.
```

## Milestone 6: v0.6 Compatibility Expansion

Goal: support more real clients while keeping the OpenAI-compatible proxy as the
core.

Status: shipped as v0.6.0. The proxy now routes and forwards `/v1/responses`
requests alongside `/v1/chat/completions`, with tests for common Responses API
request fields, streaming SSE passthrough, tool-call preservation/stripping,
and `human_confirm` blocking.

Candidate tasks:

- Add `/v1/responses` passthrough/routing if real agents need it. Done.
- Add better tool-call compatibility tests.
- Add provider-specific presets for more OpenAI-compatible gateways.
- Add config migration/version checks.

Done when:

- More agents can use the proxy without custom glue.

Prompt:

```text
Please handle Milestone 6: v0.6 Compatibility Expansion.

Goal: support more real clients while keeping the OpenAI-compatible proxy as the core.

Tasks:
- Review real client compatibility gaps from dogfood notes, issues, and routing logs.
- Decide whether /v1/responses support is necessary now; implement only if real clients need it.
- Add or improve tool-call compatibility tests.
- Add provider-specific presets for additional OpenAI-compatible gateways only when they are concrete and testable.
- Add config migration/version checks if config drift is becoming a real support risk.
- Keep optional compatibility work outside route_fast.
- Run ruff, pytest, proxy tests, and route_fast latency checks.

At the end, tell me whether we should tag this work as v0.6.0, cut a smaller v0.5.x compatibility release, or move on to Milestone 7: Optional Advanced Routing.
```

## Milestone 7: Optional Advanced Routing

Goal: improve hard cases only if data proves deterministic rules are not enough.

Status: deferred after audit. Checked-in replay fixtures, golden tests, parity
tests, and the fixture audit had 0 expected mismatches, no local replay logs
were present, and no matching GitHub wrong-route issues were found. See
`docs/advanced-routing.md` for the decision record and criteria for revisiting
an optional classifier.

Candidate tasks:

- Add optional second-pass classification only for uncertain prompts. Deferred
  until labeled replay data proves need.
- Keep it off by default.
- Load it lazily.
- Prove latency cost is isolated outside the normal `route_fast` path.
- Compare against replay logs before accepting.

Done when:

- Current data is audited, the implement/defer/drop decision is documented, and
  any future classifier has replay and latency gates before acceptance.

Prompt:

```text
Please handle Milestone 7: Optional Advanced Routing.

Goal: improve hard cases only if data proves deterministic rules are not enough.

Tasks:
- Review replay logs, feedback labels, golden tests, and remaining wrong-route cases.
- Decide whether deterministic scoring is insufficient for the remaining hard cases.
- If and only if the data supports it, design an optional second-pass classifier for uncertain prompts.
- Keep it disabled by default, lazily loaded, and outside the normal route_fast path.
- Add benchmarks proving default route_fast latency is unchanged.
- Add replay comparisons proving the optional classifier improves labeled accuracy enough to justify its cost.
- Run ruff, pytest, replay checks, and route_fast latency checks.

At the end, tell me whether this should become a v0.6.x experimental feature, wait for more data, or be dropped.
```

## Milestone 8: Routing Telemetry Dogfood Loop

Goal: make routing telemetry dogfooding easier, safer, and more useful before
considering classifier-based routing.

Status: implemented. The CLI now exposes `model-router telemetry summary` for
event/replay coverage, unlabeled replayable request ids, skipped private events,
route changes, and `expected -> actual` mismatch groups. It also exposes
`model-router telemetry feedback` for label inspection without printing notes
by default. See `docs/telemetry-dogfood.md`.

Done when:

- Dogfood users can inspect routing events, feedback labels, replay coverage,
  mismatch groups, and private/no-prompt skips without reading raw JSONL.
- The workflow keeps prompt/privacy defaults safe.
- Revisit criteria for optional advanced routing are based on roughly 20-30
  labeled wrong routes with repeated patterns that deterministic scoring cannot
  fix cleanly.

## Milestone 9: Installer And First-Run Polish

Goal: make install, init, doctor, and first-run proxy setup easier and more
self-guiding so ModelRouter can be dogfooded for real routing data.

Status: implemented. `model-router init --auto` now chooses a local proxy preset
from Ollama/LM Studio signals, reports start and model-pull guidance, and keeps
telemetry enabled. Readable `doctor` output now includes agent endpoint,
telemetry log path, and next-step remediation for unreachable local upstreams
or missing configured models.

Done when:

- Fresh users can run `model-router init --auto --yes`, see concrete next
  steps, run `doctor`, start the proxy, and begin collecting telemetry without
  reading YAML first.
- Installer and doctor tests cover local signals with mocks and no real network
  dependency.

## Milestone 10: Managed Local Runtime For llama.cpp And MLX-LM

Goal: keep ModelRouter a proxy router while making local model-server ownership
first-class when users explicitly opt in.

Status: implemented. Proxy backends can now declare a `runtime` block with an
argv-only command, runtime kind, readiness URL, idle timeout, shutdown timeout,
and log path. `model-router-proxy` starts managed `llama-server`, `mlx_lm.server`,
or generic local processes on the first route that needs them, keeps them warm,
stops them after the configured idle timeout, and stops all managed children on
proxy shutdown. The `mlx-lm` preset generates one managed `mlx_lm.server` process
per route with placeholder model ids that users must replace. MLX-LM support is
chat/models-first; `/v1/responses` requires an upstream that supports Responses
API and translation remains deferred.

Done when:

- Runtime config validation rejects shell-string commands, invalid kinds,
  invalid readiness URLs, invalid timeouts, and invalid log paths.
- Runtime startup failures return safe `runtime_start_failed` proxy responses
  with route-identification headers.
- Streaming requests keep managed runtimes active until stream cleanup.
- `doctor` reports managed runtime status, missing commands, readiness failures,
  port conflicts, placeholders, and the MLX-LM `/v1/responses` limitation.
- The implementation stays outside `route_fast(...)` and the latency guard
  remains green.

## Milestone 11: Hardware-Aware Runtime Recommendations And Benchmarks

Goal: make first-run local runtime setup recommend several good choices instead
of forcing one model or runtime.

Status: implemented. Setup recommendations now keep local model discovery and
download recommendations separate, score both local and downloadable options,
and treat RAM as the fit/load gate rather than the whole ranking. The score
breakdown includes fit, runtime match, expected speed, route quality, setup
friction, and optional benchmark evidence. Hardware signals now include CPU
architecture, CPU core count, Apple Silicon/Metal, CUDA/ROCm hints, model
format, and quantization. `model-router setup benchmark` can plan or run a
privacy-safe synthetic benchmark against configured backends, storing metrics in
`~/.model-router/benchmarks.json` without prompt bodies, request bodies, API
keys, or secrets. The settings UI shows score labels and benchmark status.
Benchmark data improves future recommendations but never silently mutates config
or routing policy.

Next:

- Dogfood benchmark-backed recommendations on a few real local setups.
- Add richer benchmark modes later if needed, such as streaming first-token
  timing or route-specific quality smoke checks.
- Keep downloads and benchmark-driven config changes user-approved.

## Milestone 12: Local Admin Settings UI

Goal: make ModelRouter easier to configure and dogfood visually while keeping
it a proxy/router, not a chat app or agent.

Status: implemented. `model-router settings` starts a localhost-only FastAPI
admin UI, defaulting to `127.0.0.1:8099`, with server-rendered pages and
minimal JavaScript. The UI can scan local models, show recommended downloads,
edit safe proxy/observability/backend/runtime fields, run doctor, start/stop/
restart `model-router-proxy` as a settings-owned child process, inspect
telemetry counts/recent request ids, and write feedback labels using the same
JSONL format as the CLI.

Done when:

- The UI has no prompt box, no chat transcript surface, and no agent behavior.
- Literal API keys and raw prompts are not displayed in settings state or
  telemetry panels.
- Downloads require explicit confirmation.
- Config writes happen only through Save and validate before replacing
  `routing_proxy.yaml`.
- Proxy process changes happen only through explicit Start/Stop/Restart actions.
- Managed model runtimes remain owned by `model-router-proxy`; the settings UI
  controls them indirectly through config and proxy lifecycle.
- The implementation stays outside `route_fast(...)` and the latency guard
  remains green.

Next:

- Dogfood the settings UI during real local model setup.
- If editing runtime commands as text remains too clunky, add more structured
  runtime command builders for MLX-LM and llama.cpp.
- If telemetry labeling still feels manual, add the deferred interactive
  `model-router telemetry review` queue.
- Continue Milestone 11's benchmark/scoring work before auto-tuning
  recommendations.

## Milestone 13: Open Switchboard Robustness

Goal: absorb the best product lessons from hosted model orchestration while
staying in ModelRouter's lane: open, auditable, local-first routing and proxying.

Status: Tracks 1-7 implemented: routing profiles, provider/backend policy
controls, productized receipts, explicit verification boundary, workflow
benchmarks, catalog update workflow, and product language/onboarding polish.

Tracks:

- Routing profiles that expose `fast`, `balanced`, `quality`, `private`, and
  `safe` modes without making users start from engine names. Done for CLI,
  proxy defaults, settings defaults, receipts, and private local-only
  constraints.
- Provider pools and policy controls for allowlists, denylists, local-only
  mode, hosted-allowed mode, max cost tier, max latency tier, per-route pools,
  proxy backend allowlists/denylists, and receipt-visible rejections. Done for
  YAML config, CLI hints, proxy forwarding, settings visibility, health output,
  fallback constraints, and tests.
- Productized receipts with human-readable summaries, reason codes, rejected
  providers, fallback explanations, and wrong-route next steps. Done for JSON
  receipts, CLI `decide --explain`, proxy telemetry fields, settings receipt
  copy, and tests.
- Explicit optional verification outside `route_fast(...)`, disabled by
  default and observable through telemetry. Done for versioned proxy verifier
  config, receipt-only/sampled/always-for-risky-output modes, log-only or
  fail-closed behavior, streaming skip metadata, and tests.
- Workflow benchmarks that measure routing correctness, profile behavior,
  safety gates, and route changes alongside the existing latency guard. Done
  for the offline `model-router workflow-benchmark` command, sanitized
  fixtures, readable/JSON reports, prompt-hash-only output, and tests.
- Catalog update commands that preview, diff, and apply packaged router catalog
  updates only after confirmation. Done for `model-router catalog
  status|diff|apply`, no-network defaults, backups, migration logs, settings
  status visibility, and tests.
- Product language and onboarding that position ModelRouter as the open
  switchboard for AI model routing, not a hidden multi-agent system. Done for
  the README promise, first-run profile/provider/receipt loop, and production
  boundary language.

Done when:

- Users can pick a plain-language profile, constrain providers, route through
  the proxy, inspect a useful receipt, and label wrong routes without reading
  YAML first.
- CI covers profile behavior, provider-policy fallback, receipt summaries,
  workflow benchmarks, catalog preservation, and optional verification defaults.
- The implementation keeps default routing deterministic, preserves
  `human_confirm` safety, adds no default network calls, and keeps
  `route_fast(...)` within the documented latency guard.

Prompt:

```text
Please implement Milestone 13 from docs/open-switchboard-plan.md.

Start with the next unfinished track, keep the router in its open-switchboard
lane, and avoid hidden multi-agent orchestration. Preserve deterministic default
routing, provider transparency, privacy-safe telemetry, and human-confirm safety.

For each track:
- Read the related code and docs first.
- Add focused tests and docs.
- Keep optional runtime behavior outside route_fast.
- Run ruff, pytest, and scripts/check_route_fast_latency.py --json.
- End with a short recommendation for the next track.
```

## Next Planned Work After Milestone 13

Goal: turn the open switchboard foundation into a tighter daily-use loop.

Recommended tasks:

- Real proxy dogfood: run LM Studio, Ollama, llama.cpp, and MLX-LM setups
  through `/v1/chat/completions`, `/v1/responses`, streaming, fallback,
  `human_confirm`, backend policy rejection, and optional verifier modes.
- Settings UI follow-through: add explicit catalog diff/apply controls with
  confirmation, show workflow benchmark status, and keep all write actions
  local and user-confirmed.
- Telemetry review queue: build the deferred `model-router telemetry review`
  flow for labeling wrong routes without exposing notes or prompt bodies by
  default.
- Receipt calibration: dogfood the wrong-route next actions and promote stable
  clusters into workflow benchmark fixtures or replay tests.
- Release hardening: prepare a small release with changelog, benchmark output,
  route-fast latency output, catalog workflow notes, and upgrade guidance.

Do not add hidden orchestration, default hosted-provider calls, automatic
downloads, runtime auto-start, verifier calls, or prompt logging while doing
this work. Keep `route_fast(...)` deterministic and keep safety gates explicit.
