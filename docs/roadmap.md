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

Status: initial compatibility work landed after v0.5.4. The proxy now routes
and forwards `/v1/responses` requests alongside `/v1/chat/completions`, with
tests for common Responses API request fields, streaming SSE passthrough,
tool-call preservation/stripping, and `human_confirm` blocking.

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

Candidate tasks:

- Add optional second-pass classification only for uncertain prompts.
- Keep it off by default.
- Load it lazily.
- Prove latency cost is isolated outside the normal `route_fast` path.
- Compare against replay logs before accepting.

Done when:

- Accuracy improves measurably on labeled logs with negligible default-path
  impact.

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
