# Public Preview Readiness

This checklist is the repo-level gate for a public developer-preview push. It
is intentionally narrower than a tagged release: no PyPI publish, no GitHub
Release, and no version tag are implied by passing this checklist.

## Preview Positioning

ModelRouter should be described as a local AI control center and
routing/control plane. It can run common local-model workflows through explicit
operator surfaces, and it can sit above LM Studio, Ollama, llama.cpp,
MLX/MLX-LM, LocalAI, vLLM, hosted OpenAI-compatible providers, and internal
gateways.

Keep these boundaries visible:

- Host agents own planning, tool use, context management, delegation, and
  final review.
- External runtimes own model execution unless a future bundled runtime is
  explicitly selected.
- ModelRouter owns policy, routing, receipts, safety gates, local telemetry,
  pricing catalog reporting, runtime status, model registry state, and
  operator controls.
- ModelRouter should not imply hidden worker orchestration, silent installs,
  silent downloads, recursive model benchmarking, live pricing fetches in hot
  paths, or unsupported benchmark/cost/performance claims.

## Docs And Assets

Before pushing a public-preview readiness commit:

1. Check README positioning, quickstart commands, architecture diagram,
   screenshots, maturity language, and non-goals.
2. Check product boundary docs: `docs/product-boundaries.md`,
   `docs/product-north-star.md`, `docs/runtime-strategy.md`,
   `docs/runtime-adapter-contract.md`, `docs/lm-studio-parity-roadmap.md`,
   `docs/model-router-evals.md`, `docs/pricing-catalog.md`, and
   `docs/telemetry-dogfood.md`.
3. Capture screenshots from a sanitized config directory, not a real user
   config with private paths, API keys, prompts, or local secrets.
4. Store public images under `docs/assets/` or `docs/assets/screenshots/`.
5. Use `docs/assets/social-preview.png` for GitHub's repository social preview
   image. It is derived from the sanitized full control-center screenshot.
   GitHub currently exposes this as a Settings UI upload; the checked-in PNG is
   sized to 1280x640 and kept under 1 MB.
6. Confirm compact mode is presented as a standalone smaller app surface, not
   an overlay on the full control center.

## Hot-Path Boundaries

`route_fast(...)` must remain allocation-light and independent of:

- Receipt generation.
- Telemetry summary/review.
- Pricing catalog loading, refreshing, or live fetching.
- Runtime detection/adapters.
- Model discovery/import.
- Workflow benchmarks.
- Model evals.
- Network requests.
- Filesystem scans.

`route(...)` may build diagnostics and receipts, but it must not run network
calls, live pricing fetches, runtime lifecycle actions, workflow benchmarks, or
model evals. Proxy forwarding must not run runtime discovery, pricing
maintenance, workflow benchmarks, or evals by default.

## Minimum Local Checks

Run these from the repo before a public-preview readiness push:

```bash
git status --short
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_route_fast_latency.py --json
.venv/bin/model-router --help
.venv/bin/model-router pricing status --json
.venv/bin/model-router telemetry summary --json
.venv/bin/model-router settings --host 127.0.0.1 --port 8099 --no-open
```

For the settings command, confirm both `/` and `/compact` respond, capture
screenshots, and shut the server down cleanly.

When practical, also run a non-editable install smoke from a temporary
directory and virtual environment outside the repo. Confirm the installed
package imports from site-packages rather than the development checkout.

## Current Preview Limits

These limits should be documented rather than hidden:

- Runtime lifecycle support depends on what each adapter/runtime exposes.
- LM Studio, Ollama, LocalAI, vLLM, llama.cpp, MLX/MLX-LM, and hosted backends
  may have partial or deferred capability support.
- Evals are explicit, bounded operator actions. They do not run during setup,
  routing, proxy forwarding, runtime detection, model discovery, or model
  import, and they do not automatically change route policy.
- Pricing estimates are reporting-only and depend on local catalog coverage and
  captured usage fields. Routing still uses configured policy and cost tiers.
- Telemetry and review surfaces must not display raw prompts, request bodies,
  secrets, or response text by default.
