# Maturity model and escape hatches

This plan intentionally separates implementation from maturity. A feature may be present in code but still experimental until it has tests, docs, dogfood coverage, safe fallback behavior, and UI/TUI clarity.

## Maturity levels

| Level | Meaning | UI/TUI label | Release rule |
| --- | --- | --- | --- |
| Planned | Documented but not implemented. | Hidden or disabled with “planned”. | Must not appear as a working control. |
| Experimental | Implemented behind a flag or limited adapter. | Badge: Experimental. | May fail safely; must not break stable flows. |
| Beta | Real users can dogfood it; known gaps are documented. | Badge: Beta. | Requires tests and rollback path. |
| Stable | Default-safe and covered by dogfood/replay. | No warning badge. | Can be part of normal onboarding. |

## Feature maturity targets

| Feature | Initial maturity | Stable criteria |
| --- | --- | --- |
| Basic router manual mode | Beta | Tests for chat/Responses, telemetry, headers, UI/TUI mode controls. |
| Basic router model_map/passthrough | Experimental | Alias validation, `/v1/models`, unknown model policy, dogfood coverage. |
| Installer | Beta | Deterministic JSON output, safe existing-config behavior, docs for uv/pipx/editable installs. |
| Model Library installed tab | Beta | Local scan + runtime adapter state + empty states. |
| Discover external search | Experimental | Timeout handling, rate-limit handling, curated fallback, stable scoring. |
| Downloads | Experimental | Plan/confirm/run/status/retry, failure recovery, disk checks. |
| Runtime adapters | Experimental per adapter | Health + capabilities + loaded models + disabled reasons + tests. |
| TUI read-only | Beta | Uses shared admin state, works without browser, render tests. |
| TUI mutating actions | Experimental | Confirmation modals, action refresh, tests. |
| API `/v1/embeddings` | Beta | Fake upstream tests, capability checks, alias behavior. |
| API `/v1/messages` | Experimental | Compatibility tests with Claude-style clients and clear unsupported behavior. |

## Escape hatches

### Basic router mode

If the complete mode set is too large, ship only:

```text
routing_mode = decision | manual
```

Reject `model_map` and `passthrough` with clear config-validation errors until implemented. Do not silently treat them as decision mode.

### Model discovery

If external Hugging Face search is unreliable, keep Discover backed by the packaged curated catalog and local scan results. Show external search as experimental and disabled by default.

### Downloads

If progress tracking is not ready, support plan + execute + final status first. Render progress as unknown rather than fake percentages.

### Runtime adapters

If a runtime cannot support load/unload/list-loaded through a stable API, expose health and capabilities only. Disable unsupported controls with a reason such as “External runtime does not expose load/unload through this adapter yet.”

### TUI

If read/write TUI is too large, ship read-only TUI first. It must still use real shared admin state and must not include fake models, fake routes, or fake logs.

### Installer

If interactive install is too large, ship deterministic JSON/dry-run and `--yes` flows first. Add interactive menus later.

### API parity

If the endpoint matrix is too large, ship `/v1/embeddings` first because it unlocks RAG/model-library routing. Add legacy completions and messages after capabilities are reliable.

## Rollback strategy

- Keep `routing_mode: decision` as the default.
- Preserve old config fields and migrate new fields with defaults.
- If basic router mode fails validation, fail closed with a clear config error rather than falling back silently.
- If UI/TUI action fails, return an action result with `ok=false` and do not mutate config.
- If a runtime adapter fails, mark that adapter unavailable and keep the proxy usable.
- If model discovery fails, keep installed/local models visible.

## Release gates

A feature cannot be marked stable until all are true:

1. Unit tests cover success and failure paths.
2. Dogfood command or fixture covers the user flow.
3. Web UI state is wired.
4. TUI state is wired or intentionally disabled with a reason.
5. Docs explain configuration and rollback.
6. Telemetry does not expose raw prompts by default.
7. No silent downloads, provider enablement, process start, or config mutation.
