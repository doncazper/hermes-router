# Codex implementation pack

This folder turns the installer, TUI, UI, model-library, and basic-router-mode discussion into implementation-ready work for Codex.

## Files

- [control-plane-roadmap.md](control-plane-roadmap.md) — product direction, scope, milestones, and implementation order.
- [../lm-studio-parity-roadmap.md](../lm-studio-parity-roadmap.md) — local-model app parity matrix and beyond-LM-Studio direction.
- [ui-tui-wireframes.md](ui-tui-wireframes.md) — mock screenshots, page/tab purposes, and the backend state/actions each field must use.
- [implementation-briefs.md](implementation-briefs.md) — concise implementation briefs with acceptance criteria for each milestone.
- [pricing-catalog-prompts.md](pricing-catalog-prompts.md) — starter and sequenced prompts for local versioned pricing, cost estimates, and pricing maintenance commands.
- [admin-state-contract.md](admin-state-contract.md) — shared state/action contract for the web UI, TUI, installer, and future admin API.
- [maturity-and-escape-hatches.md](maturity-and-escape-hatches.md) — feature maturity levels, safe fallback behavior, and clear paths from experimental to stable.

## Core product decision

ModelRouter should support two first-class operating styles inside one local AI
control center and routing/control plane:

1. **Decision router mode** — current differentiator. The proxy classifies each request and selects the best route/backend/model.
2. **Basic router mode** — decision layer disabled. The proxy behaves like a simple OpenAI-compatible router/model selector with explicit backend/model selection, model aliases, fallback, health, telemetry, and manual policy controls.

Both modes should share the same installed-model library, runtime controls, settings UI, TUI, installer, telemetry shape, and explicit mutation safety model.

The control center should cover common local-model workflows: discover models,
recommend route-specific models, plan and run confirmed downloads, start or stop
configured local runtimes, expose a local OpenAI-compatible endpoint, and route
requests. It should integrate with LM Studio, Ollama, LocalAI, llama.cpp,
MLX/MLX-LM, vLLM, generic OpenAI-compatible backends, and hosted providers
without locking users into any one runtime or building a custom inference engine.

UI work should prioritize the operational control plane: proxy status, routing
mode, active backend/model, local runtime status, model library,
recommendation/download state, telemetry/cost/outcome/catalog coverage, and
safety/policy state. Recommendation and download areas should be compact by
default; chat or playground concepts stay secondary if they appear at all.
