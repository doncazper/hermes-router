# Codex implementation pack

This folder turns the installer, TUI, UI, model-library, and basic-router-mode discussion into implementation-ready work for Codex.

## Files

- [control-plane-roadmap.md](control-plane-roadmap.md) — product direction, scope, milestones, and implementation order.
- [ui-tui-wireframes.md](ui-tui-wireframes.md) — mock screenshots, page/tab purposes, and the backend state/actions each field must use.
- [milestones-and-prompts.md](milestones-and-prompts.md) — prepared Codex prompts with acceptance criteria for each milestone.
- [admin-state-contract.yaml](admin-state-contract.yaml) — machine-readable state/action contract for the web UI, TUI, installer, and future admin API.
- [maturity-and-escape-hatches.md](maturity-and-escape-hatches.md) — feature maturity levels, safe fallback behavior, and clear paths from experimental to stable.

## Core product decision

ModelRouter should support two first-class operating styles:

1. **Decision router mode** — current differentiator. The proxy classifies each request and selects the best route/backend/model.
2. **Basic router mode** — decision layer disabled. The proxy behaves like a simple OpenAI-compatible router/model selector with explicit backend/model selection, model aliases, fallback, health, telemetry, and manual policy controls.

Both modes should share the same installed-model library, runtime controls, settings UI, TUI, installer, telemetry shape, and explicit mutation safety model.
