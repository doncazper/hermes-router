# Hermes Router

Hermes Model Router is a deterministic decision router for choosing which
engine category should handle a prompt.

## Configure Models

Edit `configs/model_router.yaml`:

- Add or modify entries under `engines`.
- Point semantic routes under `routing_targets` at the engine names users want.
- For coding, set `routing_targets.coding` to `code_agent`, `claude_code`,
  `codex`, or any other configured local/remote coding engine.

See `docs/model-router.md` for CLI usage, receipt examples, and config details.
