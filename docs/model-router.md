# Hermes Model Router

## Purpose

The model router is a deterministic decision layer for Hermes prompts. It
scores an incoming prompt, selects an engine category, and emits a receipt that
explains the decision.

This milestone does not execute prompts, call model providers, perform web
research, run code, send messages, delete files, purchase anything, or dispatch
to an agent. It only decides.

## Architecture

The router is implemented under `hermes/plugins/model_router/`.

- `models.py` defines JSON-safe dataclass models for engines, prompt features,
  scores, decisions, receipts, and config.
- `config.py` loads and validates the engine catalog from
  `configs/model_router.yaml`.
- `scorer.py` performs fast deterministic prompt analysis with regular
  expressions and length heuristics.
- `policy.py` maps scores and features to engine categories.
- `receipts.py` converts routing decisions into serializable receipts.
- `cli.py` exposes the `decide` command.

## Scoring Dimensions

The scorer inspects:

- Prompt length and estimated token count.
- Coding and repository intent.
- Current-information or citation-backed research intent.
- Multi-step reasoning, planning, architecture, and long-context needs.
- Tool, file, shell, GitHub, email, and calendar intent.
- Legal, medical, and financial sensitivity.
- Destructive, sending, purchasing, payment, scheduling, publishing, and other
  external actions.
- Structured output requests.
- Ambiguity in short high-impact prompts.

High-risk external actions raise risk even when the prompt is short.

## Routing Policy

The default engine categories are:

- `fast_local`: simple rewrite, extraction, copyediting, and formatting.
- `balanced_local`: ordinary summarization and general tasks.
- `reasoning_local`: architecture, deep planning, long-context, or uncertain
  prompts.
- `codex`: code, repository, shell, tests, Git, or implementation work.
- `web_research`: current/fresh research and citation-heavy prompts.
- `human_confirm`: high-risk, destructive, sending, purchasing, or fail-closed
  decisions.

When the router is uncertain, it routes upward to a safer or stronger category.
When config is missing or invalid, it fails closed to `human_confirm`.

## Model Catalog

The model catalog lives at `configs/model_router.yaml`. It defines engine
categories rather than hardcoding provider model names throughout the code.

Each engine supports:

```yaml
provider: local
model: hermes-balanced-local
adapter: local_chat
strengths:
  - summarization
max_context: 16384
cost_tier: free
latency_tier: medium
enabled: true
fallback: reasoning_local
```

The required categories are:

```text
fast_local
balanced_local
reasoning_local
codex
web_research
human_confirm
```

Fallbacks are followed only for decision routing. They do not execute any
provider call.

## CLI Usage

Readable output:

```bash
python -m hermes.plugins.model_router.cli decide "rewrite this text"
```

JSON receipt output:

```bash
python -m hermes.plugins.model_router.cli decide --json "fix the repo and run tests"
```

Custom catalog:

```bash
python -m hermes.plugins.model_router.cli decide --config configs/model_router.yaml "research current GLP-1 supplement trends"
```

Example receipt:

```json
{
  "selected_engine": "codex",
  "complexity_score": 45,
  "risk_score": 25,
  "confidence_score": 90,
  "reasons": [
    "coding or repository intent",
    "tool use likely",
    "file, shell, or GitHub operation",
    "coding or repository work"
  ],
  "fallback_engine": "reasoning_local",
  "requires_confirmation": false,
  "requires_tools": true,
  "requires_freshness": false,
  "requires_code_execution": true,
  "config_valid": true
}
```

## Known Limitations

- Scoring is heuristic and conservative.
- The router does not use an LLM to classify prompts.
- The router does not verify whether a configured model is installed or
  reachable.
- Receipts intentionally do not include the raw prompt.
- The CLI exits successfully when it emits a fail-closed receipt; the decision
  itself carries `config_valid: false`.

## Future Gateway Mode

Future milestones can add a gateway that dispatches decisions to actual
engines. That work should remain behind explicit confirmation gates for risky
actions, preserve receipt emission, and keep the decision logic testable without
provider calls.
