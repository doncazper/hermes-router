# Hermes Router

Hermes Router is a deterministic decision router for AI prompts. It quickly
scores prompt complexity, risk, freshness needs, and tool needs, then selects a
configured engine category and emits a JSON-safe routing receipt.

This repository is the first milestone of an OpenRouter-like system for Hermes:
it decides where work should go, but it does not dispatch, execute prompts, call
external model APIs, run tools, or perform user actions.

## What It Does

- Scores prompt complexity, risk, and confidence without using an LLM.
- Detects coding/repo work, current research, high-risk actions, structured
  output, tool intent, ambiguity, and sensitive domains.
- Routes prompts to configured engines such as local models, Claude Code,
  Codex, web research, or human confirmation.
- Emits explainable routing receipts that are safe to serialize as JSON.
- Fails closed to `human_confirm` when config is missing or invalid.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/doncazper/Hermes-Router.git
cd Hermes-Router
python -m pip install -e ".[dev]"
```

If your system Python is older, use `uv`:

```bash
uv run --python 3.11 --with pytest --with PyYAML python -m pytest
```

## CLI Usage

Readable decision output:

```bash
python -m hermes.plugins.model_router.cli decide "rewrite this text"
```

JSON receipt output:

```bash
python -m hermes.plugins.model_router.cli decide --json "fix the repo and run tests"
```

Use a custom catalog:

```bash
python -m hermes.plugins.model_router.cli decide \
  --config configs/model_router.yaml \
  "research current GLP-1 supplement trends"
```

Example JSON receipt:

```json
{
  "complexity_score": 45,
  "confidence_score": 90,
  "config_valid": true,
  "fallback_engine": "reasoning_local",
  "reasons": [
    "coding or repository intent",
    "tool use likely",
    "file, shell, or GitHub operation",
    "coding or repository work"
  ],
  "requires_code_execution": true,
  "requires_confirmation": false,
  "requires_freshness": false,
  "requires_tools": true,
  "risk_score": 25,
  "selected_engine": "code_agent"
}
```

## Configure Models And Agents

Edit `configs/model_router.yaml`.

The router separates semantic routes from concrete engines:

```yaml
routing_targets:
  simple: fast_local
  balanced: balanced_local
  reasoning: reasoning_local
  coding: code_agent
  research: web_research
  confirmation: human_confirm
```

Each route points at an engine entry. For example, coding can use the default
local code agent, Claude Code, Codex, or any user-defined local/remote engine:

```yaml
routing_targets:
  coding: claude_code

engines:
  claude_code:
    provider: anthropic
    model: claude-code
    adapter: claude_code
    strengths:
      - repository edits
      - tests
    max_context: 200000
    cost_tier: high
    latency_tier: medium
    enabled: true
    fallback: code_agent
```

The included catalog has disabled examples for `claude_code` and `codex`.
Enable one by setting `enabled: true` and pointing `routing_targets.coding` at
that engine name. Local users can keep `coding: code_agent` and change the
`provider`, `model`, and `adapter` fields for their local runtime.

## Default Routes

| Route | Default Engine | Intended Use |
| --- | --- | --- |
| `simple` | `fast_local` | Rewrites, extraction, formatting |
| `balanced` | `balanced_local` | Summaries and ordinary general tasks |
| `reasoning` | `reasoning_local` | Planning, architecture, long-context reasoning |
| `coding` | `code_agent` | Repo edits, code, tests, shell/Git workflows |
| `research` | `web_research` | Current research and citation-heavy prompts |
| `confirmation` | `human_confirm` | High-risk or fail-closed decisions |

## Development

Run tests:

```bash
python -m pytest
```

Run lint:

```bash
python -m ruff check .
```

With `uv` and Python 3.11:

```bash
uv run --python 3.11 --with pytest --with PyYAML python -m pytest
uv run --python 3.11 --with ruff --with PyYAML python -m ruff check .
```

## Project Layout

```text
hermes/plugins/model_router/
  models.py      # Dataclass models and JSON-safe serialization helpers
  config.py      # YAML catalog loading and validation
  scorer.py      # Deterministic heuristic prompt scoring
  policy.py      # Engine selection and fail-closed fallback rules
  receipts.py    # Routing receipt helpers
  cli.py         # CLI entrypoint
configs/
  model_router.yaml
tests/
docs/
  model-router.md
```

## Safety Model

- The router never executes user requests.
- The router never sends email, deletes files, buys anything, or calls external
  APIs.
- High-risk external actions require confirmation.
- Missing or invalid config routes to `human_confirm`.
- Receipts omit raw prompt text.

## Roadmap

- Add richer model/agent capability metadata.
- Add optional availability checks for local and remote engines.
- Add gateway dispatch behind explicit confirmation gates.
- Add telemetry-free receipt storage for audit trails.
- Add learned or LLM-assisted classification as an optional second-pass scorer.

See `docs/model-router.md` for more detail on scoring, routing policy, and
known limitations.
