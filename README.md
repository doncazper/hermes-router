# Hermes Router

Deterministic, fast, safety-first model routing for Hermes.

Hermes Router is a small Python decision layer that scores an incoming prompt,
selects the configured engine that should handle it, and emits a JSON-safe
receipt explaining why. It keeps routing cheap and predictable: simple work can
go to fast local engines, complex work to stronger reasoning engines, fresh
research to research tools, repo work to code agents, and risky actions to human
confirmation.

This project is intentionally a decision router only. It does not execute
prompts, call model providers, load local model weights, browse the web, run
shell commands, send messages, delete files, or purchase anything.

## At a Glance

| Need | Hermes Router provides |
| --- | --- |
| Fast hot-path routing | `ModelRouter.route_fast(prompt)` returns an engine string |
| Diagnostic decisions | `ModelRouter.route(prompt)` returns scores, flags, reasons, and alternatives |
| CLI integration | `decide`, `validate-config`, `dispatch-plan`, and `setup` commands |
| Local/API flexibility | YAML routing targets for local models, hosted APIs, Codex, Claude Code, vision, image generation, and custom adapters |
| Safety boundaries | High-risk or invalid requests fail closed to `human_confirm` |
| Setup help | Safe local scans, config recommendations, and opt-in Hugging Face download plans |

## Highlights

- Deterministic heuristic routing with no LLM classification call.
- Fast initialized hot path: `router.route_fast(prompt)` returns only the
  selected engine.
- Rich receipt path: `router.route(prompt)` returns scores, reasons, rejected
  engines, alternatives, requirements, and safety flags.
- YAML-driven engine catalog; model names are not hardcoded throughout the
  router.
- User-configurable routing targets for local models, hosted APIs, Codex,
  Claude Code, web/RAG tools, vision, image generation, or custom adapters.
- Fail-closed safety: missing/invalid config and high-risk actions route to
  `human_confirm`.
- Declarative availability checks for env vars, commands, and local paths.
- Setup assistant for local/API/mixed model configuration and optional Hugging
  Face download plans.

## Project Status

Hermes Router is a lean production-ready decision layer when embedded through
the initialized Python API. The stable surface today is:

- `ModelRouter.route_fast(...)` for production routing.
- `ModelRouter.route(...)` for diagnostic and audit receipts.
- Config-driven model/agent catalog.
- Safe dry-run dispatch plans.
- Local setup wizard and recommendations.

Gateway execution is intentionally not implemented. Future dispatch should stay
behind explicit adapter boundaries and confirmation gates.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/doncazper/hermes-router.git
cd hermes-router
python -m pip install -e ".[dev]"
```

If your shell does not provide `python`, use `python3`. If your system Python is
older, use `uv`:

```bash
uv run --python 3.11 --with pytest --with PyYAML python -m pytest
```

## Quick Start

Readable CLI output:

```bash
python -m hermes.plugins.model_router.cli decide "rewrite this text"
```

JSON receipt:

```bash
python -m hermes.plugins.model_router.cli decide --json "fix the repo and run tests"
```

Expected default routing examples:

| Prompt | Selected engine |
| --- | --- |
| `rewrite this text` | `fast_local` |
| `summarize these notes` | `balanced_local` |
| `design a distributed task scheduler architecture` | `reasoning_local` |
| `fix the repo and run tests` | `code_agent` |
| `search the web for the latest TypeScript release notes` | `web_research` |
| `extract text from this screenshot` | `multimodal_vision` |
| `generate an image of a router dashboard` | `image_generation` |
| `drop the production database` | `human_confirm` |

## Python API

Initialize once and reuse the router. Runtime calls stay in memory and do not
re-read YAML, scan disk, or run setup helpers.

```python
from hermes.plugins.model_router import ModelRouter

router = ModelRouter.from_config("configs/model_router.yaml")

# Production hot path: selected engine only.
engine = router.route_fast("fix the repo and run tests")

# Diagnostic path: scores, reasons, rejected engines, alternatives, and flags.
decision = router.route("fix the repo and run tests")

print(engine)
print(decision.requires_code_execution)
```

Use `route_fast(...)` for production routing, live routing loops, UI
responsiveness, and high-volume classification. Use `route(...)` when you need
a receipt, explanation, audit trail, or ranked alternatives. If you need a rich
decision but not ranked alternatives:

```python
decision = router.route("rewrite this text", include_alternatives=False)
```

For one-off scripts, the compatibility function remains available:

```python
from hermes.plugins.model_router import route_prompt

decision = route_prompt("research current GLP-1 supplement trends")
```

## CLI

After installation, you can use the console command:

```bash
hermes-router decide "rewrite this text"
hermes-router decide --json "fix the repo and run tests"
```

Use a custom catalog:

```bash
python -m hermes.plugins.model_router.cli decide \
  --config configs/model_router.local.yaml \
  "research current GLP-1 supplement trends"
```

Pass routing hints:

```bash
python -m hermes.plugins.model_router.cli decide \
  --attachment image \
  --force-engine multimodal_vision \
  --max-cost-tier medium \
  --max-latency-tier medium \
  "summarize this attachment"
```

Validate a config:

```bash
python -m hermes.plugins.model_router.cli validate-config
python -m hermes.plugins.model_router.cli validate-config --json
```

Create a dry-run dispatch plan:

```bash
python -m hermes.plugins.model_router.cli dispatch-plan "fix the repo and run tests"
python -m hermes.plugins.model_router.cli dispatch-plan --json "rewrite this text"
python -m hermes.plugins.model_router.cli dispatch-plan --include-alternatives --json "rewrite this text"
```

Dispatch plans only describe what a future adapter would do. They do not execute
models, tools, shell commands, provider calls, or external actions. They skip
ranked alternatives by default for speed; pass `--include-alternatives` when a
full receipt is useful.

## Example Receipt

```json
{
  "selected_engine": "code_agent",
  "complexity_score": 56,
  "risk_score": 38,
  "confidence_score": 90,
  "fallback_engine": "reasoning_local",
  "requires_confirmation": false,
  "requires_tools": true,
  "requires_freshness": false,
  "requires_code_execution": true,
  "requires_vision": false,
  "requires_image_generation": false,
  "config_valid": true,
  "availability_valid": true,
  "reasons": [
    "coding or repository intent",
    "tool use likely",
    "file, shell, or GitHub operation",
    "coding or repository work"
  ],
  "rejected_engines": [
    {
      "engine": "fast_local",
      "reason": "tools required but engine does not support tools"
    }
  ],
  "alternatives": [
    {
      "engine": "web_research",
      "rank_score": 61,
      "capability": 70,
      "trust": 60,
      "cost": 50,
      "latency": 75,
      "reasons": [
        "capability 70/100",
        "trust 60/100",
        "cost 50/100",
        "latency 75/100"
      ]
    }
  ]
}
```

Receipts intentionally do not include the raw prompt.

## Configure Engines

The default catalog lives at `configs/model_router.yaml`. Machine-specific
settings should go in `configs/model_router.local.yaml` and be passed with
`--config`.

Routing targets map semantic routes to configured engines:

```yaml
routing_targets:
  simple: fast_local
  balanced: balanced_local
  reasoning: reasoning_local
  coding: code_agent
  research: web_research
  vision: multimodal_vision
  image_generation: image_generation
  confirmation: human_confirm
```

Each target points at an engine entry:

```yaml
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
    capability: 90
    trust: 90
    cost: 80
    latency: 45
    supports_tools: true
    enabled: true
    fallback: code_agent
    availability:
      status: auto
      required_commands:
        - claude
```

Coding does not have to use Codex. You can point `routing_targets.coding` at
`claude_code`, `codex`, `code_agent`, a local coding model, or any custom
engine you define.

Optional numeric metadata uses a 0-100 scale:

- `capability`: model/agent strength.
- `trust`: reliability for sensitive work.
- `cost`: relative cost, where higher means more expensive.
- `latency`: relative latency, where higher means slower.

These values rank compatible alternatives. They do not override the configured
target when that target is enabled, available, and compatible.

## Setup Assistant

Hermes can help create a local config without guessing what you want.

Scan your machine:

```bash
python -m hermes.plugins.model_router.cli setup scan
python -m hermes.plugins.model_router.cli setup scan --json
```

Get recommendations:

```bash
python -m hermes.plugins.model_router.cli setup recommend
python -m hermes.plugins.model_router.cli setup recommend --json
```

Run the wizard:

```bash
python -m hermes.plugins.model_router.cli setup wizard \
  --output configs/model_router.local.yaml
```

Write a recommended config non-interactively:

```bash
python -m hermes.plugins.model_router.cli setup write \
  --output configs/model_router.local.yaml
```

`setup write` will not overwrite an existing file unless `--force` is passed.

The wizard asks whether you want:

- Local LLMs only.
- API keys / hosted models.
- A mix of local models, hosted APIs, and agent tools.

It then walks each main route and shows numbered local model choices plus
recommended downloads when a local role is missing. Downloads are never run by
ordinary routing commands. They require explicit confirmation.

The scanner includes current LM Studio model storage at
`~/.lmstudio/models`, plus Ollama, Hugging Face cache, and common local model
folders, so wizard choices should reflect the models your local tools can see.

If recommended downloads are available and the Hugging Face `hf` CLI is missing,
the wizard warns at the beginning and asks whether to install it into the current
Python environment before model choices start. Declining is safe; Hermes can
still write the config, and downloads can be run later.

Plan downloads:

```bash
python -m hermes.plugins.model_router.cli setup download
python -m hermes.plugins.model_router.cli setup download --route fast_local
```

Run an approved Hugging Face download:

```bash
python -m hermes.plugins.model_router.cli setup download \
  --route balanced_local \
  --repo-id custom-org/custom-model \
  --execute
```

For non-interactive scripts, add `--yes`.

## Engine Roles

| Role | Default coverage |
| --- | --- |
| Intent classifier/router | `intent_router` plus deterministic router code |
| Fast response/summarization | `fast_local`, `balanced_local` |
| Deep reasoning/planning | `reasoning_local` |
| Coding/repo work | `code_agent`, with optional `codex` or `claude_code` |
| Web research/RAG | `web_research` |
| Multimodal/vision/OCR | `multimodal_vision` |
| Image generation | `image_generation` |
| Confirmation/fail-closed | `human_confirm` |

## Safety Model

- The router never executes user requests.
- The router never calls hosted model APIs.
- The router never loads local model weights.
- The router never sends email, deletes files, buys anything, or runs shell
  commands.
- High-risk destructive, sending, purchasing, payment, scheduling, publishing,
  and external-action prompts require confirmation.
- `force_engine` cannot bypass human confirmation.
- Missing or invalid config routes to `human_confirm`.
- Unavailable or incompatible engines are skipped through configured fallbacks.
- Receipts omit raw prompt text.

## Performance

Use the initialized API for runtime performance:

```bash
python scripts/benchmark_route_fast.py
python scripts/benchmark_route_fast.py --json
python scripts/check_route_fast_latency.py --json
```

`route_fast(...)` is the production hot path. It returns only the selected
engine string. The scorer precompiles its stable regex patterns at import time, and
initialized routers keep YAML config and availability results in memory. The
richer `route(...)` path does more work by design because it builds scores,
explanations, rejected-engine details, alternatives, and receipt fields.

The CLI is intended for humans, diagnostics, and scripts. Latency-sensitive
services should not spawn a Python process per prompt; instantiate `ModelRouter`
once and call the Python API in process.

The default production SLO for initialized ordinary prompts is <= 25 us best
sample and <= 50 us mean sample for `route_fast(...)`. The benchmark guard
enforces those budgets in CI. See
[Production readiness](docs/production-readiness.md) for the API contract,
benchmark command, SLOs, and logging guidance.

## Install For Local Testing

Use a virtual environment so the router does not modify a managed Python
installation:

```bash
cd /path/to/hermes-router
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
hermes-router decide --json "fix the repo and run tests"
```

For a non-editable install from GitHub:

```bash
python -m pip install "git+https://github.com/doncazper/hermes-router.git@v0.3.0"
hermes-router decide "rewrite this text"
```

The package exposes a console command, `hermes-router`, and the importable
Python API:

```python
from hermes.plugins.model_router import ModelRouter

router = ModelRouter.from_config()
engine = router.route_fast(prompt)
```

The default catalog is included as package data, so `ModelRouter.from_config()`
works after wheel installation without relying on the repository checkout. Pass
an explicit config path when an embedding app needs its own engine catalog.

Hermes Router does not currently claim a Desktop plugin manifest or automatic
per-turn model switching. Any Desktop integration should use that app's actual
plugin/API contract and call the stable `route_fast(...)` production API.

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
  availability.py     # Non-executing availability validation
  cli.py              # CLI entrypoint
  config.py           # YAML catalog loading and validation
  data/               # Packaged default config
  dispatch.py         # Safe dry-run dispatch plans
  models.py           # Dataclass models and JSON-safe serialization
  policy.py           # Engine selection and fail-closed fallback rules
  receipts.py         # Routing receipt helpers
  scorer.py           # Deterministic heuristic prompt scoring
  setup_assistant.py  # Local setup scanning and config recommendations
configs/
  model_router.yaml
  model_router.local.example.yaml
docs/
  adapter-contract.md
  model-router.md
scripts/
  benchmark_route_fast.py
tests/
```

## Documentation

- [Model router details](docs/model-router.md)
- [Production readiness](docs/production-readiness.md)
- [Future adapter contract](docs/adapter-contract.md)

## Roadmap

- Keep the router small, deterministic, and fast.
- Add opt-in provider health checks that do not execute user tasks.
- Add gateway dispatch only behind explicit adapter contracts and confirmation
  gates.
- Consider optional second-pass classification for uncertain prompts without
  slowing down the normal path.
