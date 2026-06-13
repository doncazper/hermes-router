# Hermes Router

Hermes Router is a deterministic decision router for AI prompts. It quickly
scores prompt complexity, risk, freshness needs, and tool needs, then selects a
configured engine category and emits a JSON-safe routing receipt.

This repository is the first milestone of an OpenRouter-like system for Hermes:
it decides where work should go, but it does not dispatch, execute prompts, call
external model APIs, run tools, or perform user actions.

## What It Does

- Scores prompt complexity, risk, and confidence without using an LLM.
- Detects coding/repo work, current research, multimodal vision, image
  generation, high-risk actions, structured output, tool intent, ambiguity, and
  sensitive domains.
- Routes prompts to configured engines such as local models, Claude Code,
  Codex, web/RAG research, vision, image generation, or human confirmation.
- Validates declared engine availability before choosing an engine.
- Emits explainable routing receipts, alternatives, and rejection reasons that
  are safe to serialize as JSON.
- Fails closed to `human_confirm` when config is missing or invalid.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/doncazper/Hermes-Router.git
cd Hermes-Router
python -m pip install -e ".[dev]"
```

If your shell does not have a `python` command, use `python3` for the examples
below. If your system Python is older, use `uv`:

```bash
uv run --python 3.11 --with pytest --with PyYAML python -m pytest
uv run --python 3.11 --with PyYAML python -m hermes.plugins.model_router.cli setup wizard
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

Pass routing hints:

```bash
python -m hermes.plugins.model_router.cli decide \
  --attachment image \
  --force-engine multimodal_vision \
  --max-cost-tier medium \
  --max-latency-tier medium \
  "summarize this attachment"
```

Example JSON receipt:

```json
{
  "complexity_score": 45,
  "confidence_score": 90,
  "config_valid": true,
  "fallback_engine": "reasoning_local",
  "fallback_used": false,
  "availability_valid": true,
  "availability_reasons": [
    "code_agent: no availability requirements declared"
  ],
  "reasons": [
    "coding or repository intent",
    "tool use likely",
    "file, shell, or GitHub operation",
    "coding or repository work"
  ],
  "requires_code_execution": true,
  "requires_confirmation": false,
  "requires_freshness": false,
  "requires_image_generation": false,
  "requires_tools": true,
  "requires_vision": false,
  "requirements": {
    "max_cost_tier": null,
    "max_latency_tier": null,
    "needs_tools": true,
    "required_modalities": []
  },
  "rejected_engines": [],
  "alternatives": [
    {
      "engine": "reasoning_local",
      "rank_score": 76,
      "capability": 80,
      "trust": 60,
      "cost": 0,
      "latency": 75,
      "reasons": [
        "capability 80/100",
        "trust 60/100",
        "cost 0/100",
        "latency 75/100"
      ]
    }
  ],
  "risk_score": 25,
  "selected_engine": "code_agent"
}
```

## Runtime API

Use the initialized router API when embedding Hermes in another process. This
loads YAML, validates static config, and caches availability once; each
call stays in memory.

```python
from hermes.plugins.model_router import ModelRouter

router = ModelRouter.from_config("configs/model_router.local.yaml")

# Fast hot path: selected engine only.
engine = router.route_fast("fix the repo and run tests")

# Rich path: scores, reasons, fallbacks, alternatives, and receipt fields.
decision = router.route("fix the repo and run tests")
```

Use `route_fast(...)` for live routing loops, UI responsiveness, and high-volume
classification. It returns only the selected engine string, never parses YAML
per call, and still routes high-risk actions to `human_confirm`. Use
`route(...)` when you need a receipt or explanation. If you need a rich decision
but not ranked alternatives, call `router.route(prompt, include_alternatives=False)`.

The older `route_prompt(...)` function remains available for one-off scripts and
CLI-style usage.

## Dry-Run Dispatch Plans

Hermes can also produce a safe dispatch plan without executing anything:

```bash
python -m hermes.plugins.model_router.cli dispatch-plan "fix the repo and run tests"
python -m hermes.plugins.model_router.cli dispatch-plan --json "rewrite this text"
```

Dispatch plans name the selected provider, model, and adapter, then say whether
future dispatch would be allowed. They never load model weights, start Ollama or
LM Studio, call hosted APIs, run shell commands, or perform external actions.
See [docs/adapter-contract.md](docs/adapter-contract.md) for the future adapter
boundary and lazy-loading guidance.

## Configure Models And Agents

Hermes supports a hybrid setup process:

- Edit a plain YAML file directly.
- Ask the setup assistant to scan local commands, API-key presence, and model
  cache directories.
- Use an interactive wizard that asks whether you want local LLMs, API-backed
  engines, or a mix, then walks each route category.
- Generate a recommended local config file.
- Review Hugging Face download-plan commands before downloading anything.
- Execute approved Hugging Face downloads with an explicit confirmation gate.

The default catalog is `configs/model_router.yaml`. For machine-specific
settings, use `configs/model_router.local.yaml` and pass it with `--config`.
You can start from `configs/model_router.local.example.yaml`.

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

Use the interactive wizard:

```bash
python -m hermes.plugins.model_router.cli setup wizard \
  --output configs/model_router.local.yaml
```

The wizard asks for:

- Local-only, API-only, or mixed setup.
- Numbered local model choices discovered on your machine.
- Numbered recommended download choices when a route has no local model yet.
- Simple, balanced, reasoning, coding, research, vision, and image-generation
  route choices.
- Direct route overrides by engine name, such as `claude_code`, `codex`,
  `openai_api`, `anthropic_api`, `balanced_local`, or `reasoning_local`.
- Final confirmation before writing `configs/model_router.local.yaml`.
- Optional download confirmation for any recommended models you selected.

Write a local config:

```bash
python -m hermes.plugins.model_router.cli setup write \
  --output configs/model_router.local.yaml
```

Plan downloads without running them:

```bash
python -m hermes.plugins.model_router.cli setup download
python -m hermes.plugins.model_router.cli setup download --route fast_local
```

Run approved downloads:

```bash
python -m hermes.plugins.model_router.cli setup download \
  --route fast_local \
  --execute
```

For non-interactive use, add `--yes`. Downloads use the Hugging Face `hf`
CLI and are never run by `decide`, `recommend`, or `write`. The interactive
wizard only runs downloads for recommended models you selected, and only after
it asks for a separate confirmation.

Download your own preferred Hugging Face model:

```bash
python -m hermes.plugins.model_router.cli setup download \
  --route balanced_local \
  --repo-id custom-org/custom-model \
  --execute
```

The router separates semantic routes from concrete engines:

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
    capability: 90
    trust: 90
    cost: 80
    latency: 45
    enabled: true
    fallback: code_agent
    availability:
      status: auto
      required_commands:
        - claude
```

The included catalog has disabled examples for `claude_code` and `codex`.
Enable one by setting `enabled: true` and pointing `routing_targets.coding` at
that engine name. Local users can keep `coding: code_agent` and change the
`provider`, `model`, and `adapter` fields for their local runtime.

If the setup assistant detects the `claude` command, it recommends
`routing_targets.coding: claude_code`. If it detects `codex` but not `claude`,
it recommends `routing_targets.coding: codex`. Otherwise, it keeps coding on
the local `code_agent` fallback.

Engines can optionally include numeric ranking metadata on a 0-100 scale:
`capability`, `trust`, `cost`, and `latency`. If omitted, Hermes derives those
values from the existing tier fields. Numeric values only affect ranked
alternatives; the configured route target remains selected when it is enabled,
available, and compatible.

Prompt scoring is also tunable through the optional top-level `scoring:` section
in the YAML catalog. Missing weights use Hermes defaults; invalid scoring config
is treated as invalid config and routes fail closed.

The default catalog covers these engine roles:

| Role | Default Route/Engine | Notes |
| --- | --- | --- |
| Intent classifier/router | `intent_router` | Catalogs the decision layer itself; the MVP uses deterministic heuristics rather than an LLM call. |
| Deep reasoning/coding | `reasoning_local`, `code_agent` | Split so non-code planning and repo execution can use different engines. |
| Fast response/summarization | `fast_local`, `balanced_local` | Lightweight transforms and ordinary summaries. |
| Web research/RAG | `web_research` | Current research, citations, local RAG, and HTML extraction. |
| Multimodal/vision | `multimodal_vision` | Screenshots, charts, OCR, and image description. |
| Image generation | `image_generation` | Local diffusion or image-generation API adapters. |

## Validate Availability

Availability validation is declarative and safe. It does not execute engines,
run provider health checks, or call external APIs. It only checks configured
signals:

- `status: available`, `auto`, or `unavailable`
- `required_env`: environment variable names that must be present
- `required_commands`: binaries that must exist on `PATH`
- `required_paths`: local paths that must exist

Run:

```bash
python -m hermes.plugins.model_router.cli validate-config
python -m hermes.plugins.model_router.cli validate-config --json
```

During routing, unavailable engines are skipped through their fallback chain. If
no available fallback exists, the router fails closed to `human_confirm`.
Receipts include rejected engines and reasons, such as missing tool support,
missing modality support, or cost/latency tier limits.

High-risk destructive, sending, purchasing, and external-action prompts always
route to `human_confirm`; `--force-engine` cannot bypass that confirmation gate.

## Default Routes

| Route | Default Engine | Intended Use |
| --- | --- | --- |
| `simple` | `fast_local` | Rewrites, extraction, formatting |
| `balanced` | `balanced_local` | Summaries and ordinary general tasks |
| `reasoning` | `reasoning_local` | Planning, architecture, long-context reasoning |
| `coding` | `code_agent` | Repo edits, code, tests, shell/Git workflows |
| `research` | `web_research` | Current research and citation-heavy prompts |
| `vision` | `multimodal_vision` | Screenshots, charts, OCR, and image description |
| `image_generation` | `image_generation` | Local diffusion/image creation requests |
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

Benchmark the initialized hot path:

```bash
python scripts/benchmark_route_fast.py
python scripts/benchmark_route_fast.py --json
uv run --python 3.11 --with PyYAML python scripts/benchmark_route_fast.py --json
```

## Project Layout

```text
hermes/plugins/model_router/
  models.py       # Dataclass models and JSON-safe serialization helpers
  config.py       # YAML catalog loading and validation
  availability.py # Non-executing engine availability validation
  scorer.py       # Deterministic heuristic prompt scoring
  policy.py       # Engine selection and fail-closed fallback rules
  receipts.py     # Routing receipt helpers
  setup_assistant.py # Local setup scanning and config recommendation
  cli.py          # CLI entrypoint
configs/
  model_router.yaml
  model_router.local.example.yaml
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
- Unavailable or unsuitable engines are skipped through fallbacks before dispatch
  is possible.
- Receipts omit raw prompt text.

## Roadmap

- Add active provider health checks behind explicit opt-in.
- Add gateway dispatch behind explicit confirmation gates.
- Add telemetry-free receipt storage for audit trails.
- Add learned or LLM-assisted classification as an optional second-pass scorer.

See `docs/model-router.md` for more detail on scoring, routing policy, and
known limitations.
