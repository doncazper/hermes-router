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
- `availability.py` validates declared engine availability without executing
  commands or calling providers.
- `scorer.py` performs fast deterministic prompt analysis with regular
  expressions and length heuristics.
- `policy.py` maps scores and features to engine categories.
- `receipts.py` converts routing decisions into serializable receipts.
- `setup_assistant.py` scans local commands/model directories and produces
  setup recommendations without downloading or executing models.
- `cli.py` exposes decision, validation, and setup commands.

## Scoring Dimensions

The scorer inspects:

- Prompt length and estimated token count.
- Coding and repository intent.
- Current-information or citation-backed research intent.
- Multimodal vision, screenshots, OCR, charts, and image-description intent.
- Image generation or local diffusion intent.
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

- `intent_router`: the router/classifier role itself; the MVP uses fast
  heuristics and catalogs this role for future second-pass classifiers.
- `fast_local`: simple rewrite, extraction, copyediting, and formatting.
- `balanced_local`: ordinary summarization and general tasks.
- `reasoning_local`: architecture, deep planning, long-context, or uncertain
  prompts.
- `code_agent`: deep coding, repository, shell, tests, Git, or implementation
  work.
- `web_research`: current/fresh research, citation-heavy prompts, local RAG,
  and HTML extraction.
- `multimodal_vision`: screenshots, charts, OCR, diagrams, and image
  description.
- `image_generation`: local diffusion or image-generation adapter requests.
- `human_confirm`: high-risk, destructive, sending, purchasing, or fail-closed
  decisions.

When the router is uncertain, it routes upward to a safer or stronger category.
When config is missing or invalid, it fails closed to `human_confirm`.

## Model Catalog And User Setup

The model catalog lives at `configs/model_router.yaml`. It defines engine
categories and semantic routing targets rather than hardcoding provider model
names throughout the code.

For machine-specific setup, use one of three paths:

1. Copy and edit `configs/model_router.local.example.yaml`.
2. Run `setup scan` and `setup recommend` to inspect local models and commands.
3. Run `setup wizard` if you want Hermes to ask before writing a config.
4. Run `setup write` to generate a local YAML file from those recommendations.

The generated local config can be passed explicitly:

```bash
python -m hermes.plugins.model_router.cli decide \
  --config configs/model_router.local.yaml \
  "fix the repo and run tests"
```

### Setup Assistant

The setup assistant is intentionally safe and local-first. It scans:

- Known local model directories such as Hugging Face cache, Ollama, LM Studio,
  and `~/models`.
- Optional `--model-dir` paths supplied by the user.
- Command availability for tools such as `claude`, `codex`, `hf`, `ollama`,
  `llama-server`, and `lmstudio`.

It does not execute model providers, call external APIs, download files, or
modify the default catalog during `scan`, `recommend`, `write`, or `wizard`.
Downloads require the separate `setup download --execute` path.

Scan:

```bash
python -m hermes.plugins.model_router.cli setup scan
python -m hermes.plugins.model_router.cli setup scan --json
```

Recommend:

```bash
python -m hermes.plugins.model_router.cli setup recommend
python -m hermes.plugins.model_router.cli setup recommend --json
```

Interactive wizard:

```bash
python -m hermes.plugins.model_router.cli setup wizard \
  --output configs/model_router.local.yaml
```

Write a generated config:

```bash
python -m hermes.plugins.model_router.cli setup write \
  --output configs/model_router.local.yaml
```

The writer will not overwrite an existing file unless `--force` is passed.

### Hugging Face Download Plans

Setup recommendations include `hf download` commands for missing local roles,
and `setup download` shows the same plan without running it:

```bash
python -m hermes.plugins.model_router.cli setup download
python -m hermes.plugins.model_router.cli setup download --route fast_local
```

Execution is a separate opt-in:

```bash
python -m hermes.plugins.model_router.cli setup download \
  --route fast_local \
  --execute
```

The command asks for confirmation before running `hf download`. For
non-interactive scripts, pass `--yes`. This keeps large downloads, gated
licenses, and hardware choices under user control.

Users can also provide their own Hugging Face repo id:

```bash
python -m hermes.plugins.model_router.cli setup download \
  --route balanced_local \
  --repo-id custom-org/custom-model \
  --execute
```

Users choose which model or agent handles each task class by editing
`routing_targets`. For example, coding work can point to a local code engine,
Claude Code, Codex, or any other configured engine:

```yaml
routing_targets:
  simple: fast_local
  balanced: balanced_local
  reasoning: reasoning_local
  coding: claude_code
  research: web_research
  vision: multimodal_vision
  image_generation: image_generation
  confirmation: human_confirm
```

Then define or enable the referenced engine:

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
    enabled: true
    fallback: code_agent
    availability:
      status: auto
      required_commands:
        - claude
```

The built-in catalog also includes disabled examples for `claude_code` and
`codex`. To use one, set `enabled: true` on that engine and set
`routing_targets.coding` to its engine name. Local coding can stay on
`code_agent`, whose provider/model/adapter fields can be changed to match the
user's local runtime.

The example engine roles from the design map to the catalog like this:

| Role | Catalog Coverage |
| --- | --- |
| Intent classifier/router | `intent_router`, plus the deterministic router code in this MVP |
| Deep reasoning/coding | `reasoning_local` for planning and `code_agent` for repo/code execution |
| Fast response/summarization | `fast_local` and `balanced_local` |
| Web research/RAG | `web_research` |
| Multimodal/vision | `multimodal_vision` |
| Image generation | `image_generation` |

## Availability Validation

Each engine can declare availability checks:

```yaml
availability:
  status: auto
  required_env:
    - OPENAI_API_KEY
  required_commands:
    - codex
  required_paths:
    - ~/.config/my-local-runtime
```

`status` accepts:

- `auto`: available when all declared checks pass.
- `available`: manually marked available, while still enforcing declared checks.
- `unavailable`: always treated as unavailable.

The validator never executes commands or calls provider APIs. It only checks
environment-variable presence, command presence on `PATH`, and local path
existence. Environment variable values are never printed.

Run:

```bash
python -m hermes.plugins.model_router.cli validate-config
python -m hermes.plugins.model_router.cli validate-config --json
```

Routing uses the same safe validation. If a selected engine is unavailable, the
router follows its fallback chain. If no available fallback exists, the decision
fails closed to `human_confirm` and includes availability reasons in the
receipt.

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
availability:
  status: auto
```

The required categories are:

```text
intent_router
fast_local
balanced_local
reasoning_local
code_agent
web_research
multimodal_vision
image_generation
human_confirm
```

The required routing targets are:

```text
simple
balanced
reasoning
coding
research
vision
image_generation
confirmation
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
  "selected_engine": "code_agent",
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
  "requires_vision": false,
  "requires_image_generation": false,
  "availability_valid": true,
  "availability_reasons": [
    "code_agent: no availability requirements declared"
  ],
  "config_valid": true
}
```

## Known Limitations

- Scoring is heuristic and conservative.
- The router does not use an LLM to classify prompts.
- Availability checks are declarative and local; they do not prove a provider
  API call will succeed.
- Receipts intentionally do not include the raw prompt.
- The CLI exits successfully when it emits a fail-closed receipt; the decision
  itself carries `config_valid: false`.

## Future Gateway Mode

Future milestones can add a gateway that dispatches decisions to actual
engines. That work should remain behind explicit confirmation gates for risky
actions, preserve receipt emission, and keep the decision logic testable without
provider calls.
