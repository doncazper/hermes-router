# ModelRouter

## Purpose

The model router is a local AI control center and deterministic routing/control
plane. At the routing layer, it scores an incoming prompt, selects an engine
category, and emits a receipt that explains the decision. Around that layer, the
proxy/settings/TUI surfaces help users discover models, review
recommendations, plan explicit downloads, control configured local runtimes,
expose a local OpenAI-compatible endpoint, and inspect telemetry.

The core routing decision path does not execute prompts, call model providers,
perform web research, run code, send messages, delete files, purchase anything,
or dispatch to an agent. It only decides.

That makes ModelRouter a control layer for host agents, not a replacement for
them. A Fusion-like multi-agent harness could call ModelRouter for model and
provider policy, route receipts, telemetry, and safety gates, but the host
harness owns task execution, persistent context, worker delegation, monitoring,
and final review.

ModelRouter can be the single local control surface for common local-model
workflows, but it is runtime-neutral. It should work alongside or above LM
Studio, Ollama, LocalAI, llama.cpp servers, MLX/MLX-LM, vLLM, generic
OpenAI-compatible backends, and hosted providers. It should not build a custom
inference engine from scratch when proven runtimes can be wrapped, supervised,
or integrated through explicit adapters.

## Product Truth

ModelRouter's current product north star is a local AI control center and proxy
routing plane, not a chat UI or agent workspace. The admin surface should make
model discovery, recommendations, explicit downloads, routing policy, runtime
status, safety gates, route receipts, telemetry, model/runtime controls, and
wrong-route feedback visible while preserving privacy-safe defaults.

The product should stay transparent about that boundary: it can make routing
decisions observable and enforce provider/safety policy for external agents, but
it should not hide a planner-worker system inside the proxy.

Users should be able to choose one integrated control center without lock-in:
ModelRouter can handle the common discover/recommend/download/runtime/proxy
loop itself, or sit above existing local-model apps and provider gateways.
The product ownership model is documented in `docs/product-boundaries.md`.

See `docs/product-north-star.md` for the canonical north-star screenshot and
the implemented/in-progress/future split.

The current settings dashboard is data-backed where implemented: latest route
receipt, routing map, provider/runtime panel, recent telemetry, benchmark
status, and wrong-route feedback all come from local config and telemetry files
rather than a demo chat surface.

Feature maturity is explicit in `model-router doctor`, `model-router settings`,
and `model-router tui`. Use `docs/release-checklist.md` for release gates and
`docs/upgrade-uninstall.md` for upgrade, rollback, uninstall, and config
migration notes.

## Architecture

The router is implemented under `hermes/plugins/model_router/`. This is a
legacy Python package namespace from the original Hermes Router package, not a
host-application plugin registration point, host manifest, or automatic runtime
integration.

- `models.py` defines JSON-safe dataclass models for engines, prompt features,
  scoring config, alternatives, decisions, receipts, and config.
- `config.py` loads and validates the engine catalog from
  `configs/model_router.yaml`.
- `availability.py` validates declared engine availability without executing
  commands or calling providers.
- `scorer.py` performs fast deterministic prompt analysis with weighted signals,
  saturation, regular expressions, and length heuristics.
- `policy.py` exposes the initialized `ModelRouter` runtime and maps scores and
  features to engine categories.
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

Scoring uses deterministic weighted signals. Public scores stay on a 0-100
scale. The optional top-level `scoring:` YAML section can override feature
weights and `saturation_k`; missing values use router defaults. Invalid scoring
config is treated as invalid config, so compatibility routing fails closed to
`human_confirm`.

```yaml
scoring:
  saturation_k: 50
  weights:
    complexity:
      multi_step_reasoning: 25
      architecture: 25
      coding_intent: 30
    risk:
      destructive_action: 100
      file_shell_github: 30
```

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
Routing hints can add constraints without executing anything. Supported hints
include forced engine preference, attachment modalities, maximum cost tier,
maximum latency tier, and latency sensitivity. High-risk actions still route to
confirmation even if a weaker engine is forced.

Human confirmation is configured separately from scoring. It defaults on:

```yaml
safety:
  require_human_confirmation: true
  confirmation_overrides:
    allow_destructive_actions: false
    allow_send_actions: false
    allow_purchase_actions: false
    allow_high_impact_external_actions: false
    allow_ambiguous_high_impact: false
```

Set only the narrow override needed for the embedding application. Disabling
`require_human_confirmation` removes action-based confirmation routing for valid
configs, but fail-closed routing still uses `human_confirm` for invalid config,
undefined routes, unavailable engines without compatible fallbacks, and fallback
cycles.

For embedded use, initialize the router once and reuse it:

```python
from model_router import ModelRouter

router = ModelRouter.from_config("configs/model_router.local.yaml")
engine = router.route_fast("fix the repo and run tests")
decision = router.route("fix the repo and run tests")
```

`ModelRouter` loads YAML, validates static config, and caches declared
availability once. Per-prompt routing uses the in-memory config and does not run
setup scans or parse YAML again.

Use `route_fast(...)` as the production API when the runtime only needs an
engine choice. It returns the selected engine string through a precompiled
in-memory path and still keeps the hard safety rule that high-risk actions route
to `human_confirm`. Use `route(...)` as the diagnostic and audit API when
callers need scores, reasons, rejected engines, alternatives, or receipt
serialization. For a lighter diagnostic decision, pass
`include_alternatives=False` to skip candidate ranking on that call.

The hot path does not perform built-in logging. Services that need telemetry
should measure and emit metrics around router calls at the service boundary so
`route_fast(...)` stays allocation-light and privacy policy remains explicit.
The optional proxy can write privacy-safe JSONL routing events for hindsight
testing; raw prompts are opt-in. See `docs/production-readiness.md` for SLOs,
benchmark guardrails, and replay workflow.

Optional advanced routing is intentionally data-gated. Milestone 7 reviewed the
checked-in replay, golden, and parity evidence and deferred a second-pass
classifier because there were no labeled unresolved mismatches. See
`docs/advanced-routing.md` for the decision record and acceptance criteria for
revisiting classifier-based routing.

The installed package exposes `model-router` as the generic console command and
`hermes-router` as a backward-compatible alias for diagnostics and scripts.
Host-app integrations should call the Python API directly or implement the host
application's actual plugin contract.

## Dry-Run Dispatch Plans

The router can produce a dispatch plan without executing adapters:

```bash
model-router dispatch-plan "rewrite this text"
model-router dispatch-plan --json "fix the repo and run tests"
model-router dispatch-plan --include-alternatives --json "rewrite this text"
```

Dispatch plans name the selected provider, model, and adapter and include the
routing receipt. They never call providers, load local model weights, run tools,
or perform external actions. Dispatch plans skip ranked alternatives by default
for speed; pass `--include-alternatives` in the CLI or
`include_alternatives=True` in Python when a full alternatives list is useful.
The adapter boundary and lazy-loading policy are documented in
`docs/adapter-contract.md`.

## Model Catalog And User Setup

The model catalog lives at `configs/model_router.yaml`. It defines engine
categories and semantic routing targets rather than hardcoding provider model
names throughout the code.

For machine-specific setup, use one of these paths:

1. Copy and edit `configs/model_router.local.example.yaml`.
2. Run `setup scan` and `setup recommend` to inspect local models and commands.
3. Run `setup wizard` if you want an interactive review before writing a config.
4. Run `setup write` to generate a local YAML file from those recommendations.

The generated local config can be passed explicitly:

```bash
model-router decide \
  --config configs/model_router.local.yaml \
  "fix the repo and run tests"
```

If your shell does not provide `python`, use `python3` or run through `uv`, for
example:

```bash
uv run --python 3.11 --with-editable . model-router setup wizard
```

### Setup Assistant

The setup assistant is intentionally safe and local-first. It scans:

- Known local model directories such as project `models/`, Hugging Face cache,
  Ollama, modern LM Studio storage at `~/.lmstudio/models`, legacy LM Studio
  storage, `~/models`, and Downloads.
- Optional `--model-dir` paths supplied by the user.
- Command availability for tools such as `claude`, `codex`, `hf`, `ollama`,
  `llama-server`, and `lmstudio`.
- Environment-variable presence for API keys such as `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, and `HF_TOKEN`. Values are never printed.

It does not execute model providers, call external APIs, download files, or
modify the default catalog during `scan`, `recommend`, or `write`. The wizard
can run `hf download` only for recommended models you selected, and only after
it asks for a separate download confirmation. Downloads can also be run through
the explicit `setup download --execute` path.

Installer plan:

```bash
model-router install --quick
model-router install --auto --json
model-router install --local-only --mlx-lm
```

`model-router install` is deterministic onboarding, not a background daemon or
silent setup script. It detects the install method, package version, Python,
command availability, optional dependencies, existing config files, port status,
and local runtime signals. It prints the next explicit commands for prereqs,
`init`, `doctor`, settings, proxy startup, downloads, telemetry, and TUI when
available. It does not download models, install services, enable hosted
providers, overwrite configs, or start runtimes.

Terminal control center:

```bash
python -m pip install "hermes-router[tui]"
model-router tui --config-dir ~/.model-router
```

`model-router tui` is an optional local terminal control center backed by the
same shared admin state as `model-router settings`. V1 is read-only and exposes
Status, Models, Routing, Runtimes, Telemetry, Logs, and Settings tabs. It does
not provide a chat prompt, does not show raw prompts by default, and does not
perform writes; mutating shared actions remain explicit and require
`confirm=true` in the shared action layer.

Scan:

```bash
model-router setup scan
model-router setup scan --json
```

Recommend:

```bash
model-router setup recommend
model-router setup recommend --json
```

Recommendations come from the packaged setup-time model advisor catalog at
`hermes/plugins/model_router/data/model_catalog.yaml`. The advisor uses local
hardware signals such as RAM, CPU architecture, CPU core count, Apple Silicon,
accelerator backend hints, and free disk space to rank Hugging Face candidates
for each route. RAM is the fit/load gate; CPU/GPU/accelerator backend, runtime
format, quantization, and measured benchmark results drive whether a model is
expected to feel usable. This catalog is not loaded by `ModelRouter`, and
hardware detection never runs inside `route_fast(...)` or `route(...)`.

Recommendation JSON includes `fit_score`, `runtime_match_score`,
`expected_speed_score`, `quality_role_score`, `setup_friction_score`,
`benchmark_score`, `overall_score`, labels such as `recommended`,
`fits_but_likely_slow`, `too_large`, `needs_runtime`, and
`benchmark_recommended`, plus human-readable reasons and warnings.

The model-library foundation now has a durable local registry concept in
`hermes.plugins.model_router.model_registry`. The registry records known models
from the router engine catalog, proxy backend config, local setup discovery,
explicit user-declared models, and caller-supplied runtime discovery results.
Each `KnownModel` record is JSON-safe and can carry provider, runtime, model id,
source, local path, format, context length, quantization, size, license,
install/download state, health/load state, tags, capabilities, routing
eligibility, backend, and assigned routes.

This registry is a control-center and reporting primitive, not a routing hot
path dependency. It does not fetch models, call runtime APIs by itself, download
files, mutate config, or change `route_fast(...)` decisions. Runtime adapters or
host/admin surfaces may pass already-discovered model ids into the registry,
but discovery remains explicit and outside the routing decision path. That lets
ModelRouter move toward LM Studio-level model-library ownership while staying
provider-neutral and compatible with LM Studio, Ollama, LocalAI, llama.cpp,
MLX/MLX-LM, vLLM, hosted providers, and generic OpenAI-compatible runtimes.

Runtime management remains adapter-based. The
`hermes.plugins.model_router.runtime_adapters` module defines the optional
control surface each backend can expose: runtime detection, endpoint URL,
health, model listing, loaded-model listing, start/stop support, load/unload
support, log metadata, and disabled reasons for unsupported operations. Generic
OpenAI-compatible, LM Studio, Ollama, and managed-runtime adapters report their
capabilities honestly. Hosted backends are not probed by default, and
externally managed local runtimes report that their lifecycle is outside
ModelRouter unless a configured managed process owns it. Settings/admin
surfaces may use bounded adapter calls to populate runtime status and
runtime-discovered models; `route_fast(...)`, `route(...)`, and proxy forwarding
do not require runtime adapters to decide a route.

Optional local backend benchmarks are explicit:

```bash
model-router setup benchmark --config ~/.model-router/routing_proxy.yaml
model-router setup benchmark --config ~/.model-router/routing_proxy.yaml --execute --yes
```

The command benchmarks configured backends with a fixed synthetic prompt and
stores metrics in `~/.model-router/benchmarks.json`. It stores no prompt bodies,
request bodies, API keys, or secrets, and benchmark-driven recommendations never
silently mutate config.

Interactive wizard:

```bash
model-router setup wizard \
  --output configs/model_router.local.yaml
```

The wizard is a guided configurator. It asks whether you want local LLMs,
API-backed engines, or a mixed setup, then walks each main routing category:
simple, balanced, reasoning, coding, research, vision, and image generation.
For each route, it shows numbered local models discovered on your machine and
numbered hardware-aware recommended downloads for missing local roles. You can
type a number, accept the default engine, or type another known engine name such as
`claude_code`, `codex`, `openai_api`, `anthropic_api`, `balanced_local`, or
`reasoning_local`. It still asks for final confirmation before writing the local
YAML file. If you selected recommended downloads, it then asks whether to run
those `hf download` commands into the configured local model folders.

When recommended downloads are possible but the Hugging Face `hf` CLI is missing,
the wizard prompts at the beginning and can install it into the Python
environment running the setup assistant. This prerequisite prompt is separate from the
later model-download confirmation.

Write a generated config:

```bash
model-router setup write \
  --output configs/model_router.local.yaml
```

The writer will not overwrite an existing file unless `--force` is passed.

### Hugging Face Download Plans

Setup recommendations include `hf download` commands for missing local roles,
and `setup download` shows the same plan without running it:

```bash
model-router setup download
model-router setup download --route fast_local
```

Execution is a separate opt-in:

```bash
model-router setup download \
  --route fast_local \
  --execute
```

The command asks for confirmation before running `hf download`. For
non-interactive scripts, pass `--yes`. This keeps large downloads, gated
licenses, and hardware choices under user control.

Users can also provide their own Hugging Face repo id:

```bash
model-router setup download \
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
model-router validate-config
model-router validate-config --json
```

Routing uses the same safe validation. If a selected engine is unavailable, the
router follows its fallback chain. If no available fallback exists, the decision
fails closed to `human_confirm` and includes availability reasons in the
receipt. Engines can also be rejected when they lack required tool support,
required modalities, or exceed requested cost/latency tiers.

Each engine supports:

```yaml
provider: local
model: modelrouter-balanced-local
adapter: local_chat
strengths:
  - summarization
max_context: 16384
cost_tier: free
latency_tier: medium
capability: 65
trust: 60
cost: 0
latency: 45
supports_tools: false
modalities: []
enabled: true
fallback: reasoning_local
availability:
  status: auto
```

The numeric ranking fields are optional 0-100 values. If omitted, ModelRouter derives
them from the tier fields and context window. They rank compatible alternatives;
they do not override the configured route target when that target is enabled,
available, and compatible.

Example scoring override:

```yaml
scoring:
  saturation_k: 50
  weights:
    complexity:
      multi_step_reasoning: 25
      architecture: 25
    risk:
      destructive_action: 100
      sensitive_domain: 25
    confidence:
      ambiguous: 25
```

Example safety override:

```yaml
safety:
  require_human_confirmation: true
  confirmation_overrides:
    allow_send_actions: true
```

The router does not learn from prior confirmations at runtime. Any relaxation is
visible in YAML and should be paired with tests for the embedding application.

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
model-router decide "rewrite this text"
```

JSON receipt output:

```bash
model-router decide --json "fix the repo and run tests"
```

Custom catalog:

```bash
model-router decide --config configs/model_router.yaml "research current GLP-1 supplement trends"
```

Routing hints:

```bash
model-router decide \
  --attachment image \
  --force-engine multimodal_vision \
  --max-cost-tier medium \
  --max-latency-tier medium \
  "summarize this attachment"
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
  "config_valid": true,
  "requirements": {
    "needs_tools": true,
    "required_modalities": [],
    "max_cost_tier": null,
    "max_latency_tier": null
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
  "fallback_used": false
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

## Optional Proxy And Future Adapters

The optional `model-router-proxy` command is the supported runtime boundary for
OpenAI-compatible clients. It exposes local `/v1/chat/completions`,
`/v1/responses`, `/v1/embeddings`, `/v1/completions`, and `/v1/models`
endpoints. In decision mode it routes supported request text through initialized
`route_fast(...)`, maps the selected engine to a configured upstream backend,
and forwards the request to that OpenAI-compatible server. In manual mode it
forwards without prompt classification to the configured default backend/model.
It remains outside the router hot path and is installed only with the `proxy`
extra.

For `/v1/responses`, ModelRouter extracts routing text from the `input` field
and preserves the common Responses API request shape when forwarding, including
instructions, tools, metadata, previous response ids, streaming, and fallback
behavior.

For `/v1/embeddings`, ModelRouter extracts routing text only from string inputs
or bounded string arrays, preserves the request body, and does not stream. For
legacy `/v1/completions`, it routes from the `prompt` string or bounded string
array and preserves streaming when the upstream supports it. `/v1/models` lists
proxy aliases plus configured backend models with capability hints. `/v1/messages`
returns a shaped `unsupported_endpoint` response; Anthropic Messages
compatibility is planned but not silently bridged.

Managed local runtimes are an optional proxy feature, not part of
`route_fast(...)`. A backend may declare an explicit argv-only runtime command,
readiness URL, idle timeout, shutdown timeout, and log path. The proxy starts
that child process on the first routed request that needs the backend, keeps it
warm, stops it after the idle timeout, and stops all managed children on proxy
shutdown. Starting the process loads the model; stopping the process unloads it
from memory. The proxy never downloads models automatically and never infers
commands beyond what is configured in YAML.

For LM Studio:

```bash
model-router init --auto --yes
```

`--auto` checks local first-run signals and chooses Ollama when Ollama is
installed/reachable, LM Studio when an LM Studio-style local server is
reachable, and LM Studio as the conservative fallback. Use an explicit preset
when you already know the target:

```bash
model-router init --preset lmstudio --yes
model-router-proxy --config ~/.model-router/routing_proxy.yaml
```

Start the LM Studio local server on `http://127.0.0.1:1234/v1`, then edit the
generated backend `model:` values to match the exact model ids LM Studio
advertises.

For Ollama:

```bash
ollama pull qwen3:0.6b
ollama pull qwen3:4b
ollama pull qwen3:14b
ollama pull qwen2.5-coder:7b
model-router init --preset ollama --yes
model-router-proxy --config ~/.model-router/routing_proxy.yaml
```

The Ollama preset targets `http://127.0.0.1:11434/v1`; change generated
`model:` values if you prefer different local models.

When Ollama is selected and expected models are missing, first-run output shows
the exact `ollama pull ...` commands. When LM Studio is selected, first-run
output reminds you to edit generated backend model ids to match the exact ids
advertised by the LM Studio local server.

For MLX-LM managed runtimes:

```bash
python -m pip install mlx-lm
model-router init --preset mlx-lm --yes
model-router doctor --config ~/.model-router/routing_proxy.yaml
model-router-proxy --config ~/.model-router/routing_proxy.yaml
```

Replace every generated `REPLACE_WITH_MLX_*` placeholder with an exact
MLX/Hugging Face repo id or local model path before dogfooding. The preset uses
one `mlx_lm.server` process per route on ports `8090`, `8091`, `8093`, and
`8094`. MLX-LM support is chat/models-first: `/v1/chat/completions` can be
forwarded to MLX-LM, `/v1/models` is used for readiness/model checks, and
`/v1/responses` requires an upstream that supports the Responses API.

For llama.cpp managed runtimes, start from the `llamacpp` preset and add runtime
blocks to the backends you want the proxy to own:

```yaml
runtime:
  enabled: true
  kind: llama-server
  command:
    - llama-server
    - "-m"
    - /Users/you/models/model.gguf
    - --port
    - "8090"
  readiness_url: http://127.0.0.1:8090/v1/models
  readiness_timeout_seconds: 30
  idle_timeout_seconds: 900
  shutdown_timeout_seconds: 5
  log_path: ~/.model-router/logs/llama-fast.log
```

`model-router doctor` reports whether managed runtimes are enabled, whether a
runtime command is missing, whether a readiness URL is down, whether placeholders
remain, and whether a readiness port appears occupied by a conflicting process.

Use these values in an OpenAI-compatible agent or SDK:

```text
Base URL: http://127.0.0.1:8082/v1
Model: model-router
API key: leave blank unless proxy auth is configured
```

Runtime adapter work stays separate from scoring policy. The current adapter
foundation reports local health, visible models, loaded-model placeholders,
capabilities, disabled action reasons, and log metadata for admin surfaces. If a
later runtime adapter executes non-chat actions or talks to host-specific APIs,
it must remain behind explicit adapter contracts and confirmation gates for
risky actions, preserve receipt emission, and keep decision logic testable
without provider calls.

## Feedback To Regression Workflow

When a real request routes to the wrong engine, prefer this path:

1. Enable proxy observability for the calibration run.
2. If replay is needed, temporarily set `prompt_capture: full`; otherwise keep
   the default redacted preview.
3. Label the wrong route:

```bash
model-router feedback req-123 balanced_local \
  --notes "summary prompt escalated to reasoning"
```

4. Replay labeled events before changing scoring:

```bash
python scripts/replay_routing_log.py \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl \
  --json
```

5. Add the prompt to a fixture or parametrized test, update deterministic rules,
   rerun replay, and keep the test with the fix.

This keeps the route-quality loop deterministic and auditable while preserving
the `route_fast(...)` performance contract.

Use `model-router telemetry summary` and `model-router telemetry feedback` while
dogfooding to track replayable events, unlabeled request ids, skipped private
events, and mismatch groups without printing prompt text. See
`docs/telemetry-dogfood.md` for the full workflow and data threshold for
revisiting optional advanced routing.
