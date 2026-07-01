# ModelRouter

ModelRouter is the open switchboard for AI model routing: one local
OpenAI-compatible endpoint that routes each request to the right model, with
receipts, safety gates, and full provider control.

Simple work can go to fast local models, complex work to stronger reasoning
models, fresh research to research tools, repo work to code-capable backends,
screenshots to vision/OCR, image requests to image generation, and risky
actions to human confirmation.

ModelRouter is the routing/control layer, not the agent harness. A
Fusion-like multi-agent system can use it for model and provider policy,
receipts, telemetry, and safety gates, while the host agent remains responsible
for task execution, context management, delegation, and final review.

## Use With Your Agent In 3 Minutes

Install the proxy extra:

```bash
pip install "hermes-router[proxy]"
```

Plan deterministic onboarding:

```bash
model-router install --quick
model-router install --quick --json
```

The installer command is plan-only in v1. It detects package/install method,
Python, command availability, optional proxy/runtime dependencies, config files,
ports, and local runtime signals, then prints the next explicit commands. It
does not download models, install services, enable hosted providers, overwrite
configs, or start runtimes.

Create first-run configs:

```bash
model-router init --auto --yes
```

Start the local routing proxy:

```bash
model-router-proxy --config ~/.model-router/routing_proxy.yaml
```

Point any OpenAI-compatible agent/client at:

```text
http://127.0.0.1:8082/v1
```

Useful follow-ups:

```bash
model-router validate-proxy-config --config ~/.model-router/routing_proxy.yaml
model-router doctor --config ~/.model-router/routing_proxy.yaml
curl http://127.0.0.1:8082/health
```

The normal onboarding loop is:

1. Choose a routing profile: `fast`, `balanced`, `quality`, `private`, or
   `safe`.
2. Choose provider policy: local-only, hosted-allowed, allowlists, denylists,
   or per-route pools.
3. Inspect receipts with `model-router decide --explain` or proxy telemetry
   before changing routing rules.

Prefer a small local settings screen instead of editing YAML first:

```bash
model-router settings --config-dir ~/.model-router
```

This opens a localhost-only admin UI, defaulting to
`http://127.0.0.1:8099`. It is for configuration and operations, not chat: pick
presets, scan models, edit backend model ids/ports/runtime commands, toggle
observability, run doctor, start/stop/restart the proxy, inspect telemetry, and
label wrong routes. The Models section shows installed local models, curated
discover candidates, hardware-aware recommendations, download plans, and
route-to-model assignments.

The current product north star is documented in
[Product north star](docs/product-north-star.md): ModelRouter should feel like a
local proxy control center with routing maps, runtime status, receipts, safety
gates, telemetry, and feedback labeling, while remaining explicitly not a chat
UI or agent workspace.

Prefer a terminal control center instead of a browser:

```bash
python -m pip install "hermes-router[tui]"
model-router tui --config-dir ~/.model-router
```

The TUI uses the same shared admin state as `model-router settings`. The first
version is read-only: Status, Models, Routing, Runtimes, Telemetry, Logs, and
Settings tabs show real config/runtime/telemetry state, while mutating actions
remain confirmation-gated in the shared action layer.

Before tagging a release, use the maturity and dogfood gates in
[Release checklist](docs/release-checklist.md). Upgrade, rollback, and
uninstall notes live in [Upgrade and uninstall](docs/upgrade-uninstall.md).

If you are testing from a local checkout or setting up managed runtimes, install
prerequisites into the active Python environment:

```bash
model-router setup install-prereqs --preset mlx-lm --execute --yes
```

Use `--preset proxy`, `--preset mlx-lm`, `--preset llamacpp`, or `--preset all`.
The command runs pip through the current Python executable, so inside the repo
venv it installs into `.venv` rather than Homebrew's externally managed Python.

The router core is intentionally a decision router only. It does not execute
prompts, browse the web, run shell commands, send messages, delete files,
purchase anything, spawn workers, or review delegated work. The optional proxy
forwards OpenAI-compatible requests to configured upstreams; when explicitly
configured with managed local runtimes, it can start and stop only those
configured local model-server processes.

## At a Glance

| Need | ModelRouter provides |
| --- | --- |
| Fast hot-path routing | `ModelRouter.route_fast(prompt)` returns an engine string |
| Productized receipts | `ModelRouter.route(prompt)` returns summary, reason codes, policy/fallback/safety explanations, and audit fields |
| Plain-language profiles | `fast`, `balanced`, `quality`, `private`, and `safe` compile to routing constraints |
| Provider policy controls | Versioned allowlists, denylists, local-only mode, and backend pools |
| CLI tooling | `decide`, `workflow-benchmark`, `validate-config`, `dispatch-plan`, and `setup` commands |
| Local/API flexibility | YAML routing targets for local models, hosted APIs, vision, image generation, and custom adapters |
| Safety boundaries | High-risk or invalid requests fail closed to `human_confirm` |
| Setup help | Safe local scans, config recommendations, and opt-in Hugging Face download plans |

## Highlights

- Deterministic heuristic routing with no LLM classification call.
- Fast initialized hot path: `router.route_fast(prompt)` returns only the
  selected engine.
- Rich receipt path: `router.route(prompt)` returns scores, reasons, rejected
  engines, alternatives, requirements, reason codes, and concise explanations.
- Routing profiles for low-latency, balanced, quality, private, and safer
  routing defaults.
- Versioned provider/backend policy controls for local-only operation,
  allowlists, denylists, tier caps, and receipt-visible rejections.
- Offline workflow benchmarks for release-friendly routing correctness reports.
- YAML-driven engine catalog; model names are not hardcoded throughout the
  router.
- OpenAI-compatible proxy for agents that only know how to call a local AI
  endpoint.
- First-run `model-router init` for local proxy configs.
- Opt-in managed local runtimes for `llama-server` and `mlx_lm.server`.
- Optional terminal control center with `model-router tui`.
- User-configurable routing targets for local models, hosted APIs, web/RAG
  tools, vision, image generation, or custom adapters.
- Fail-closed safety: missing/invalid config and high-risk actions route to
  `human_confirm`.
- Declarative availability checks for env vars, commands, and local paths.
- Setup assistant for local/API/mixed model configuration and optional Hugging
  Face download plans.

## Project Status

ModelRouter is a lean production-ready decision layer when embedded through
the initialized Python API. The stable surface today is:

- `ModelRouter.route_fast(...)` for production routing.
- `ModelRouter.route(...)` for diagnostic and audit receipts.
- Config-driven model/agent catalog.
- Safe dry-run dispatch plans.
- Local setup wizard and recommendations.

The local proxy is the main product path for agents. Direct dispatch beyond
OpenAI-compatible chat forwarding remains intentionally behind explicit adapter
boundaries and confirmation gates.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/doncazper/model-router.git
cd model-router
python -m pip install -e ".[dev]"
```

For normal use from PyPI:

```bash
pip install "hermes-router[proxy]"
```

ModelRouter began as Hermes Router and was renamed after evolving into a
generic OpenAI-compatible routing proxy for local/custom agents. The PyPI
distribution name remains `hermes-router` for compatibility because
`model-router` is already occupied on PyPI. The primary command and Python API
are `model-router`, `model-router-proxy`, and `import model_router`.

If your shell does not provide `python`, use `python3`. If your system Python is
older, use `uv`:

```bash
uv run --python 3.11 --with pytest --with PyYAML python -m pytest
```

## Quick Start

Readable CLI output:

```bash
model-router decide "rewrite this text"
```

JSON receipt:

```bash
model-router decide --json "fix the repo and run tests"
```

Readable explanation:

```bash
model-router decide --explain "fix the repo and run tests"
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

## Routing Profiles

Profiles are plain-language routing modes that compile into routing hints and
policy constraints. They are separate from engine names, so advanced users can
still point routes or `--force-engine` at exact configured engines.

| Profile | Behavior |
| --- | --- |
| `fast` | Prefer low latency and low-cost backends. |
| `balanced` | Current default deterministic behavior. |
| `quality` | Allow stronger configured reasoning and hosted fallbacks. |
| `private` | Local-only provider policy; hosted API providers are rejected and shown in receipts. |
| `safe` | Stricter confirmation behavior for ambiguous or sensitive requests. |

Use a profile from the CLI:

```bash
model-router decide --profile private --json "research this"
model-router dispatch-plan --profile safe "handle my taxes"
```

Set the proxy default profile in `~/.model-router/routing_proxy.yaml`:

```yaml
proxy:
  host: 127.0.0.1
  port: 8082
  routing_profile: balanced
  routing_mode: decision
```

The settings UI also exposes the default proxy profile. Profiles never bypass
`human_confirm`, and `private` does not silently enable or call hosted providers.

The proxy can also run with the decision layer off:

```yaml
proxy:
  routing_mode: manual
  default_backend: fast
  default_model: local-fast-model
  respect_client_model: false
  unknown_model_behavior: fallback_to_default
```

`decision` is the default and preserves the current prompt-classifying router.
`manual` does not call `route_fast(...)` or inspect prompts for routing; it
forwards to `proxy.default_backend` and uses `proxy.default_model` unless
`respect_client_model` is enabled and the inbound model matches the manual
default model or selected backend model. Deferred modes such as `model_map` and
`passthrough` fail validation until they are implemented.

## Provider And Backend Policies

Provider policy is a versioned router constraint layer that applies after the
selected profile and before engine fallback resolution. It lets you set global
provider rules once, then optionally narrow them per semantic route.

```yaml
provider_policy:
  version: 1
  provider_allowlist: []
  provider_denylist: []
  local_only: false
  hosted_allowed: true
  max_cost_tier: null
  max_latency_tier: null
  route_pools:
    simple:
      local_only: true
    balanced:
      local_only: true
    reasoning:
      hosted_allowed: true
```

- `provider_allowlist` and `provider_denylist` constrain engine providers such
  as `local`, `openai`, `anthropic`, or `human`.
- `local_only: true` and `hosted_allowed: false` compile to local/human-only
  routing. `human_confirm` stays reachable even under restrictive policies.
- `max_cost_tier` and `max_latency_tier` reuse the existing engine tier values.
- Caller hints may narrow provider policy, but they cannot loosen configured
  allowlists, denylists, local-only mode, or max tier caps.
- Per-route `route_pools` apply to targets such as `simple`, `balanced`,
  `reasoning`, `coding`, `research`, `vision`, `image_generation`, and
  `confirmation`.
- Receipts include policy reasons and rejected engines so denied providers are
  visible instead of silently skipped.

Proxy backend policy is separate because backend names exist only in the local
OpenAI-compatible proxy config:

```yaml
backend_policy:
  version: 1
  backend_allowlist: []
  backend_denylist: []
```

Backend policy is enforced before forwarding and on explicit fallback chains.
If a selected engine maps to a denied backend, the proxy returns
`backend_policy_rejected` without calling upstream.

## Explicit Verification Boundary

The optional proxy verifier is disabled by default and stays outside
`route_fast(...)`. When enabled, it runs only after the proxy has routed and
forwarded a non-streaming request:

```text
route -> forward to selected backend -> optional verifier -> response
```

```yaml
verifier:
  version: 1
  mode: "off"                 # off, receipt-only, sampled, always-for-risky-output
  backend: null               # required for sampled or always-for-risky-output
  sample_rate: 0.0
  route_codes: []
  timeout_seconds: 10
  failure_behavior: log_only  # log_only or fail_closed
  include_response_preview: false
  max_response_preview_chars: 500
```

`receipt-only` logs whether a request qualifies without calling another
backend. `sampled` verifies a deterministic percentage of low-risk
non-confirmation requests. `always-for-risky-output` verifies configured route
codes but still never bypasses `human_confirm`. Streaming requests are marked
`skipped_streaming`; the proxy does not buffer streams for verification.

## Workflow Benchmarks

Run offline workflow benchmarks when you want release evidence for routing
correctness and profile behavior:

```bash
model-router workflow-benchmark
model-router workflow-benchmark --json --fail-on-mismatch
```

The benchmark uses checked-in sanitized fixture prompts for simple,
balanced, coding, research, vision, image generation, safety, private-profile,
quality-profile, and sidekick-delegation-shaped workflows. Reports include
expected and selected engines, provider, confirmation state, route-change
counts, receipt summaries, reason codes, delegation suitability signals, and
policy/fallback/safety explanations. Reports serialize prompt hashes instead of
prompt bodies and make no backend, verifier, download, or hosted API calls.

## Route Receipts

Receipts are the transparent answer to "why did this go there?" The JSON shape
keeps the original audit fields and adds deterministic product fields:

- `summary`: concise human-readable route outcome.
- `reason_codes`: stable lowercase codes such as `route.coding`,
  `policy.local_only`, `rejection.provider_denied`,
  `delegation.mechanical_work_likely`, or `fallback.used`.
- `delegation_suitability`: diagnostic task-shape signals for external agents
  considering sidekick-style delegation; ModelRouter still does not delegate.
- `selected_route_explanation`: why the selected engine matched the request.
- `policy_explanation`: profile/provider constraints that mattered.
- `rejection_explanation`: rejected engines/providers and the reason.
- `fallback_explanation`: whether fallback was used or only configured.
- `safety_explanation`: confirmation or fail-closed explanation.
- `privacy_explanation`: local-only/hosted-policy context and prompt privacy.
- `wrong_route_next_action`: how to label the route for dogfooding.

Receipts never include raw prompt text by default. `route_fast(...)` remains the
string-only production hot path; use `route(...)`, `model-router decide --json`,
or `model-router decide --explain` when you need receipt detail.

## Python API

Initialize once and reuse the router. Runtime calls stay in memory and do not
re-read YAML, scan disk, or run setup helpers.

```python
from model_router import ModelRouter

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
from model_router import route_prompt

decision = route_prompt("research current GLP-1 supplement trends")
```

The historical `hermes.plugins.model_router` import path remains available for
backward compatibility, but new custom-agent integrations should use
`model_router`.

## CLI

After installation, you can use the console command:

```bash
model-router decide "rewrite this text"
model-router decide --json "fix the repo and run tests"
```

The old `hermes-router` command remains as a compatibility alias for existing
scripts.

Use a custom catalog:

```bash
model-router decide \
  --config configs/model_router.local.yaml \
  "research current GLP-1 supplement trends"
```

Pass routing hints:

```bash
model-router decide \
  --profile private \
  --attachment image \
  --force-engine multimodal_vision \
  --max-cost-tier medium \
  --max-latency-tier medium \
  "summarize this attachment"
```

Validate a config:

```bash
model-router validate-config
model-router validate-config --json
```

Create a dry-run dispatch plan:

```bash
model-router dispatch-plan "fix the repo and run tests"
model-router dispatch-plan --json "rewrite this text"
model-router dispatch-plan --include-alternatives --json "rewrite this text"
```

Dispatch plans only describe what a future adapter would do. They do not execute
models, tools, shell commands, provider calls, or external actions. They skip
ranked alternatives by default for speed; pass `--include-alternatives` when a
full receipt is useful.

Run offline workflow correctness benchmarks:

```bash
model-router workflow-benchmark
model-router workflow-benchmark --json --fail-on-mismatch
```

This command exercises sanitized route fixtures and emits prompt hashes, not
prompt bodies. It does not call providers or local model servers.

Inspect packaged catalog updates without changing local policy:

```bash
model-router catalog status
model-router catalog diff
model-router catalog apply --yes
```

`catalog status` and `catalog diff` are read-only and perform no remote checks.
`catalog apply` writes the packaged router catalog only after explicit
confirmation, backs up an existing local config first, and appends a JSONL
migration entry.

## Local Routing Proxy

Most agents can talk to an OpenAI-compatible local endpoint. Install the optional
proxy extra to expose one local endpoint that routes common OpenAI-compatible
requests to the configured upstream model server:

```bash
model-router init --preset lmstudio --yes
model-router-proxy --config ~/.model-router/routing_proxy.yaml
```

Then point the agent at:

```text
http://127.0.0.1:8082/v1
```

The proxy supports `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`,
`/v1/completions`, `/v1/models`, and `/health`. It calls initialized
`route_fast(...)` once per decision-mode routed request, maps the selected
engine to a configured backend, overrides the outgoing backend model, and
forwards to an OpenAI-compatible upstream such as LM Studio, llama.cpp server,
LocalAI, or a frontier gateway. `human_confirm` returns HTTP `409` and is never
forwarded. Tools are preserved by default and can be stripped per backend for
small local models.

`/v1/models` returns ModelRouter proxy aliases and configured backend models
with capability hints for chat completions, Responses, embeddings, completions,
models, and planned Messages support. `/v1/messages` is intentionally returned
as a shaped `unsupported_endpoint` error until Anthropic Messages compatibility
has real capability plumbing.

The configured `proxy.routing_profile` is applied to every decision-mode proxy
request and is reported in `/health` plus the `X-ModelRouter-Profile` response
header. Routing mode is reported in `/health`, telemetry, and response headers
such as `X-ModelRouter-Mode`, `X-ModelRouter-Decision-Layer`,
`X-ModelRouter-Backend`, and `X-ModelRouter-Model`.

Managed local runtimes are opt-in per backend. A backend can declare an argv-only
`runtime.command`, readiness URL, idle timeout, shutdown timeout, and log path.
The proxy starts that configured process on the first route that needs it, keeps
it warm, stops it after the idle timeout, and stops all managed child processes
on proxy shutdown. Model loading/unloading is process-level: starting
`llama-server` or `mlx_lm.server` loads the model, and stopping that process
unloads it from memory. ModelRouter does not download models automatically and
does not infer or execute arbitrary commands.

Packaged presets:

```bash
model-router init --preset lmstudio --yes
model-router init --auto --yes
model-router init --preset ollama --yes
model-router init --preset llamacpp --yes
model-router init --preset mlx-lm --yes
model-router init --preset localai --yes
model-router init --preset hosted-openai-compatible --yes
```

Use `--auto` when you want ModelRouter to choose from local first-run signals.
It checks whether Ollama is installed, whether Ollama is reachable at
`http://127.0.0.1:11434/v1`, and whether an LM Studio-style server is reachable
at `http://127.0.0.1:1234/v1`. The generated config still uses privacy-safe
telemetry defaults.

### Local Admin Settings UI

Run a small server-rendered admin UI when you want visual controls for the
proxy without turning ModelRouter into a chat app:

```bash
model-router settings --config-dir ~/.model-router
```

Use `--no-open` when running in a terminal-only session:

```bash
model-router settings --config-dir ~/.model-router --no-open
```

The settings UI binds to `127.0.0.1:8099` by default and manages only local
config and child processes you explicitly start. It can:

- Show a data-backed route receipt for the latest actual proxy request, including
  the safe request id needed for wrong-route feedback.
- Render the routing map from the configured `engine_backends`, backend models,
  runtime types, privacy/cost hints, and fallback chains.
- Show provider/runtime rows from real backend config and latest telemetry,
  rather than demo-only sample rows.
- Show scanned local models and recommended Hugging Face downloads.
- Show recommendation score labels, local backend benchmark status, and workflow
  benchmark status.
- Edit `~/.model-router/routing_proxy.yaml` fields for the proxy, observability,
  and per-route backends.
- Show runtime state for managed `llama-server` and `mlx_lm.server` backends.
- Start, stop, and restart `model-router-proxy` as a child process of the
  settings command.
- Run `doctor`, plan/run local backend benchmarks after confirmation, inspect
  recent request telemetry, copy request ids, and write feedback labels in the
  same JSONL format as `model-router feedback`.

Privacy and mutation defaults stay conservative. The UI has no prompt box, no
chat transcript, does not display literal API keys, does not display raw prompt
text, and requires explicit Save/Download/Start/Restart clicks before changing
files or processes.

Productization planning now lives in
[`docs/codex/productization-roadmap.md`](docs/codex/productization-roadmap.md),
with the shared admin state/action contract in
[`docs/codex/admin-state-contract.yaml`](docs/codex/admin-state-contract.yaml).
The key next direction is a shared admin control plane, followed by optional
"decision layer off" routing modes for basic gateway usage.

### Known-Good Local Setups

LM Studio:

1. In LM Studio, download the chat models you want for fast, balanced,
   reasoning, and coding work.
2. Start the LM Studio local server with an OpenAI-compatible endpoint on
   `http://127.0.0.1:1234/v1`.
3. Generate the preset:

```bash
model-router init --preset lmstudio --yes
```

4. Open `~/.model-router/routing_proxy.yaml` and replace
   `lmstudio-fast-model`, `lmstudio-balanced-model`,
   `lmstudio-reasoning-model`, and `lmstudio-code-model` with the exact model
   ids LM Studio lists.

Ollama:

```bash
ollama pull qwen3:0.6b
ollama pull qwen3:4b
ollama pull qwen3:14b
ollama pull qwen2.5-coder:7b
model-router init --preset ollama --yes
```

The Ollama preset targets `http://127.0.0.1:11434/v1` and uses those model ids
by default. If you choose different local models, edit the `model:` values in
`~/.model-router/routing_proxy.yaml`.

MLX-LM managed runtime:

```bash
python -m pip install mlx-lm
model-router init --preset mlx-lm --yes
```

Then edit `~/.model-router/routing_proxy.yaml` and replace every
`REPLACE_WITH_MLX_*` value with an exact MLX/Hugging Face repo id or local model
path. The generated preset uses one `mlx_lm.server` process per route on ports
`8090`, `8091`, `8093`, and `8094`; each process starts on first use and idles
out after 900 seconds. This first pass targets MLX-LM's OpenAI-compatible
`/v1/chat/completions` and `/v1/models` shape. `/v1/responses` requires an
upstream that supports the Responses API; MLX-LM translation is intentionally
deferred.

llama.cpp managed runtime example:

```yaml
runtime:
  enabled: true
  kind: llama-server
  command:
    - llama-server
    - "-m"
    - /Users/you/models/fast.gguf
    - --port
    - "8090"
  readiness_url: http://127.0.0.1:8090/v1/models
  readiness_timeout_seconds: 30
  idle_timeout_seconds: 900
  shutdown_timeout_seconds: 5
  log_path: ~/.model-router/logs/llama-fast.log
```

Use argv lists, not shell strings. Runtime stdout/stderr is captured to the
configured log path, and startup/readiness failures return a safe
`runtime_start_failed` proxy response with route-identification headers.

For managed-runtime presets, `--auto-models` scans local model folders and fills
only route-matched backends. It does not reuse an unrelated model just to remove
placeholders, and it does not download weights:

```bash
model-router init --preset mlx-lm --auto-models --yes
model-router init --preset llamacpp --auto-models --yes
```

Apple Silicon machines can use MLX-LM or GGUF/llama.cpp. Other machines can use
GGUF/llama.cpp, Ollama, LM Studio, or any OpenAI-compatible upstream. Setup
commands prefer already-installed compatible models for config generation, but
still show strong download candidates when a better or more targeted model is
available.

### First-Run Transcript

```text
$ model-router init --auto --yes
Created config directory: /Users/you/.model-router
Created log directory: /Users/you/.model-router/logs
Configuration ready.
Recommended preset: ollama.
Ollama is installed but not reachable; start it with `ollama serve`.
No LM Studio-style server detected at http://127.0.0.1:1234/v1.
Start Ollama before running the proxy: ollama serve
Recommended Ollama model pulls:
- ollama pull qwen3:0.6b
- ollama pull qwen3:4b
- ollama pull qwen3:14b
- ollama pull qwen2.5-coder:7b
Run: model-router-proxy --config /Users/you/.model-router/routing_proxy.yaml
Agent endpoint: http://127.0.0.1:8082/v1
Telemetry: model-router telemetry summary --events /Users/you/.model-router/logs/routing-events.jsonl --feedback /Users/you/.model-router/routing-feedback.jsonl
Written:
- /Users/you/.model-router/model_router.yaml
- /Users/you/.model-router/routing_proxy.yaml
```

If the upstream server is not running yet, `doctor` should fail usefully:

```text
$ model-router doctor --config ~/.model-router/routing_proxy.yaml
Proxy config valid: true
Router config valid: true
Overall ok: false
Backends:
- fast: unreachable (<urlopen error [Errno 61] Connection refused>)
- balanced: unreachable (<urlopen error [Errno 61] Connection refused>)
- reasoning: unreachable (<urlopen error [Errno 61] Connection refused>)
- code: unreachable (<urlopen error [Errno 61] Connection refused>)
Next steps:
- Ollama backend unreachable; start Ollama with `ollama serve`.
- Telemetry is enabled; inspect dogfood data with `model-router telemetry summary`.
- Agent base URL: http://127.0.0.1:8082/v1 with model `model-router`.
- Start proxy: model-router-proxy --config /Users/you/.model-router/routing_proxy.yaml
```

Once LM Studio, Ollama, or another OpenAI-compatible upstream is running, start
the proxy:

```text
$ model-router-proxy --config ~/.model-router/routing_proxy.yaml --log-level info
INFO routing proxy ready host=127.0.0.1 port=8082 backends=balanced,code,fast,reasoning
INFO:     Uvicorn running on http://127.0.0.1:8082 (Press CTRL+C to quit)
```

Generic agent/client configuration:

```text
Base URL: http://127.0.0.1:8082/v1
Model: model-router
API key: leave blank unless you set proxy.api_key or proxy.api_key_env
Chat endpoint: /v1/chat/completions
Responses endpoint: /v1/responses
```

Use `model-router doctor --config ~/.model-router/routing_proxy.yaml` when a
backend is unavailable or a model name/endpoint is wrong.

### Real Proxy Dogfood

Plan the local proxy dogfood checklist without contacting the proxy or any
runtime:

```bash
model-router dogfood proxy --config ~/.model-router/routing_proxy.yaml
```

From a source checkout, `python scripts/dogfood_proxy.py --config
~/.model-router/routing_proxy.yaml` runs the same harness.

Run the live checks only when the proxy and local runtimes are intentionally
available:

```bash
model-router dogfood proxy \
  --config ~/.model-router/routing_proxy.yaml \
  --execute
```

The harness covers `/health`, `/v1/models`, `/v1/chat/completions`, streaming
chat, `/v1/responses` when supported, fallback, backend policy rejection,
`human_confirm`, and verifier-mode visibility. Live checks are opt-in, use fixed
sanitized smoke prompts, skip clearly when a runtime or policy setup is not
available, and do not start providers, download models, enable hosted APIs, or
turn on verifiers.

Routed proxy responses include privacy-safe identifiers you can copy while
dogfooding:

```text
X-ModelRouter-Request-ID: req id to label with model-router feedback
X-ModelRouter-Engine: selected route, such as fast_local or code_agent
X-ModelRouter-Backend: configured backend name when a backend was used
X-ModelRouter-Fallback: true when an upstream fallback was used
X-ModelRouter-Route-API: route_fast
```

When the proxy shuts down, it prints a session summary with route counts and the
`model-router telemetry summary ...` command for the configured event log.

## Hindsight Routing Logs

The proxy can write privacy-safe JSONL events for calibration and replay:

```yaml
observability:
  enabled: true
  log_path: ~/.model-router/logs/routing-events.jsonl
  prompt_capture: redacted_preview
```

By default events keep a prompt hash, length, estimated tokens, selected engine,
scores, feature flags, backend, fallback status, and latencies. Raw prompts are
not stored unless `prompt_capture: full` or `MODEL_ROUTER_LOG_PROMPTS=1` is set.
Use full capture only during deliberate calibration runs.

Cost and outcome telemetry follows the same privacy-first boundary. Routing uses
configured `cost_tier` metadata and provider policy; `route_fast(...)` and
`route(...)` do not fetch live prices. Proxy telemetry records upstream token
usage when providers return it, accepts explicit outcome labels from
users/operators, and estimates cost only from a local versioned pricing catalog.
Missing usage or pricing metadata means no exact estimate.

Pricing metadata is packaged and local-first. Operators can inspect and maintain
the local override explicitly:

```bash
model-router pricing status
model-router pricing diff
model-router pricing apply --yes
```

These commands read packaged/local files only. They do not scrape provider pages
or run during routing/proxy forwarding. See
[Versioned pricing catalog](docs/pricing-catalog.md) for the override file
shape and an operator-verified provider example.

When a route is wrong, copy `X-ModelRouter-Request-ID` from the response and
label it:

```bash
model-router feedback req-123 code_agent --notes "repo prompt routed too small"
```

Replay captured traffic against the current router:

```bash
model-router telemetry summary \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl
```

Inspect labels without printing notes:

```bash
model-router telemetry feedback \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl
```

Review unlabeled wrong-route candidates without printing prompt bodies,
previews, request bodies, or feedback notes:

```bash
model-router telemetry review \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl
```

Run strict replay for CI-style checks:

```bash
python scripts/replay_routing_log.py \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl \
  --json \
  --fail-on-regression
```

Rows without full prompts are skipped for replay but still useful for aggregate
latency, score, fallback, and route distribution analysis.

### Wrong Route To Regression Test

1. Turn on observability during a calibration run. Use `prompt_capture: full`
   only when you are deliberately collecting replay fixtures.
2. Send the prompt through the proxy and copy `X-ModelRouter-Request-ID` from
   the response. If the client hides response headers, copy the matching
   `request_id` from `~/.model-router/logs/routing-events.jsonl`.
3. Label the intended engine:

```bash
model-router feedback req-123 balanced_local \
  --notes "short summary prompt should not escalate to reasoning"
```

4. Replay before changing router rules:

```bash
python scripts/replay_routing_log.py \
  --events ~/.model-router/logs/routing-events.jsonl \
  --feedback ~/.model-router/routing-feedback.jsonl \
  --json
```

5. Add the prompt to a small fixture or parametrized test, update deterministic
   scoring/routing rules, rerun replay, and keep the new test with the fix.

This loop is the preferred way to turn a real wrong route into a durable
regression case without adding LLM classification or slowing `route_fast(...)`.
See [Routing telemetry dogfood](docs/telemetry-dogfood.md) for coverage
summaries, feedback inspection, privacy defaults, and the data threshold for
revisiting optional advanced routing.

## Troubleshooting

- Wrong route: enable observability, label the request with
  `model-router feedback`, and replay logs before changing scoring.
- Backend unavailable or wrong model: run `model-router doctor --config
  ~/.model-router/routing_proxy.yaml` and check `/health`. Both diagnostics
  verify backend reachability and, when `/v1/models` returns a model list, that
  each configured backend model is advertised by the upstream server.
- `human_confirm`: the prompt matched a destructive, sending, purchase/payment,
  deployment, or other high-impact action. Use explicit safety overrides only
  in versioned configs.
- Proxy auth: if `proxy.api_key` or `proxy.api_key_env` is configured, clients
  must send `Authorization: Bearer <token>`.
- Logs/replay: default logs do not include raw prompts. Use
  `prompt_capture: full` or `MODEL_ROUTER_LOG_PROMPTS=1` only during deliberate
  calibration runs.

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
  "summary": "Selected code_agent under the balanced profile; no confirmation required; fallback available: reasoning_local.",
  "reason_codes": [
    "profile.balanced",
    "route.coding",
    "requirement.tools",
    "requirement.code_execution",
    "delegation.verification_heavy_likely",
    "delegation.repo_wide_likely",
    "fallback.configured",
    "safety.no_confirmation_required"
  ],
  "delegation_suitability": {
    "mechanical_work_likely": false,
    "judgment_heavy_likely": false,
    "verification_heavy_likely": true,
    "repo_wide_likely": true,
    "risky_or_external_action": false,
    "ambiguity_sensitive": false,
    "reasons": [
      "Verification or test-running cost is likely.",
      "Repository-wide or multi-file scope is likely."
    ],
    "guidance": "Candidate for sidekick-style delegation when host policy and provider constraints allow it."
  },
  "selected_route_explanation": "Selected code_agent for coding or repository work.",
  "policy_explanation": "Profile: routing profile balanced uses default deterministic routing.",
  "fallback_explanation": "No fallback was used; reasoning_local remains the configured fallback.",
  "safety_explanation": "No human confirmation is required by the current safety policy.",
  "privacy_explanation": "No raw prompt text is stored in this receipt; provider use follows the configured catalog and policy.",
  "wrong_route_next_action": "If this route was wrong, review the local event with `model-router telemetry review`, then label the proxy request id with `model-router feedback <request_id> <expected_engine>` or rerun `model-router decide --explain` with adjusted profile/provider policy.",
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

Human confirmation is a default-on safety feature. Escape hatches are explicit,
scoped config choices:

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

The setup assistant can create a local config without guessing what you want.

For normal onboarding, start with the deterministic installer plan:

```bash
model-router install --quick
model-router install --auto --json
model-router install --local-only --mlx-lm
```

Use the generated next commands for prerequisites, `init`, `doctor`, settings,
proxy startup, download planning, and telemetry. Config writes and downloads
remain explicit follow-up commands.

Scan your machine:

```bash
model-router setup scan
model-router setup scan --json
```

Get recommendations:

```bash
model-router setup recommend
model-router setup recommend --download-alternatives 2
model-router setup recommend --json
```

Recommendations are produced by a bundled, versioned model advisor catalog at
`hermes/plugins/model_router/data/model_catalog.yaml`. The advisor detects basic
local hardware signals such as RAM, CPU architecture, CPU core count, Apple
Silicon, accelerator backend hints, and free disk space, then ranks setup-time
Hugging Face suggestions for each route. RAM is treated as the fit/load gate;
CPU, accelerator backend, model format, quantization, and benchmark results
drive the usability score. This does not run during `route_fast(...)`,
`route(...)`, or ordinary `decide` calls.

Each recommendation includes a score breakdown:

- `fit_score`: memory fit/load headroom.
- `runtime_match_score`: MLX-LM, llama.cpp/GGUF, Ollama, LM Studio, or generic
  runtime fit for the detected machine.
- `expected_speed_score`: CPU/core/accelerator, model size, and quantization
  usability estimate.
- `quality_role_score`: route-specific quality fit.
- `setup_friction_score`: expected setup effort.
- `benchmark_score`: neutral until a local benchmark result exists.

Run a privacy-safe local backend benchmark when you want measured data:

```bash
model-router setup benchmark \
  --config ~/.model-router/routing_proxy.yaml

model-router setup benchmark \
  --config ~/.model-router/routing_proxy.yaml \
  --execute --yes
```

The benchmark command targets configured backends only, sends a fixed synthetic
smoke prompt, stores metrics in `~/.model-router/benchmarks.json`, and never
stores prompt bodies, request bodies, API keys, or secrets. Benchmark results
can improve future recommendation ranking, but they only propose choices; they
never mutate config or routing policy automatically.

## Catalog Update Workflow

Model and preset recommendations come from packaged catalogs. Use the catalog
workflow to see what the installed package would change before accepting it:

```bash
model-router catalog status \
  --config ~/.model-router/model_router.yaml

model-router catalog diff \
  --config ~/.model-router/model_router.yaml

model-router catalog apply \
  --config ~/.model-router/model_router.yaml \
  --yes
```

The workflow is packaged-only today: it does not fetch remote catalogs or
silently change routing policy. `status` reports the packaged model catalog
version, local config hash, migration log path, and local overrides. `diff`
previews packaged router catalog changes. `apply` requires confirmation, backs
up an existing local config before writing, and records the action in
`catalog-migrations.jsonl`. The settings UI shows catalog status but does not
apply updates.

Run the wizard:

```bash
model-router setup wizard \
  --output configs/model_router.local.yaml
```

Write a recommended config non-interactively:

```bash
model-router setup write \
  --output configs/model_router.local.yaml
```

`setup write` will not overwrite an existing file unless `--force` is passed.

The wizard asks whether you want:

- Local LLMs only.
- API keys / hosted models.
- A mix of local models, hosted APIs, and agent tools.

It then walks each main route and shows numbered local model choices plus
hardware-aware recommended downloads. Local route-matched models can fill config
defaults, but recommended downloads are still shown as alternatives when the
catalog knows a stronger or better-fitting option. Downloads are never run by
ordinary routing commands. They require explicit confirmation.

The scanner includes current LM Studio model storage at
`~/.lmstudio/models`, plus Ollama, Hugging Face cache, and common local model
folders, so wizard choices should reflect the models your local tools can see.

If recommended downloads are available and the Hugging Face `hf` CLI is missing,
the wizard warns at the beginning and asks whether to install it into the current
Python environment before model choices start. Declining is safe; the router can
still write the config, and downloads can be run later.

Plan downloads:

```bash
model-router setup download
model-router setup download --route fast_local
model-router setup download --route fast_local --alternatives 2
```

Run an approved Hugging Face download:

```bash
model-router setup download \
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
  and external-action prompts require confirmation by default.
- Confirmation escape hatches must be explicit in `safety.confirmation_overrides`;
  the router does not learn approvals or silently relax safety rules.
- `force_engine` cannot bypass human confirmation or provider/backend policy.
- Missing or invalid config routes to `human_confirm`.
- Unavailable, incompatible, or policy-denied engines are skipped through
  configured fallbacks without escaping denied providers/backends.
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
cd /path/to/model-router
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
model-router decide --json "fix the repo and run tests"
```

For a non-editable install from GitHub:

```bash
python -m pip install "git+https://github.com/doncazper/model-router.git@v0.7.2"
model-router decide "rewrite this text"
```

The package exposes console commands, `model-router` and the legacy
`hermes-router` alias, plus
the importable Python API:

```python
from model_router import ModelRouter

router = ModelRouter.from_config()
engine = router.route_fast(prompt)
```

The default catalog is included as package data, so `ModelRouter.from_config()`
works after wheel installation without relying on the repository checkout. Pass
an explicit config path when an embedding app needs its own engine catalog.

See `examples/basic_custom_agent.py` for a minimal host-neutral integration.

ModelRouter does not currently claim any host-app plugin manifest or automatic
per-turn model switching contract. Embedding applications should use their own
runtime integration boundary and call the stable `route_fast(...)` production
API.

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
model_router/
  __init__.py          # Generic public import path
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
examples/
  basic_custom_agent.py
scripts/
  benchmark_route_fast.py
tests/
```

The `model_router` package is the generic public import path. The
`hermes/plugins` path is retained only as a backward-compatible legacy Python
namespace from the original Hermes Router package. Neither path is a
host-application plugin manifest, host plugin registry, or automatic
integration point.

## Documentation

- [Model router details](docs/model-router.md)
- [Product north star](docs/product-north-star.md)
- [Production readiness](docs/production-readiness.md)
- [Release checklist](docs/release-checklist.md)
- [Upgrade and uninstall](docs/upgrade-uninstall.md)
- [Host adapter contract](docs/adapter-contract.md)
- [Open switchboard robustness plan](docs/open-switchboard-plan.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)

## Roadmap

### v0.5: Usable Local Proxy Beta

- Keep the proxy-first install path polished: `pip install "hermes-router[proxy]"`.
- Keep `model-router init`, `validate-proxy-config`, `doctor`, `/health`, log
  rotation, and provider presets reliable.
- Publish releases with a changelog, GitHub release notes, and benchmark output.

### v0.6: Passthrough And Gateway Mode

- Add router mode and passthrough mode.
- Keep legacy command/import aliases for one release.
- Add backend request overrides for temperature, context, max tokens, and common
  generation controls.
- Add first-class llama.cpp and MLX gateway templates.

### v0.6.5: Managed Local Runtime Beta

- Add explicit proxy-owned process configuration for llama.cpp and MLX backends.
- Demand-start configured runtimes on first route and stop them after an idle
  timeout.
- Detect missing commands, readiness failures, placeholder model ids, and port
  conflicts through `doctor`.
- Capture per-backend process logs to configured files.
- Keep process management opt-in and transparent; never auto-start arbitrary
  commands without user configuration.

### v1.0: Finished Local AI Gateway

- Version the public config schema and provide migrations.
- Use labeled real-world routing logs as release-blocking regression checks.
- Document security/privacy expectations for logs, proxy auth, process commands,
  and local network exposure.
- Keep the local admin UI aligned with the product north star: a proxy control
  center, not chat, not an agent workspace, and not a prompt transcript product.
