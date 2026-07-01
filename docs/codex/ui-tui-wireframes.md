# UI and TUI wireframes

These are implementation wireframes, not decorative mockups. The current settings UI style should stay: light surface, sticky topbar, compact cards, plain tables, conservative accent color, explicit save/start/restart actions, and privacy-first text. The goal is to make ModelRouter feel like one local AI control center for proxy status, routing mode, active backend/model, local runtime status, model library, route-aware recommendations, explicit downloads, telemetry/cost/outcome/catalog coverage, safety/policy state, and request routing, while still integrating with external runtimes instead of copying LM Studio's layout wholesale or replacing proven inference servers.

## Shared UI rules

1. Do not render placeholder cards. Render real state, a real empty state, or a disabled control with a clear reason.
2. Every visible field must map to shared admin state.
3. Every button must call a shared admin action.
4. Any config write, download, runtime load/unload, benchmark execution, or proxy process change requires confirmation.
5. Web UI and TUI use the same state and action layer.
6. Basic router mode must be visible everywhere routing decisions are visible.
7. Runtime controls must distinguish ModelRouter-managed processes from
   external runtimes such as LM Studio, Ollama, LocalAI, llama.cpp, MLX/MLX-LM,
   vLLM, generic OpenAI-compatible services, and hosted providers.
8. Do not imply ModelRouter builds its own inference engine; adapter capability
   gaps and unsupported actions must be visible.
9. Recommendation and download surfaces are operational controls. Keep them
   compact by default and expand only for install details, alternatives, logs,
   or confirmed download actions.
10. Chat/playground surfaces, if added later, are secondary to control-plane
   operations and must not dominate the main viewport.

## Web UI navigation

Recommended tabs:

```text
Dashboard | Models | Routing | Runtimes | Telemetry | Logs | Settings
```

Each tab has a specific job:

| Tab | Purpose | Must be fully wired to |
| --- | --- | --- |
| Dashboard | Fast operational overview: endpoint, proxy state, routing mode, active backend/model, runtime health, telemetry/catalog coverage, safety/policy, latest receipt, next actions. | `proxy`, `latest_receipt`, `telemetry`, `backends`, `actions` |
| Models | Installed models, discover/search, route recommendations, downloads, route assignment. | `model_library`, `model_aliases`, `routes`, `backends`, `actions` |
| Routing | Smart-router route map and basic-router model mapping. | `proxy.routing_mode`, `routes`, `model_aliases`, `backends`, `actions` |
| Runtimes | LM Studio/Ollama/MLX/llama.cpp/LocalAI/vLLM status, loaded models, runtime controls. | `backends`, `actions`, `logs` |
| Telemetry | Recent requests, route counts, usage, cost/outcome labels, catalog coverage, fallback counts, wrong-route queue. | `telemetry`, `latest_receipt`, `actions` |
| Logs | Proxy/settings/runtime log tails with safe copy/open actions. | `logs`, `backends`, `proxy` |
| Settings | Proxy config, mode, policy, observability, verifier, installer status, catalog updates. | `proxy`, `installer`, `actions`, provider/backend policy state |

## Window Modes

The web UI has two alternate states:

- Full control center/main window: the complete dashboard and settings surface.
- Compact minimal control panel/windowed mode: a smaller standalone page/window
  for quick status and safe controls.

The compact control panel must not be drawn as a modal, overlay, child window, or
floating element on top of the dashboard. When wireframes show the full and
compact states near each other, treat them as a side-by-side comparison of
modes, not as simultaneous in-app layers.

## Mock screenshot: Dashboard

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ ModelRouter                 Endpoint http://127.0.0.1:8082/v1     ● Running │
│ Dashboard  Models  Routing  Runtimes  Telemetry  Logs  Settings             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Status: proxy running · mode decision · balanced → ollama → qwen3:4b         │
│ Runtime: reachable · telemetry on · catalog coverage medium · safety normal  │
│ [Start/Stop proxy] [Run doctor] [Copy endpoint] [Review coverage gaps]       │
│                                                                              │
│ Mode                                                                    Save │
│ ┌ Smart router ─────────┐  ┌ Manual backend ───────┐  ┌ Model aliases ─────┐ │
│ │ ● Decision layer on   │  │ ○ No classification   │  │ ○ Client model map │ │
│ │ Profile: Balanced     │  │ Default: balanced     │  │ 4 aliases enabled  │ │
│ └───────────────────────┘  └───────────────────────┘  └───────────────────┘ │
│                                                                              │
│ Latest route receipt                                                         │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ req_8fa21c  decision  reasoning_local → mlx-reasoning → qwen3-r1-8b     │ │
│ │ Why: route.reasoning, profile.balanced, local backend available          │ │
│ │ Fallback: no   Safety: no confirmation   Route latency: 2.8 ms          │ │
│ │ [Copy receipt JSON] [Label wrong route]                                  │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ Health                                                                       │
│ ┌─────────────┬───────────┬──────────┬────────────┬───────────────────────┐ │
│ │ Backend     │ Runtime   │ Reachable│ Model      │ Capability gaps       │ │
│ ├─────────────┼───────────┼──────────┼────────────┼───────────────────────┤ │
│ │ fast        │ LM Studio │ yes      │ qwen3-0.6b │ embeddings            │ │
│ │ balanced    │ Ollama    │ yes      │ qwen3:4b   │ structured output?    │ │
│ │ reasoning   │ MLX-LM    │ managed  │ r1-qwen8b  │ responses translation │ │
│ └─────────────┴───────────┴──────────┴────────────┴───────────────────────┘ │
│ [Run doctor] [Restart proxy] [Open TUI help]                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard field mapping

| Field/control | State/action |
| --- | --- |
| Endpoint | `proxy.endpoint` |
| Running indicator | `proxy.state`, `proxy.pid` |
| Mode cards | `proxy.routing_mode`, `proxy.decision_layer_enabled` |
| Profile | `proxy.routing_profile` |
| Active backend/model | latest route and `proxy.default_backend`/`proxy.default_model` fallback |
| Runtime health | `backends[].health`, `backends[].runtime` |
| Telemetry/catalog coverage | `telemetry.catalog_coverage`, usage/cost/outcome summary fields |
| Safety/policy state | provider/backend policy state and `proxy.safety_gate_mode` |
| Default backend | `proxy.default_backend` |
| Alias count | count of enabled `model_aliases` |
| Latest receipt | `latest_receipt` |
| Backend table | `backends` |
| Run doctor | `doctor.run` |
| Restart proxy | `proxy.restart` |
| Label wrong route | `telemetry.feedback.write` |

## Mock screenshot: Compact windowed mode

This is a standalone smaller app surface, not a layer over the dashboard.

```text
┌──────────────────────────────────────┐
│ ● ● ●  ModelRouter                 ⚙ │
├──────────────────────────────────────┤
│ 127.0.0.1:8082/v1  ● Running decision│
│                                      │
│ Active balanced → ollama → qwen3:4b  │
│ Runtime reachable  Telemetry on      │
│ Catalog medium     Safety normal     │
│                                      │
│ Request → reasoning_local → mlx → OK │
│                                      │
│ Selected        reasoning_local      │
│ Backend         mlx-reasoning        │
│ Privacy         local/provider policy│
│ Safety          no confirmation      │
│                                      │
│ Recent                               │
│ 10:42  fast_local      fast      84ms│
│ 10:43  reasoning_local mlx      1.2s │
│                                      │
│ [Full] [Pause] [Receipt] [Providers] │
│ Proxy running   Telemetry on          │
└──────────────────────────────────────┘
```

### Compact window field mapping

| Field/control | State/action |
| --- | --- |
| Endpoint | `proxy.endpoint` |
| Running indicator | `proxy.state`, `proxy.pid` |
| Profile | `proxy.routing_profile` |
| Routing mode | `proxy.routing_mode`, `proxy.decision_layer_enabled` |
| Active backend/model | latest route/backend/model or configured default |
| Runtime status | selected backend health/runtime status |
| Telemetry/catalog state | usage/cost/outcome/catalog coverage summary |
| Safety/policy state | safety gate and provider/backend policy summary |
| Latest route/backend | `latest_receipt` |
| Recent rows | privacy-safe `telemetry` summaries |
| Pause proxy | confirmed `proxy.stop` |
| Full | link to the full control center |

## Mock screenshot: Models tab

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Models                                                        Scan  Discover │
├──────────────────────────────────────────────────────────────────────────────┤
│ Model operations: Installed 3 · Recommended 4 · Downloads 1 planned         │
│ Active: balanced → ollama → qwen3:4b · code → llama.cpp → coder.gguf        │
│ [Scan] [Discover] [Plan downloads] [Assign selected]                         │
│                                                                              │
│ ▸ Installed        3 local models                                            │
│ ▸ Discover         12 catalog candidates                                     │
│ ▸ Recommended      4 candidates; compact by default                          │
│ ▸ Downloads        1 planned; expand for commands                            │
│ ▸ Assignments      8 route bindings                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

Recommendation and download state should be compact by default: show route,
model, fit label, runtime compatibility, eligibility, status, and next action in
rows. Expand a row only when the user chooses to inspect install details,
alternatives, logs, commands, or download actions.

### Models tab field mapping

| Field/control | State/action |
| --- | --- |
| Search query | `model_library.discover.query` |
| Runtime filter | `model_library.discover.filters.runtime_kind` |
| Route filter | `model_library.discover.filters.route` |
| Installed rows | `model_library.installed` |
| Loaded value | `backends.loaded_models` joined by model id |
| Fit label | installed model score label/reasons |
| Scan | `model.scan` |
| Discover | `model.discover` |
| Load | `runtime.load_model`, enabled only if backend supports load |
| Unload | `runtime.unload_model`, enabled only if backend supports unload |
| Assign route | `model.assign_route` |
| Download run | `model.download.run` with confirmation |

## Mock screenshot: Discover / marketplace flow

```text
┌ Discover models ─────────────────────────────────────────────────────────────┐
│ [Search Hugging Face or curated catalog:  qwen3 gguf              ] [Search] │
│ Filters: Route [Balanced ▾] Runtime [llama.cpp ▾] Max size [12 GB] [Local ok]│
├──────────────────────────────────────────────────────────────────────────────┤
│ Recommended results                                                         │
│ ┌───────────────────────────────┬────────────┬────────────┬───────────────┐ │
│ │ Model                         │ Fit        │ Requirements│ Action        │ │
│ ├───────────────────────────────┼────────────┼────────────┼───────────────┤ │
│ │ lmstudio-community/Qwen3-4B   │ Great      │ 8-16 GB RAM│ Plan download │ │
│ │ Qwen/Qwen2.5-3B-Instruct-GGUF │ Good       │ 8 GB RAM   │ Plan download │ │
│ │ mlx-community/Qwen3-4B-4bit   │ Apple-only │ 8-16 GB RAM│ Plan download │ │
│ └───────────────────────────────┴────────────┴────────────┴───────────────┘ │
│                                                                              │
│ Selected model                                                               │
│ Why this model: balanced route, local runtime match, size fits hardware.     │
│ Warnings: structured output not verified.                                    │
│ [Plan download] [Copy command] [Assign after download: balanced ▾]           │
└──────────────────────────────────────────────────────────────────────────────┘
```

Required behavior:

- The first implementation may search the packaged curated catalog only.
- Hugging Face search is an experimental adapter behind a feature flag until result scoring and rate-limit behavior are reliable.
- Download buttons create a plan first. They must not immediately download.
- Download execution must show status and failure details.

## Mock screenshot: Routing tab in decision mode

```text
┌ Routing ─────────────────────────────────────────────────────────────────────┐
│ Routing mode: [Smart router ▾] Profile: [Balanced ▾]       [Save] [Restart] │
│ Decision layer: enabled                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Route map                                                                    │
│ ┌───────────────┬───────────────┬──────────────┬────────────┬──────────────┐ │
│ │ Route class   │ Engine        │ Backend      │ Model      │ Fallback     │ │
│ ├───────────────┼───────────────┼──────────────┼────────────┼──────────────┤ │
│ │ Simple        │ fast_local    │ fast         │ qwen3-0.6b │ balanced     │ │
│ │ Balanced      │ balanced_local│ balanced     │ qwen3-4b   │ reasoning    │ │
│ │ Reasoning     │ reasoning_local│ reasoning   │ r1-qwen8b  │ hosted       │ │
│ │ Coding        │ code_agent    │ code         │ coder-7b   │ human_confirm│ │
│ └───────────────┴───────────────┴──────────────┴────────────┴──────────────┘ │
│                                                                              │
│ Test route                                                                   │
│ Prompt preview [design a migration plan...] [Run dry route]                  │
│ Result: reasoning_local because route.reasoning + architecture markers       │
│ [Copy receipt] [Add regression fixture note]                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Mock screenshot: Routing tab in basic router mode

```text
┌ Routing ─────────────────────────────────────────────────────────────────────┐
│ Routing mode: [Model aliases ▾]                         [Save] [Restart]    │
│ Decision layer: disabled                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ Basic router behavior                                                       │
│ Default backend [balanced ▾] Default model [qwen3-4b ▾]                     │
│ Respect client model [on] Unknown model [fallback_to_default ▾]             │
│ Safety gate [decision_only ▾]                                                │
│                                                                              │
│ Model aliases                                                                │
│ ┌────────────┬──────────┬──────────────┬───────────────┬──────────────────┐ │
│ │ Alias      │ Backend  │ Model        │ Capabilities  │ Description      │ │
│ ├────────────┼──────────┼──────────────┼───────────────┼──────────────────┤ │
│ │ qwen-fast  │ fast     │ qwen3-0.6b   │ chat,stream   │ Fast lane        │ │
│ │ qwen-code  │ code     │ coder-7b     │ chat,tools?   │ Coding           │ │
│ │ bge        │ research │ bge-m3       │ embeddings    │ Retrieval        │ │
│ └────────────┴──────────┴──────────────┴───────────────┴──────────────────┘ │
│ [Add alias] [Edit selected alias] [Delete alias] [Preview /v1/models]        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Routing tab field mapping

| Field/control | State/action |
| --- | --- |
| Routing mode select | `proxy.routing_mode`, writes through `config.set_routing_mode` |
| Profile select | `proxy.routing_profile`, writes through `config.save_proxy_patch` |
| Default backend | `proxy.default_backend`, candidates from `backends` |
| Default model | `proxy.default_model`, candidates from backend models |
| Respect client model | `proxy.respect_client_model` |
| Unknown model behavior | `proxy.unknown_model_behavior` |
| Safety gate | `proxy.safety_gate_mode` |
| Route map | `routes` |
| Aliases table | `model_aliases` |
| Dry route/test route | read-only route preview action; must not call an upstream model |

## Mock screenshot: Runtimes tab

```text
┌ Runtimes ────────────────────────────────────────────────────────────────────┐
│ Runtime providers: LM Studio  Ollama  MLX-LM  llama.cpp  LocalAI  Generic   │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌───────────┬─────────────┬──────────┬───────────────┬────────────────────┐ │
│ │ Backend   │ Runtime     │ Health   │ Loaded models │ Actions            │ │
│ ├───────────┼─────────────┼──────────┼───────────────┼────────────────────┤ │
│ │ fast      │ LM Studio   │ reachable│ qwen3-0.6b    │ Load Unload Logs   │ │
│ │ balanced  │ Ollama      │ reachable│ qwen3:4b      │ Pull Tags Logs     │ │
│ │ reasoning │ MLX-LM      │ managed  │ r1-qwen8b     │ Start Stop Logs    │ │
│ │ code      │ llama.cpp   │ stopped  │ coder.gguf    │ Start Stop Logs    │ │
│ └───────────┴─────────────┴──────────┴───────────────┴────────────────────┘ │
│                                                                              │
│ Capability details for balanced                                              │
│ chat yes | responses yes | embeddings no | tools unknown | structured unknown│
└──────────────────────────────────────────────────────────────────────────────┘
```

Required behavior:

- If a runtime cannot load/unload models through an API, the button is disabled with a reason.
- If a runtime is managed by ModelRouter, start/stop controls call the managed runtime manager.
- If a runtime is external, show exact external command or docs hint where known.
- If a runtime owns inference, ModelRouter should show status and supported
  adapter controls rather than pretending to execute the model itself.

## Mock screenshot: Settings tab

```text
┌ Settings ────────────────────────────────────────────────────────────────────┐
│ Proxy                                                                        │
│ Host [127.0.0.1] Port [8082] Mode [decision ▾] Profile [balanced ▾]         │
│ Endpoint preview: http://127.0.0.1:8082/v1                                  │
│                                                                              │
│ Observability                                                                │
│ Telemetry [on] Prompt capture [redacted_preview ▾] Log path [.../events.jsonl]
│                                                                              │
│ Provider/backend policy                                                      │
│ Local only [off] Hosted allowed [on] Backend denylist [              ]       │
│                                                                              │
│ Installer                                                                    │
│ Method: uv_tool   Optional deps: proxy yes, tui no, HF CLI yes              │
│ [Install TUI extra] [Run doctor] [Validate config] [Save]                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

Settings must remain conservative: no literal secret display, no hidden hosted-provider enablement, and no silent restart.

## TUI layout

The TUI should mirror the web UI tabs and use Textual widgets such as `TabbedContent`, `DataTable`, `Input`, `Select`, `Switch`, `Log`, and confirmation modals.

```text
┌ ModelRouter ────────────────────────────────────────────────────────────────┐
│ Endpoint http://127.0.0.1:8082/v1  Proxy RUNNING  Mode decision  Profile bal │
├ Status ┬ Models ┬ Routing ┬ Runtimes ┬ Telemetry ┬ Logs ┬ Settings ────────┤
│                                                                              │
│ Status                                                                       │
│   Config valid: yes                                                          │
│   Decision layer: enabled                                                    │
│   Latest: req_8fa21c reasoning_local → mlx-reasoning                         │
│                                                                              │
│ Routes                                                                       │
│   Simple      fast_local       fast        qwen3-0.6b                        │
│   Balanced    balanced_local   balanced    qwen3-4b                          │
│   Reasoning   reasoning_local  reasoning   r1-qwen8b                         │
│                                                                              │
│ Actions: [s] start/stop  [r] restart  [d] doctor  [m] mode  [q] quit        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### TUI required bindings

| Key | Action |
| --- | --- |
| `q` | quit |
| `r` | confirm `proxy.restart` |
| `s` | start or stop proxy depending on `proxy.state` |
| `d` | `doctor.run` |
| `m` | open routing mode selector |
| `f` | write feedback for selected request |
| `/` | focus search/filter on the active tab |
| `?` | show help |

### TUI no-placeholder rule

If no models are found, render:

```text
No local models were found in configured scan paths.
Actions: [Scan again] [Open download recommendations] [Edit model paths]
```

Do not render fake model rows.

## Golden render requirement

For each major tab, add a snapshot-style render test or stable text fixture:

```text
tests/snapshots/settings_dashboard.txt
tests/snapshots/settings_models.txt
tests/snapshots/settings_routing_decision.txt
tests/snapshots/settings_routing_basic.txt
tests/snapshots/tui_status.txt
tests/snapshots/tui_models_empty.txt
```

The snapshots should not assert exact terminal colors, but they should assert that key labels, state-derived values, disabled reasons, and actions appear.
