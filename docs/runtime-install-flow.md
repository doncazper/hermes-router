# Runtime Install Flow

This document defines installer-assisted runtime setup for ModelRouter. It is
both the product boundary for future runtime setup and the guardrail for the
current preview-first MVP. The current CLI supports runtime status/doctor and
safe connect previews for LM Studio, Ollama, and llama.cpp; broader install
execution remains future work. The current `model-router install` command is
plan-only and prints next commands without mutating by default. Runtime setup
should preserve that shape.

ModelRouter may help operators connect or install proven runtimes, but it must
not silently install packages, pull models, write config, start services, or
change routing. Passing `--yes` to a planning command may record confirmation
intent, but it must not turn plan output into silent execution. Runtime
install/connect flows are operator maintenance surfaces only. They must never
run from `route_fast(...)`, `route(...)`, proxy forwarding, or default request
handling.

## Goals

- Help users get from "no runtime" to "configured backend with health/status"
  without editing YAML first.
- Support both "connect to something already running" and "install/configure a
  runtime I choose" paths.
- Keep LM Studio, Ollama, llama.cpp, MLX-LM, LocalAI, vLLM, hosted providers,
  and generic OpenAI-compatible servers first-class.
- Make every mutating step explicit, previewed, and confirmed.
- Convert verified runtime state into model registry entries without making
  route decisions depend on live discovery.

## Non-Goals

- No silent installs or shell profile edits.
- No silent model downloads or pulls.
- No hidden service installation, startup item registration, or daemon changes.
- No live runtime discovery as a routing signal.
- No custom inference engine.
- No automatic migration away from LM Studio, Ollama, LocalAI, vLLM, or any
  existing runtime manager.

## Flow Shape

Runtime setup should extend the existing `model-router install` plan model with
runtime-specific steps. The flow has six phases:

1. **Detect**
   Read config, PATH, platform, local ports, known model folders, optional
   Python modules, and bounded local health endpoints.

2. **Recommend**
   Pick a candidate path such as "connect LM Studio", "install Ollama", "use
   llama.cpp with this GGUF", or "connect vLLM".

3. **Preview**
   Show all commands, links, config patches, ports, model paths, environment
   variables, permissions, and files that would change.

4. **Confirm**
   Require explicit confirmation before any config write, dependency install,
   model pull/download, runtime start, or service registration.

5. **Execute**
   Run only the confirmed step. Prefer opening official installer pages for GUI
   apps. For CLI installs, show the exact command before execution.

6. **Verify And Register**
   Run bounded health/model discovery after the operator starts or connects the
   runtime. Add discovered models to the registry as runtime-discovered entries
   with source/runtime/backend metadata.

Read-only phases can be run from CLI, settings, or TUI. Mutating phases must be
explicit actions with confirmation, and they must be unavailable in routing and
proxy forwarding paths.

## Action Types

Installer plans should use JSON-safe action records:

```json
{
  "id": "runtime.ollama.install",
  "runtime_id": "ollama",
  "phase": "install",
  "kind": "external_link|shell_command|config_patch|runtime_action|health_check",
  "label": "Install Ollama",
  "preview": "curl -fsSL https://ollama.com/install.sh | sh",
  "mutates": true,
  "requires_confirmation": true,
  "official_source": "https://ollama.com/download",
  "rollback_hint": "Remove the backend config; uninstall Ollama through the OS package/app flow."
}
```

Minimum fields:

- `id`
- `runtime_id`
- `runtime_mode`: `external_managed`, `external_cli`, or `bundled_future`
- `platforms`: `macos`, `windows`, `linux`
- `kind`
- `label`
- `description`
- `preview`
- `mutates`
- `requires_confirmation`
- `official_source`
- `security_notes`
- `rollback_hint`
- `registry_effect`

## Platform Rules

### macOS

- Prefer official app installers for LM Studio, Ollama, and LocalAI when the
  runtime is GUI-first.
- Prefer per-user Python virtual environments for Python runtimes such as
  MLX-LM or vLLM. Do not install into system Python.
- Surface Apple Silicon checks before recommending MLX-LM.
- If using Homebrew or command-line installers, preview the command and explain
  whether it writes outside `~/.model-router`.
- Warn when a command requires sudo, LaunchAgent, background service, or shell
  profile changes.

### Windows

- Prefer official GUI installers or PowerShell commands copied from official
  docs.
- Never assume WSL, Docker Desktop, CUDA, or Visual Studio Build Tools are
  installed.
- Treat service installation, firewall prompts, PATH changes, and GPU driver
  changes as high-friction steps requiring explicit operator review.
- Use PowerShell-friendly command previews when a runtime provides them.

### Linux

- Prefer official container instructions for LocalAI and server runtimes when
  they are the documented default.
- Do not run package manager commands with `sudo` automatically.
- Surface Docker/Podman/Kubernetes choices as alternatives, not assumptions.
- Warn when GPU access needs extra container flags, device mounts, drivers,
  CUDA/ROCm packages, or group membership changes.

## Security And Permissions

Runtime install/connect plans must show:

- Whether the endpoint is bound to `127.0.0.1`, `0.0.0.0`, or a remote host.
- Whether API keys or bearer tokens are configured, without displaying secret
  values.
- Whether the step writes config, installs packages, starts a process, creates a
  service, opens a firewall port, downloads model weights, or mounts local
  directories into a container.
- Whether logs may contain model output or runtime errors.
- Whether the runtime may execute tools, plugins, MCP servers, or arbitrary
  model-side code.

Default generated commands should bind local development servers to
`127.0.0.1` unless the operator explicitly opts into network exposure.

## Registry Integration

After a successful health/model-list check, ModelRouter should add registry
entries with provenance:

```json
{
  "model_id": "qwen3:4b",
  "source": "runtime_discovered",
  "runtime_id": "ollama",
  "runtime_mode": "external_managed",
  "backend": "fast",
  "endpoint": "http://127.0.0.1:11434/v1",
  "detected_at": "2026-06-30T12:00:00Z",
  "loaded": true,
  "install_state": "operator_managed"
}
```

Registry entries from runtime discovery should:

- Be advisory metadata, not route-policy mutations.
- Keep model id, backend, runtime id, endpoint, loaded state, and detection
  timestamp.
- Avoid prompt bodies, request bodies, secrets, response text, and raw logs.
- Be refreshed only through explicit scan/status actions or bounded settings
  state refreshes.
- Keep manually configured backend/model assignments authoritative until the
  operator changes them.

## LM Studio Connection

Mode: `external_managed`.

Official references:

- https://lmstudio.ai/download
- https://lmstudio.ai/docs/app
- https://lmstudio.ai/docs/cli

Recommended flow:

1. Detect whether port `1234` is open and whether a configured backend points to
   `http://127.0.0.1:1234/v1`.
2. If LM Studio is not detected, show the official download page. Do not run a
   GUI installer from ModelRouter.
3. Instruct the operator to install/open LM Studio, download/load a model there,
   and enable the local server.
4. Preview a config patch:

   ```yaml
   backends:
     fast:
       base_url: http://127.0.0.1:1234/v1
       model: <model-id-from-lm-studio>
   ```

5. After confirmation, write the backend config or update only the selected
   backend.
6. Verify with a bounded `/v1/models` check.
7. Register visible LM Studio models with `source: runtime_discovered`.

Platform notes:

- macOS, Windows, and Linux should use LM Studio's official app installer.
- Headless/server users may use LM Studio's documented daemon path, but that
  should be a separate advanced path because it changes the operator model.

Security notes:

- LM Studio owns model execution, downloads, loading, and chat.
- ModelRouter should not infer native LM Studio load/unload unless a stable
  local API/CLI contract is wired and tested.

Rollback:

- Remove or revert the ModelRouter backend config.
- Stop the LM Studio local server in LM Studio.
- Uninstall LM Studio through the OS/app mechanism if desired.
- Registry entries from LM Studio disappear on the next explicit scan/status
  refresh if the backend is removed or unreachable.

## Ollama Install Or Connect

Mode: `external_managed` by default; `external_cli` only for explicit supported
CLI actions such as `ollama list`, `ollama ps`, and confirmed `ollama stop`.

Official references:

- https://ollama.com/download
- https://ollama.com/download/mac
- https://ollama.com/download/windows
- https://github.com/ollama/ollama

Recommended flow:

1. Detect `ollama` on PATH and port `11434`.
2. If Ollama is installed but not reachable, suggest starting the app/service or
   running `ollama serve`.
3. If Ollama is missing, show official installer options:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   ```powershell
   irm https://ollama.com/install.ps1 | iex
   ```

   These commands must be previewed and confirmed before execution. The safer
   default in UI should be opening the official download page.

4. Preview a backend config:

   ```yaml
   backends:
     fast:
       base_url: http://127.0.0.1:11434/v1
       model: <ollama-model-tag>
   ```

5. Keep model pull separate. If the selected tag is missing, show:

   ```bash
   ollama pull <model-tag>
   ```

   Do not run it as part of connect, health, route, or proxy forwarding.

6. Verify using `ollama list` when available and/or a bounded local models API
   check.
7. Register Ollama models with source `runtime_discovered` or `ollama_cli`.

Platform notes:

- macOS: official app or install script.
- Windows: official installer or documented PowerShell install path.
- Linux: official install script, package-managed installs, or containerized
  deployments as operator choices.

Security notes:

- Model pulls download weights and may consume significant disk.
- The local server should remain bound locally unless the operator explicitly
  changes Ollama networking.

Rollback:

- Remove the ModelRouter backend config.
- Stop loaded models with `ollama stop <model>` when desired.
- Remove pulled models with operator-reviewed Ollama commands.
- Uninstall Ollama through the OS package/app mechanism.

## llama.cpp Server Install Or Configure

Mode: `external_cli` when ModelRouter owns a configured `runtime.command`;
otherwise `external_managed`.

Official references:

- https://github.com/ggml-org/llama.cpp
- https://github.com/ggml-org/llama.cpp/releases
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Recommended flow:

1. Detect `llama-server` on PATH and scan configured/local GGUF paths.
2. If missing, present official installation choices:
   - Download a release artifact.
   - Build from source with the platform-appropriate CMake instructions.
   - Use a package manager only when the operator selects it.
3. Require the operator to choose a GGUF model path. Do not download weights
   inside the runtime install flow.
4. Preview a managed runtime config:

   ```yaml
   backends:
     code:
       base_url: http://127.0.0.1:8093/v1
       model: local-code.gguf
       runtime:
         enabled: true
         kind: llama-server
         command:
           - llama-server
           - -m
           - /models/local-code.gguf
           - --port
           - "8093"
         readiness_url: http://127.0.0.1:8093/health
         idle_timeout_seconds: 900
   ```

5. On confirmation, write only the selected backend/runtime stanza.
6. Start/stop only through explicit runtime actions or proxy-managed runtime
   startup, never through routing decisions.
7. Register the configured GGUF and any visible runtime model id.

Platform notes:

- macOS: Metal builds are common; Homebrew or source builds are operator
  choices.
- Windows: prebuilt releases or source builds may require Visual Studio, CMake,
  Vulkan SDK, or related tooling.
- Linux: source builds or distro/container packaging may require compiler,
  CMake, CUDA/ROCm/Vulkan, or device permissions.

Security notes:

- Managed runtime commands execute local binaries. Show argv exactly.
- Do not accept arbitrary shell strings; keep argv arrays.
- Warn before exposing `--host 0.0.0.0`.

Rollback:

- Stop the managed process.
- Remove or disable the backend `runtime` stanza.
- Remove local binaries or build directories manually if desired.
- Keep GGUF model files unless the operator explicitly deletes them.

## MLX-LM Install Or Configure On Apple Silicon

Mode: `external_cli`.

Official references:

- https://github.com/ml-explore/mlx-lm
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
- https://huggingface.co/docs/hub/en/mlx

Recommended flow:

1. Detect macOS Apple Silicon and Python environment.
2. If not Apple Silicon, show MLX-LM as unsupported or advanced only.
3. Preview Python install into the active virtual environment:

   ```bash
   python -m pip install --upgrade mlx-lm
   ```

4. Require operator selection of an existing local MLX model path or explicit
   Hugging Face model id.
5. Warn that `mlx_lm.server --model <hf-repo>` may download from Hugging Face if
   the model is not already cached. Prefer a separate `hf download` plan when
   ModelRouter is managing downloads.
6. Preview a managed runtime config:

   ```yaml
   backends:
     balanced:
       base_url: http://127.0.0.1:8088/v1
       model: mlx-local-balanced
       runtime:
         enabled: true
         kind: mlx-lm
         command:
           - mlx_lm.server
           - --model
           - /models/mlx/balanced
           - --port
           - "8088"
         readiness_url: http://127.0.0.1:8088/v1/models
   ```

7. After confirmation, write config and verify with `/v1/models`.
8. Register configured MLX model paths and visible runtime models.

Platform notes:

- macOS Apple Silicon is the primary supported path.
- Windows and Linux should not be offered this path unless a future adapter has
  tested support for the relevant MLX backend.

Security notes:

- MLX-LM's server is suitable for local development, not a hardened production
  service. Keep generated configs bound to localhost.
- Any Hugging Face download remains a separate confirmed download action.

Rollback:

- Stop the managed process.
- Remove or disable the backend runtime stanza.
- Uninstall `mlx-lm` from the virtual environment if desired.
- Delete model cache/files only after operator confirmation.

## LocalAI Install Or Connect

Mode: `external_managed` by default.

Official references:

- https://localai.io/
- https://localai.io/installation/
- https://localai.io/basics/container/

Recommended flow:

1. Detect configured LocalAI-shaped backends and local port `8080` where
   relevant.
2. Prefer connect-first when LocalAI already runs.
3. If installation is requested, show official choices:
   - Containers with Docker or Podman.
   - macOS DMG.
   - Linux binary install.
   - Kubernetes for server/team deployments.
4. Preview container command examples:

   ```bash
   docker run -p 8080:8080 --name local-ai -ti localai/localai:latest
   ```

   ```bash
   podman run -p 8080:8080 --name local-ai -ti localai/localai:latest
   ```

5. Require the operator to manage model files/volumes according to LocalAI's
   docs. ModelRouter should not invent LocalAI model layout.
6. Preview backend config:

   ```yaml
   backends:
     fast:
       base_url: http://127.0.0.1:8080/v1
       model: <localai-model-id>
   ```

7. Verify with `/v1/models` and register model ids.

Platform notes:

- Containers work across macOS, Windows, and Linux when Docker/Podman is
  installed.
- Windows users may need Docker Desktop or WSL; do not assume either.
- Linux server users may prefer systemd, Podman, Docker, or Kubernetes.

Security notes:

- Container volume mounts can expose model directories and config files. Preview
  mount paths before execution.
- GPU acceleration often requires extra device flags and driver setup.
- Network binding should default to local ports unless the operator chooses a
  remote/server deployment.

Rollback:

- Remove ModelRouter backend config.
- Stop/remove the LocalAI container or service with an operator-reviewed command.
- Preserve mounted model volumes unless explicitly deleted.

## vLLM Install Or Connect

Mode: `external_managed` for existing servers; `external_cli` only for advanced
operator-managed local/server commands.

Official references:

- https://docs.vllm.ai/en/latest/getting_started/quickstart/
- https://docs.vllm.ai/en/stable/serving/online_serving/

Recommended flow:

1. Treat vLLM as advanced/server-oriented. It often depends on GPU drivers,
   CUDA/ROCm, Python environment isolation, and deployment topology.
2. Prefer connect-first for existing vLLM deployments.
3. If local install is requested, preview an isolated-environment install plan,
   not a system Python mutation:

   ```bash
   python -m venv .venv-vllm
   . .venv-vllm/bin/activate
   python -m pip install --upgrade vllm
   ```

4. Preview server start:

   ```bash
   vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000
   ```

   If an API key is used, show only that a key is configured, not the value.

5. Preview backend config:

   ```yaml
   backends:
     reasoning:
       base_url: http://127.0.0.1:8000/v1
       model: Qwen/Qwen2.5-1.5B-Instruct
   ```

6. Verify with `/v1/models` and register the served model.

Platform notes:

- Linux GPU servers are the most common vLLM target.
- macOS and Windows should be connect-first unless a future tested path exists.
- Container/Kubernetes deployments should be represented as connect flows unless
  ModelRouter has an explicit deployment integration.

Security notes:

- vLLM may bind to network interfaces and serve high-value models. Default to
  `127.0.0.1`.
- Surface API key requirements and never log/display key values.
- Warn about GPU memory pressure, model downloads, and shared-server access.

Rollback:

- Stop the vLLM process/container/deployment outside ModelRouter unless it was
  explicitly started by a future adapter action.
- Remove ModelRouter backend config.
- Remove the dedicated virtual environment if it was created.
- Keep model caches unless explicitly deleted.

## Config Writes

Every config write should be shown as a diff before execution:

```diff
+ backends:
+   fast:
+     base_url: http://127.0.0.1:11434/v1
+     model: qwen3:4b
+ engine_backends:
+   fast_local: fast
```

The write path should:

- Refuse to overwrite existing config unless the operator confirms the exact
  file and diff.
- Back up changed config files.
- Write a local audit row in a maintenance log.
- Tell the operator whether a proxy restart is needed.
- Avoid changing routing mode unless the operator selected that change.

## MVP CLI

The first guided assistant slice should be explicit and preview-first:

```bash
model-router runtimes status --json
model-router runtimes doctor
model-router runtimes connect lmstudio
model-router runtimes connect ollama
model-router runtimes connect llamacpp --endpoint http://127.0.0.1:8080/v1
```

`connect` is preview-only by default. A backend config patch is written only
when the operator passes both `--write` and `--yes`. This MVP should cover LM
Studio connect instructions and health, Ollama connect/install guidance,
llama.cpp server configure guidance, and safe updates to an existing backend.
It should not install runtimes, pull models, create new backends, or start
servers.

## Suggested Future Commands

These command names remain design placeholders for later installer expansion:

```bash
model-router runtimes install-plan --runtime mlx-lm --json
model-router runtimes install --runtime mlx-lm --execute --yes
model-router runtimes refresh-registry --backend fast
```

The existing `model-router install` command can also include runtime-specific
next actions, but it should remain plan-only by default. Any future execution
path should be a separate explicit follow-up command or a clearly confirmed
action.

## UI Shape

Settings/control center should present runtime setup as compact operational
rows:

- Runtime name and mode.
- Detected/install status.
- Endpoint and health.
- Capability summary.
- Next action: connect, start, configure, install guide, or view details.
- Buttons for preview, copy command, open official installer, write config, and
  refresh registry.

Expanded details can show command previews, config diffs, security notes,
rollback notes, and official docs links. The main dashboard should still
prioritize proxy status, routing mode, active backend/model, telemetry health,
and safety/policy state.

## Test Expectations For A Future Implementation

- `route_fast(...)`, `route(...)`, and proxy forwarding do not import or call
  runtime installer modules.
- Install plans are deterministic and JSON-safe.
- Mutating actions require confirmation.
- CLI install commands are preview-only by default.
- Missing runtimes are guidance, not errors, unless explicitly selected.
- Config diffs redact secrets.
- Registry refreshes do not store prompts, request bodies, secrets, or response
  text.
- Rollback instructions are available for every mutating action.
