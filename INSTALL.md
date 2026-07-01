# Install ModelRouter

ModelRouter is installed from the PyPI distribution `hermes-router`. The public
product and commands are ModelRouter, `model-router`, and `model-router-proxy`.

For the full product overview, see [README.md](README.md). For runtime setup
boundaries, see [Runtime install flow](docs/runtime-install-flow.md). For
upgrade, rollback, and uninstall guidance, see
[Upgrade and uninstall](docs/upgrade-uninstall.md).

## Install

ModelRouter requires Python 3.11 or newer. The normal agent/proxy install path
is:

```bash
python -m pip install "hermes-router[proxy]"
```

If you use uv-managed environments, run the same package install through uv:

```bash
uv pip install "hermes-router[proxy]"
```

If you prefer an isolated CLI app install, pipx also works:

```bash
pipx install "hermes-router[proxy]"
```

This installs:

- `model-router`: the CLI for setup, config, routing diagnostics, settings, and
  maintenance.
- `model-router-proxy`: the local OpenAI-compatible `/v1` proxy.
- `model_router`: the importable Python API.

## First Run

Start with the installer plan:

```bash
model-router install --quick
```

By default, `model-router install` is plan-only in the current public preview.
It detects your install method, Python, command availability, optional
dependencies, config files, ports, and local runtime signals, then prints
explicit next commands. It does not mutate by default: it does not install
dependencies, download models, write configs, install services, enable hosted
providers, change routing, or start runtimes. Passing `--yes` records
confirmation intent in the plan; it does not turn this command into execution.

For automation or support logs, use JSON:

```bash
model-router install --quick --json
```

For a guided first run, use:

```bash
model-router install --guided
```

Guided mode reuses the same installer plan and asks before running any selected
follow-up command. It can run first-run config creation and doctor checks after
confirmation, but it still does not install runtimes, pull models, overwrite
existing config, start services, enable hosted providers, or change routing.

Create first-run configs:

```bash
model-router init --auto --yes
```

Start the local routing proxy:

```bash
model-router-proxy --config ~/.model-router/routing_proxy.yaml
```

Point OpenAI-compatible agents or clients at:

```text
http://127.0.0.1:8082/v1
```

Useful checks:

```bash
model-router validate-proxy-config --config ~/.model-router/routing_proxy.yaml
model-router doctor --config ~/.model-router/routing_proxy.yaml
curl http://127.0.0.1:8082/health
```

## Troubleshooting

### Missing Proxy Dependencies

If `model-router-proxy` or `model-router settings` reports missing FastAPI,
httpx, or uvicorn, reinstall with the proxy extra:

```bash
python -m pip install --upgrade "hermes-router[proxy]"
```

From a local checkout, you can also review and run the explicit prerequisite
plan:

```bash
model-router setup install-prereqs --preset proxy
model-router setup install-prereqs --preset proxy --execute --yes
```

In pipx installs, use `pipx inject hermes-router <package>` or reinstall with
extras instead of mutating the app interpreter directly. If `python -m pip` is
missing in a uv-created environment, the prerequisite plan suggests
`uv pip install --python <python> ...`.

### Missing Runtime

ModelRouter is a routing/control plane, not an inference runtime. If no local
runtime is running, start or install one explicitly, such as LM Studio, Ollama,
llama.cpp, MLX-LM, LocalAI, vLLM, or another OpenAI-compatible server. Then run:

```bash
model-router doctor --config ~/.model-router/routing_proxy.yaml
```

Runtime install/connect guidance is preview-first and confirmation-gated. See
[Runtime install flow](docs/runtime-install-flow.md).

### Existing Config

If `~/.model-router/routing_proxy.yaml` already exists, `model-router install`
will plan validation instead of overwrite. Use:

```bash
model-router doctor --config ~/.model-router/routing_proxy.yaml
model-router settings --config-dir ~/.model-router
```

Back up configs before manual edits:

```bash
cp ~/.model-router/routing_proxy.yaml ~/.model-router/routing_proxy.yaml.bak
```

### Uninstall

Stop any running proxy or settings process first, then uninstall the package:

```bash
python -m pip uninstall hermes-router
```

For pipx installs:

```bash
pipx uninstall hermes-router
```

Local config, telemetry, feedback, benchmarks, and logs are left in
`~/.model-router` unless you intentionally remove them:

```bash
rm -rf ~/.model-router
```
