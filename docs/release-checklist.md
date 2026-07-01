# Release Checklist

Use this checklist for every PyPI release. The product name is ModelRouter; the
current PyPI install spec is kept only where release commands need it.

## Before Tagging

```bash
python -m pip install -e ".[dev,proxy,tui,release]"
python -m ruff check .
python -m pytest
python scripts/check_route_fast_latency.py --json
python -m build
python -m twine check dist/*
```

## Fresh Install Smoke

Run this from the repository root before tagging when practical. It creates a
temporary virtual environment outside the checkout, installs the local package
non-editably with proxy extras, checks the console scripts, verifies the public
Python import, and runs the plan-only installer command.

```bash
ROOT="$(pwd)"
SMOKE_DIR="$(mktemp -d)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
trap 'rm -rf "$SMOKE_DIR"' EXIT

"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
"$PYTHON_BIN" -m venv "$SMOKE_DIR/venv"
"$SMOKE_DIR/venv/bin/python" -m pip install --upgrade pip
"$SMOKE_DIR/venv/bin/python" -m pip install "${ROOT}[proxy]"

cd "$SMOKE_DIR"
"$SMOKE_DIR/venv/bin/model-router" --help >/dev/null
"$SMOKE_DIR/venv/bin/model-router-proxy" --help >/dev/null
"$SMOKE_DIR/venv/bin/python" -c "import model_router; print(model_router.ModelRouter.__name__)"
"$SMOKE_DIR/venv/bin/model-router" install --quick --config-dir "$SMOKE_DIR/config" --json
```

On Windows, adapt the virtualenv paths and activation/command locations for
PowerShell or Command Prompt.

Record the test summary, route-fast benchmark output, and fresh install smoke
status in the GitHub release notes.

## Maturity Gate

Confirm maturity labels in `model-router doctor`, `model-router settings`, and
`model-router tui`:

- Basic router mode: beta.
- Installer onboarding: beta.
- Model library: beta.
- Runtime adapters: beta.
- TUI control center: experimental.
- Compatibility endpoints: beta.

Run plan-only dogfood first:

```bash
model-router dogfood proxy --config ~/.model-router/routing_proxy.yaml
```

Run live local dogfood only against a proxy/backend setup you intend to test:

```bash
model-router dogfood proxy --config ~/.model-router/routing_proxy.yaml --execute
```

Before a v0.7.x release, dogfood both routing modes:

- Decision mode: current default config, real local backend, chat/models smoke.
- Manual/basic mode: explicit `proxy.default_backend` and `proxy.default_model`.

Experimental failures must not break stable decision-mode proxy routing. If a
beta/experimental surface fails, either fix it or document the residual risk in
release notes before tagging.

## Upgrade And Rollback

Review [Upgrade And Uninstall](upgrade-uninstall.md) before release notes are
published. Include any config migration notes for new fields or defaults.

## Tag And Release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Create a GitHub release for the tag. The publish workflow uses PyPI trusted
publishing and runs when the GitHub release is published.

## Release Notes Template

````markdown
## ModelRouter vX.Y.Z

### Highlights
- <major user-facing changes>
- <maturity changes, if any>
- <compatibility or proxy changes, if any>

### Maturity
- Stable/default path: decision-mode OpenAI-compatible proxy routing.
- Beta surfaces: basic/manual router mode, installer, model library, runtime
  adapters, compatibility endpoints.
- Experimental surfaces: TUI control center.

### Verification
- Ruff: passed
- Pytest: <paste summary>
- route_fast latency: <paste JSON or mean/best values>
- Fresh install smoke: passed
- Dogfood: <decision/manual plan or execute status>

### Install

```bash
python -m pip install "hermes-router[proxy]"
```
````
