# Release Checklist

Use this checklist for every PyPI release. The distribution name remains
`hermes-router`; the product name is ModelRouter.

## Before Tagging

```bash
python -m pip install -e ".[dev,proxy,tui,release]"
python -m ruff check .
python -m pytest
python scripts/check_route_fast_latency.py --json
python -m build
python -m twine check dist/*
```

Record the test summary and route-fast benchmark output in the GitHub release
notes.

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

```markdown
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
- Dogfood: <decision/manual plan or execute status>

### Install
pip install "hermes-router[proxy]"
```
