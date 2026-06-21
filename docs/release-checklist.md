# Release Checklist

Use this checklist for each PyPI release.

## Before Tagging

```bash
python -m pip install -e ".[dev,proxy,release]"
python -m ruff check .
python -m pytest
python scripts/check_route_fast_latency.py --json
python -m build
python -m twine check dist/*
```

Record the test summary and route-fast benchmark output in the GitHub release
notes.

## Tag And Release

```bash
git tag v0.5.0
git push origin v0.5.0
```

Create a GitHub release for the tag. The publish workflow uses PyPI trusted
publishing and runs when the GitHub release is published.

## Release Notes Template

```markdown
## Hermes Router v0.5.0

### Highlights
- Usable local OpenAI-compatible routing proxy beta.
- First-run `model-router init`.
- Provider presets and proxy health diagnostics.

### Verification
- Ruff: passed
- Pytest: <paste summary>
- route_fast latency: <paste JSON or mean/best values>

### Install
pip install "hermes-router[proxy]"
```
