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
git tag v0.6.2
git push origin v0.6.2
```

Create a GitHub release for the tag. The publish workflow uses PyPI trusted
publishing and runs when the GitHub release is published.

## Release Notes Template

```markdown
## ModelRouter v0.6.2

### Highlights
- First-run proxy setup polish.
- `model-router init --auto` preset selection from local Ollama/LM Studio signals.
- `model-router doctor` remediation for upstream startup, missing models, and telemetry.

### Verification
- Ruff: passed
- Pytest: <paste summary>
- route_fast latency: <paste JSON or mean/best values>

### Install
pip install "hermes-router[proxy]"
```
