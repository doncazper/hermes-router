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
git tag v0.5.4
git push origin v0.5.4
```

Create a GitHub release for the tag. The publish workflow uses PyPI trusted
publishing and runs when the GitHub release is published.

## Release Notes Template

```markdown
## ModelRouter v0.5.4

### Highlights
- Docs refresh for the PyPI long description.
- LM Studio and Ollama setup examples.
- First-run transcripts and wrong-route regression workflow.

### Verification
- Ruff: passed
- Pytest: <paste summary>
- route_fast latency: <paste JSON or mean/best values>

### Install
pip install "hermes-router[proxy]"
```
