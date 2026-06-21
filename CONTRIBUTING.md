# Contributing

Thanks for helping make Hermes Router useful for custom-agent builders.

## Project Direction

Hermes Router should stay:

- Generic: usable by any Python agent, local tool, or hosted service.
- Fast: `ModelRouter.route_fast(...)` is the production hot path.
- Deterministic: no LLM call is required to classify a prompt.
- Safe: risky or invalid requests fail closed to `human_confirm`.
- Host-neutral: adapters belong at the edge, not in the scoring policy.

## Development Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Test Loops

Use the smallest loop that matches the work while iterating, then run the full
loop before committing.

Fast routing/calibration loop:

```bash
python -m pytest \
  tests/test_model_router_parity.py \
  tests/test_model_router_replay.py \
  tests/test_model_router_golden.py
python scripts/check_route_fast_latency.py --json
```

Proxy/product loop:

```bash
python -m pytest \
  tests/test_model_router_proxy.py \
  tests/test_model_router_proxy_config.py \
  tests/test_model_router_product.py
```

Full pre-commit loop:

```bash
python -m pytest
python -m ruff check .
python scripts/check_route_fast_latency.py --json
```

Release loop:

```bash
python -m ruff check .
python -m pytest
python scripts/check_route_fast_latency.py --json
python -m build
python -m twine check dist/*
```

## Compatibility

New integrations should import from `model_router`:

```python
from model_router import ModelRouter
```

The older `hermes.plugins.model_router` package path remains for backward
compatibility.

## Adapter Contributions

Adapter examples are welcome when they stay thin. A good adapter:

- Calls `route_fast(prompt)` once per turn.
- Maps the returned engine to the host app's model/runtime config.
- Handles `human_confirm` outside the router.
- Does not add provider calls, prompt logging, or setup scans to the hot path.
