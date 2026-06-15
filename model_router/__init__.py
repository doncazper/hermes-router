"""Generic public API for fast model routing in custom agents.

The historical implementation lives under ``hermes.plugins.model_router`` for
backward compatibility. New integrations should import from ``model_router``.
"""

from hermes.plugins.model_router import (
    ModelRouter,
    build_dispatch_plan,
    route_prompt,
    score_prompt,
)

__all__ = ["ModelRouter", "build_dispatch_plan", "route_prompt", "score_prompt"]
