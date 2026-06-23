"""Backward-compatible package path for ModelRouter.

New custom-agent integrations should prefer ``import model_router``.
"""

from hermes.plugins.model_router.dispatch import build_dispatch_plan
from hermes.plugins.model_router.policy import ModelRouter, route_prompt
from hermes.plugins.model_router.profiles import RoutingProfile
from hermes.plugins.model_router.scorer import score_prompt

__all__ = [
    "ModelRouter",
    "RoutingProfile",
    "build_dispatch_plan",
    "route_prompt",
    "score_prompt",
]
