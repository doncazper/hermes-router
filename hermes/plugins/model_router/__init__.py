"""Deterministic model routing decisions for Hermes."""

from hermes.plugins.model_router.dispatch import build_dispatch_plan
from hermes.plugins.model_router.policy import ModelRouter, route_prompt
from hermes.plugins.model_router.scorer import score_prompt

__all__ = ["ModelRouter", "build_dispatch_plan", "route_prompt", "score_prompt"]
