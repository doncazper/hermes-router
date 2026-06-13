"""Deterministic model routing decisions for Hermes."""

from hermes.plugins.model_router.policy import route_prompt
from hermes.plugins.model_router.scorer import score_prompt

__all__ = ["route_prompt", "score_prompt"]
