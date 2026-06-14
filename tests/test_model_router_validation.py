"""Config and input-validation tests.

Covers three defects:

- ``scoring.weights.confidence`` was validated and documented but ignored.
- An explicit empty ``modalities: []`` was overwritten by keyword defaults.
- Invalid ``max_cost_tier`` / ``max_latency_tier`` hints were silently ignored
  (fail-open) instead of rejected.
"""

from __future__ import annotations

import pytest

from hermes.plugins.model_router.models import ModelEngine, RoutingHints, ScoringConfig
from hermes.plugins.model_router.policy import route_prompt
from hermes.plugins.model_router.scorer import score_prompt


# --- #3 confidence weight overrides -----------------------------------------


def test_confidence_weight_override_is_applied():
    base = score_prompt("fix this").confidence_score
    overridden = score_prompt(
        "fix this",
        scoring_config=ScoringConfig(weights={"confidence": {"ambiguous": 0}}),
    ).confidence_score
    # "fix this" is ambiguous, so dropping the ambiguous penalty to 0 must raise
    # confidence by exactly the default penalty (25).
    assert overridden == base + 25


def test_default_confidence_is_unchanged():
    # Locks the default behaviour: ambiguous (-25) + weak feature match (-10).
    assert score_prompt("fix this").confidence_score == 55


def test_empty_prompt_confidence_weight_override_is_applied():
    base = score_prompt("").confidence_score
    overridden = score_prompt(
        "",
        scoring_config=ScoringConfig(weights={"confidence": {"empty_prompt": 0}}),
    ).confidence_score
    assert overridden > base


# --- #4 explicit empty modalities -------------------------------------------


def _vision_engine_data(**overrides):
    data = {
        "provider": "local",
        "model": "img-model",
        "adapter": "local_vision",
        "strengths": ["image description", "ocr"],
        "max_context": 8192,
        "cost_tier": "low",
        "latency_tier": "low",
        "enabled": True,
    }
    data.update(overrides)
    return data


def test_explicit_empty_modalities_is_honored():
    engine = ModelEngine.from_dict("vision_probe", _vision_engine_data(modalities=[]))
    assert engine.modalities == ()


def test_absent_modalities_falls_back_to_keyword_default():
    engine = ModelEngine.from_dict("vision_probe", _vision_engine_data())
    assert engine.modalities == ("image",)


def test_explicit_modalities_are_kept():
    engine = ModelEngine.from_dict(
        "vision_probe", _vision_engine_data(modalities=["pdf"])
    )
    assert engine.modalities == ("pdf",)


# --- #6 tier hint validation ------------------------------------------------


@pytest.mark.parametrize("key", ["max_cost_tier", "max_latency_tier"])
def test_unknown_tier_hint_is_rejected(key):
    with pytest.raises(ValueError):
        RoutingHints.from_dict({key: "cheap"})


def test_valid_tier_hints_are_accepted():
    hints = RoutingHints.from_dict(
        {"max_cost_tier": "medium", "max_latency_tier": "low"}
    )
    assert hints.max_cost_tier == "medium"
    assert hints.max_latency_tier == "low"


def test_invalid_tier_hint_fails_closed_through_route():
    decision = route_prompt("rewrite this text", hints={"max_cost_tier": "cheap"})
    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
