from hermes.plugins.model_router.models import ScoringConfig
from hermes.plugins.model_router.scorer import score_prompt


def test_simple_rewrite_has_low_complexity_and_risk():
    analysis = score_prompt("rewrite this text")

    assert analysis.complexity_score.value < 35
    assert analysis.risk_score.value < 20
    assert analysis.confidence_score >= 70
    assert analysis.features.simple_transform is True
    assert analysis.features.requires_confirmation is False


def test_medical_legal_and_financial_prompts_increase_risk():
    analysis = score_prompt(
        "Review this medical diagnosis and legal settlement for financial liability."
    )

    assert analysis.risk_score.value >= 25
    assert analysis.features.sensitive_domain is True
    assert any("sensitive" in reason for reason in analysis.reasons)


def test_long_prompts_increase_complexity():
    short = score_prompt("summarize this")
    long = score_prompt("architecture planning details " * 250)

    assert long.complexity_score.value > short.complexity_score.value
    assert long.complexity_score.value >= 60
    assert long.features.long_context is True


def test_short_destructive_prompts_require_confirmation():
    analysis = score_prompt("delete all my emails")

    assert analysis.risk_score.value >= 70
    assert analysis.features.external_action is True
    assert analysis.features.destructive_action is True
    assert analysis.features.requires_confirmation is True


def test_short_high_impact_external_actions_require_confirmation():
    prompts = (
        "deploy to production",
        "merge this pull request",
        "push to main",
        "schedule a meeting",
        "apply for this job",
    )

    for prompt in prompts:
        analysis = score_prompt(prompt)

        assert analysis.features.external_action is True
        assert analysis.features.requires_confirmation is True
        assert analysis.risk_score.value >= 70


def test_vision_prompts_set_multimodal_features():
    analysis = score_prompt("Extract the text from this screenshot and describe the chart.")

    assert analysis.features.vision_intent is True
    assert analysis.features.requires_vision is True
    assert analysis.features.requires_tools is True
    assert any("vision" in reason for reason in analysis.reasons)


def test_image_generation_prompts_set_generation_features():
    analysis = score_prompt("Generate an image of a clean ModelRouter dashboard.")

    assert analysis.features.image_generation_intent is True
    assert analysis.features.requires_image_generation is True
    assert analysis.features.requires_tools is True
    assert any("image generation" in reason for reason in analysis.reasons)


def test_weighted_scoring_saturates_and_preserves_zero_to_one_hundred_scale():
    prompt = (
        "Design a distributed architecture with backpressure, consensus, "
        "step-by-step rollout, and tradeoff analysis."
    )
    analysis = score_prompt(
        prompt,
        scoring_config=ScoringConfig.from_dict(
            {
                "weights": {
                    "complexity": {
                        "multi_step_reasoning": 40,
                        "architecture": 35,
                    }
                },
                "saturation_k": 25,
            }
        ),
    )

    assert 0 <= analysis.complexity_score.value <= 100
    assert analysis.complexity_score.value >= 80
    assert analysis.risk_score.value == 0


def test_scoring_config_weight_overrides_change_scores_deterministically():
    prompt = "Design a multi-step architecture plan."
    default = score_prompt(prompt)
    boosted = score_prompt(
        prompt,
        scoring_config=ScoringConfig.from_dict(
            {
                "weights": {
                    "complexity": {
                        "multi_step_reasoning": 30,
                    }
                }
            }
        ),
    )

    assert boosted.complexity_score.value > default.complexity_score.value
