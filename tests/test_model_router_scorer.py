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


def test_vision_prompts_set_multimodal_features():
    analysis = score_prompt("Extract the text from this screenshot and describe the chart.")

    assert analysis.features.vision_intent is True
    assert analysis.features.requires_vision is True
    assert analysis.features.requires_tools is True
    assert any("vision" in reason for reason in analysis.reasons)


def test_image_generation_prompts_set_generation_features():
    analysis = score_prompt("Generate an image of a clean Hermes router dashboard.")

    assert analysis.features.image_generation_intent is True
    assert analysis.features.requires_image_generation is True
    assert analysis.features.requires_tools is True
    assert any("image generation" in reason for reason in analysis.reasons)
