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
