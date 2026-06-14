"""Scorer precision tests for the authoritative route() path.

Covers three detection-precision defects:

- Plural PII terms ("passwords", "api keys") dodged the singular-only regex.
- ``image_generation_intent`` used a greedy ``.*`` so a generation verb anywhere
  before an image noun matched ("generate a summary of this image's metadata").
- "order" as a sorting noun ("in alphabetical order") was treated as a purchase.
"""

from __future__ import annotations

from hermes.plugins.model_router.scorer import score_prompt


def _pii_fired(prompt: str) -> bool:
    reasons = score_prompt(prompt).risk_score.reasons
    return any("credential" in reason for reason in reasons)


# --- plural PII --------------------------------------------------------------


def test_singular_pii_is_detected():
    assert _pii_fired("rotate the password and api key") is True


def test_plural_pii_is_detected():
    assert _pii_fired("rotate the passwords and api keys") is True


# --- image-generation proximity ---------------------------------------------


def test_immediate_image_generation_is_detected():
    assert score_prompt("generate image").features.image_generation_intent is True


def test_real_image_generation_is_detected():
    features = score_prompt("Generate an image of a dashboard.").features
    assert features.image_generation_intent is True


def test_image_metadata_is_not_image_generation():
    features = score_prompt("generate a summary of this image's metadata").features
    assert features.image_generation_intent is False
    # It is still a vision/image-understanding task.
    assert features.requires_vision is True


def test_design_company_image_is_not_image_generation():
    features = score_prompt("help me design a better company image").features
    assert features.image_generation_intent is False


def test_diffusion_is_image_generation():
    assert score_prompt("run stable diffusion locally").features.image_generation_intent


# --- "order" noun vs purchase verb ------------------------------------------


def test_alphabetical_order_is_not_a_purchase():
    features = score_prompt("put these names in alphabetical order").features
    assert features.purchase_action is False
    assert features.requires_confirmation is False


def test_in_order_to_is_not_a_purchase():
    features = score_prompt("refactor the code in order to improve speed").features
    assert features.purchase_action is False


def test_order_food_is_still_a_purchase():
    assert score_prompt("order a pizza for the team").features.purchase_action is True


def test_buy_is_still_a_purchase():
    assert score_prompt("buy two monitors").features.purchase_action is True
