import json
from pathlib import Path

import yaml

from hermes.plugins.model_router.config import REQUIRED_ENGINE_CATEGORIES
from hermes.plugins.model_router.policy import route_prompt
from hermes.plugins.model_router.receipts import decision_to_receipt, receipt_to_json


def _engine(name: str, *, fallback: str | None = None) -> dict:
    return {
        "provider": "local" if name != "human_confirm" else "human",
        "model": f"{name}-model",
        "adapter": name,
        "strengths": [name],
        "max_context": 8192,
        "cost_tier": "low",
        "latency_tier": "low",
        "enabled": True,
        "fallback": fallback,
        "supports_tools": name
        in {
            "code_agent",
            "web_research",
            "multimodal_vision",
            "image_generation",
            "human_confirm",
        },
        "modalities": ["image"] if name == "multimodal_vision" else [],
    }


def _policy_config_path(tmp_path: Path) -> Path:
    fallbacks = {
        "intent_router": "fast_local",
        "fast_local": "balanced_local",
        "balanced_local": "reasoning_local",
        "reasoning_local": "human_confirm",
        "code_agent": "reasoning_local",
        "web_research": "reasoning_local",
        "multimodal_vision": "reasoning_local",
        "image_generation": "human_confirm",
        "human_confirm": None,
    }
    engines = {
        name: _engine(name, fallback=fallbacks[name])
        for name in REQUIRED_ENGINE_CATEGORIES
    }
    engines["hosted_reasoning"] = _engine(
        "hosted_reasoning",
        fallback="reasoning_local",
    ) | {"provider": "openai", "adapter": "openai"}
    path = tmp_path / "model_router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "routing_targets": {
                    "simple": "fast_local",
                    "balanced": "balanced_local",
                    "reasoning": "hosted_reasoning",
                    "coding": "code_agent",
                    "research": "web_research",
                    "vision": "multimodal_vision",
                    "image_generation": "image_generation",
                    "confirmation": "human_confirm",
                },
                "provider_policy": {
                    "version": 1,
                    "provider_denylist": ["openai"],
                },
                "engines": engines,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_receipts_serialize_to_json():
    decision = route_prompt("rewrite this text")
    receipt = decision_to_receipt(decision)

    payload = json.loads(receipt_to_json(receipt))

    assert payload["selected_engine"] == "fast_local"
    assert payload["routing_profile"] == "balanced"
    assert payload["requirements"]["routing_profile"] == "balanced"
    assert "complexity_score" in payload
    assert "risk_score" in payload
    assert "reasons" in payload
    assert payload["summary"].startswith("Selected fast_local")
    assert "route.simple" in payload["reason_codes"]
    assert "selected_route_explanation" in payload
    assert "policy_explanation" in payload
    assert "rejection_explanation" in payload
    assert "fallback_explanation" in payload
    assert "safety_explanation" in payload
    assert "privacy_explanation" in payload
    assert "wrong_route_next_action" in payload


def test_receipt_fields_match_routing_decision():
    decision = route_prompt("research latest AI safety regulations with citations")
    receipt = decision_to_receipt(decision)

    assert receipt.selected_engine == decision.selected_engine
    assert receipt.fallback_engine == decision.fallback_engine
    assert receipt.requires_freshness == decision.requires_freshness
    assert receipt.requires_tools == decision.requires_tools
    assert receipt.requires_code_execution == decision.requires_code_execution
    assert receipt.requires_vision == decision.requires_vision
    assert receipt.requires_image_generation == decision.requires_image_generation
    assert receipt.availability_valid == decision.availability_valid
    assert receipt.availability_reasons == decision.availability_reasons
    assert receipt.requirements == decision.requirements
    assert receipt.rejected_engines == decision.rejected_engines
    assert receipt.alternatives == decision.alternatives


def test_receipt_does_not_serialize_raw_prompt():
    prompt = "delete all my emails"
    decision = route_prompt(prompt)
    receipt = decision_to_receipt(decision)

    serialized = receipt_to_json(receipt)

    assert prompt not in serialized
    assert "prompt" not in json.loads(serialized)


def test_receipt_reason_codes_for_common_code_route():
    decision = route_prompt("fix the repo and run tests")
    receipt = decision_to_receipt(decision)

    assert receipt.selected_engine == "code_agent"
    assert "route.coding" in receipt.reason_codes
    assert "requirement.tools" in receipt.reason_codes
    assert "requirement.code_execution" in receipt.reason_codes
    assert "safety.no_confirmation_required" in receipt.reason_codes
    assert "coding or repository work" in receipt.selected_route_explanation


def test_receipt_explains_human_confirmation():
    decision = route_prompt("delete all my emails")
    receipt = decision_to_receipt(decision)

    assert receipt.selected_engine == "human_confirm"
    assert "route.confirmation" in receipt.reason_codes
    assert "safety.confirmation_required" in receipt.reason_codes
    assert receipt.safety_explanation == "Human confirmation is required before dispatch."
    assert "human confirmation required" in receipt.summary


def test_receipt_explains_private_local_only_profile():
    decision = route_prompt(
        "research current routing approaches",
        hints={"profile": "private"},
    )
    receipt = decision_to_receipt(decision)

    assert receipt.routing_profile == "private"
    assert "policy.local_only" in receipt.reason_codes
    assert "Local-only routing is active" in receipt.privacy_explanation
    assert "Allowed providers: local, human" in receipt.policy_explanation


def test_receipt_serializes_constraints_and_rejections():
    decision = route_prompt("summarize this attachment", hints={"attachments": ["image"]})
    receipt = decision_to_receipt(decision)

    payload = json.loads(receipt_to_json(receipt))

    assert payload["requirements"]["required_modalities"] == ["image"]
    assert "rejected_engines" in payload


def test_receipt_serializes_ranked_alternatives():
    decision = route_prompt("rewrite this text")
    receipt = decision_to_receipt(decision)

    payload = json.loads(receipt_to_json(receipt))

    assert payload["alternatives"]
    assert {
        "engine",
        "rank_score",
        "capability",
        "trust",
        "cost",
        "latency",
        "reasons",
    } <= set(payload["alternatives"][0])


def test_receipt_explains_provider_policy_rejections(tmp_path):
    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=_policy_config_path(tmp_path),
    )
    receipt = decision_to_receipt(decision)

    payload = json.loads(receipt_to_json(receipt))

    assert payload["selected_engine"] == "reasoning_local"
    assert "provider policy denylist: openai" in payload["reasons"]
    assert "policy.provider_denylist" in payload["reason_codes"]
    assert "rejection.provider_denied" in payload["reason_codes"]
    assert "Denied providers: openai" in payload["policy_explanation"]
    assert "hosted_reasoning: provider openai denied" in payload[
        "rejection_explanation"
    ]
    assert "Fallback selected reasoning_local" in payload["fallback_explanation"]
    assert {
        "engine": "hosted_reasoning",
        "reason": "provider openai denied by provider policy",
    } in payload["rejected_engines"]


def test_receipt_explains_forced_engine_policy_fallback(tmp_path):
    decision = route_prompt(
        "rewrite this text",
        config_path=_policy_config_path(tmp_path),
        hints={"force_engine": "hosted_reasoning"},
    )
    receipt = decision_to_receipt(decision)

    assert receipt.selected_engine == "reasoning_local"
    assert "force_engine.requested" in receipt.reason_codes
    assert "rejection.provider_denied" in receipt.reason_codes
    assert "Fallback selected reasoning_local" in receipt.fallback_explanation
