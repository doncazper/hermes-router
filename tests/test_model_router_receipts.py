import json

from hermes.plugins.model_router.policy import route_prompt
from hermes.plugins.model_router.receipts import decision_to_receipt, receipt_to_json


def test_receipts_serialize_to_json():
    decision = route_prompt("rewrite this text")
    receipt = decision_to_receipt(decision)

    payload = json.loads(receipt_to_json(receipt))

    assert payload["selected_engine"] == "fast_local"
    assert "complexity_score" in payload
    assert "risk_score" in payload
    assert "reasons" in payload


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
