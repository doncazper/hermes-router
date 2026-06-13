"""Receipt helpers for routing decisions."""

from __future__ import annotations

import json

from hermes.plugins.model_router.models import RoutingDecision, RoutingReceipt


def decision_to_receipt(decision: RoutingDecision) -> RoutingReceipt:
    return RoutingReceipt(
        selected_engine=decision.selected_engine,
        complexity_score=decision.complexity_score,
        risk_score=decision.risk_score,
        confidence_score=decision.confidence_score,
        reasons=decision.reasons,
        fallback_engine=decision.fallback_engine,
        requires_confirmation=decision.requires_confirmation,
        requires_tools=decision.requires_tools,
        requires_freshness=decision.requires_freshness,
        requires_code_execution=decision.requires_code_execution,
        config_valid=decision.config_valid,
        availability_valid=decision.availability_valid,
        availability_reasons=decision.availability_reasons,
    )


def receipt_to_json(receipt: RoutingReceipt, indent: int | None = 2) -> str:
    return json.dumps(receipt.to_dict(), indent=indent, sort_keys=True)
