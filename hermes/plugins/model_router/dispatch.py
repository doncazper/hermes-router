"""Safe dry-run dispatch planning for routed prompts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from hermes.plugins.model_router.models import RoutingHints, RoutingReceipt
from hermes.plugins.model_router.policy import ModelRouter
from hermes.plugins.model_router.receipts import decision_to_receipt


@dataclass(frozen=True)
class DispatchPlan:
    selected_engine: str
    provider: str
    model: str
    adapter: str
    dry_run: bool
    can_dispatch: bool
    blocked: bool
    requires_confirmation: bool
    reasons: tuple[str, ...]
    receipt: RoutingReceipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "provider": self.provider,
            "model": self.model,
            "adapter": self.adapter,
            "dry_run": self.dry_run,
            "can_dispatch": self.can_dispatch,
            "blocked": self.blocked,
            "requires_confirmation": self.requires_confirmation,
            "reasons": list(self.reasons),
            "receipt": self.receipt.to_dict(),
        }


def build_dispatch_plan(
    prompt: str,
    *,
    router: ModelRouter | None = None,
    config_path: str | Path | None = None,
    hints: dict | RoutingHints | None = None,
    include_alternatives: bool = False,
) -> DispatchPlan:
    active_router = router or ModelRouter.from_config(config_path)
    decision = active_router.route(
        prompt,
        hints=hints,
        include_alternatives=include_alternatives,
    )
    receipt = decision_to_receipt(decision)
    engine = active_router.config.get_engine(decision.selected_engine)

    blocked = (
        decision.requires_confirmation
        or decision.selected_engine == "human_confirm"
        or not decision.config_valid
        or not decision.availability_valid
    )
    reasons = ["dry-run dispatch plan; no adapter executed"]
    if decision.requires_confirmation:
        reasons.append("confirmation required before any dispatch")
    if decision.selected_engine == "human_confirm":
        reasons.append("selected engine is the human confirmation gate")
    if not decision.config_valid:
        reasons.append("config is invalid")
    if not decision.availability_valid:
        reasons.append("selected engine is not available")

    return DispatchPlan(
        selected_engine=decision.selected_engine,
        provider=engine.provider if engine else "unknown",
        model=engine.model if engine else "unknown",
        adapter=engine.adapter if engine else "unknown",
        dry_run=True,
        can_dispatch=not blocked,
        blocked=blocked,
        requires_confirmation=decision.requires_confirmation,
        reasons=tuple(dict.fromkeys(reasons)),
        receipt=receipt,
    )


def dispatch_plan_to_json(plan: DispatchPlan, indent: int | None = 2) -> str:
    return json.dumps(plan.to_dict(), indent=indent, sort_keys=True)
