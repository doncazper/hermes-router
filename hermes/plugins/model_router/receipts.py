"""Receipt helpers for routing decisions."""

from __future__ import annotations

import json

from hermes.plugins.model_router.models import EngineRejection, RoutingDecision, RoutingReceipt


LOCAL_PROVIDER_SET = {"local", "human"}


def decision_to_receipt(decision: RoutingDecision) -> RoutingReceipt:
    return RoutingReceipt(
        selected_engine=decision.selected_engine,
        routing_profile=decision.routing_profile,
        complexity_score=decision.complexity_score,
        risk_score=decision.risk_score,
        confidence_score=decision.confidence_score,
        reasons=decision.reasons,
        fallback_engine=decision.fallback_engine,
        requires_confirmation=decision.requires_confirmation,
        requires_tools=decision.requires_tools,
        requires_freshness=decision.requires_freshness,
        requires_code_execution=decision.requires_code_execution,
        requires_vision=decision.requires_vision,
        requires_image_generation=decision.requires_image_generation,
        config_valid=decision.config_valid,
        availability_valid=decision.availability_valid,
        availability_reasons=decision.availability_reasons,
        requirements=decision.requirements,
        rejected_engines=decision.rejected_engines,
        alternatives=decision.alternatives,
        fallback_used=decision.fallback_used,
        summary=_receipt_summary(decision),
        reason_codes=_receipt_reason_codes(decision),
        selected_route_explanation=_selected_route_explanation(decision),
        policy_explanation=_policy_explanation(decision),
        rejection_explanation=_rejection_explanation(decision.rejected_engines),
        fallback_explanation=_fallback_explanation(decision),
        safety_explanation=_safety_explanation(decision),
        privacy_explanation=_privacy_explanation(decision),
        wrong_route_next_action=_wrong_route_next_action(decision),
    )


def receipt_to_json(receipt: RoutingReceipt, indent: int | None = 2) -> str:
    return json.dumps(receipt.to_dict(), indent=indent, sort_keys=True)


def _receipt_summary(decision: RoutingDecision) -> str:
    confirmation = (
        "human confirmation required"
        if decision.requires_confirmation
        else "no confirmation required"
    )
    fallback = (
        f"fallback used: {decision.fallback_engine}"
        if decision.fallback_used and decision.fallback_engine
        else (
            f"fallback available: {decision.fallback_engine}"
            if decision.fallback_engine
            else "no fallback configured"
        )
    )
    return (
        f"Selected {decision.selected_engine} under the "
        f"{decision.routing_profile.value} profile; {confirmation}; {fallback}."
    )


def _receipt_reason_codes(decision: RoutingDecision) -> tuple[str, ...]:
    codes: list[str] = [f"profile.{decision.routing_profile.value}"]
    if not decision.config_valid:
        codes.append("config.invalid")
    if not decision.availability_valid:
        codes.append("availability.issue")

    codes.append(_selected_route_code(decision))
    codes.extend(_requirement_codes(decision))
    codes.extend(_policy_codes(decision))
    codes.extend(_reason_string_codes(decision.reasons))
    codes.extend(_rejection_codes(decision.rejected_engines))
    if decision.fallback_used:
        codes.append("fallback.used")
    elif decision.fallback_engine:
        codes.append("fallback.configured")
    if decision.requires_confirmation:
        codes.append("safety.confirmation_required")
    else:
        codes.append("safety.no_confirmation_required")
    return tuple(dict.fromkeys(codes))


def _selected_route_code(decision: RoutingDecision) -> str:
    if decision.selected_engine == "human_confirm":
        return "route.confirmation"
    if decision.requires_image_generation:
        return "route.image_generation"
    if decision.requires_vision:
        return "route.vision"
    if decision.requires_freshness:
        return "route.research"
    if decision.requires_code_execution:
        return "route.coding"
    if decision.complexity_score >= 60:
        return "route.reasoning"
    if decision.selected_engine == "fast_local":
        return "route.simple"
    return "route.balanced"


def _requirement_codes(decision: RoutingDecision) -> tuple[str, ...]:
    codes: list[str] = []
    if decision.requires_tools:
        codes.append("requirement.tools")
    if decision.requires_freshness:
        codes.append("requirement.freshness")
    if decision.requires_code_execution:
        codes.append("requirement.code_execution")
    if decision.requires_vision:
        codes.append("requirement.vision")
    if decision.requires_image_generation:
        codes.append("requirement.image_generation")
    if decision.requirements.required_modalities:
        codes.append("requirement.modality")
    return tuple(codes)


def _policy_codes(decision: RoutingDecision) -> tuple[str, ...]:
    requirements = decision.requirements
    codes: list[str] = []
    allowed = set(requirements.allowed_providers)
    if allowed:
        codes.append("policy.provider_allowlist")
    if allowed and allowed <= LOCAL_PROVIDER_SET:
        codes.append("policy.local_only")
    if requirements.denied_providers:
        codes.append("policy.provider_denylist")
    if requirements.max_cost_tier:
        codes.append("policy.max_cost_tier")
    if requirements.max_latency_tier:
        codes.append("policy.max_latency_tier")
    if requirements.profile_reasons:
        codes.append("policy.profile")
    if requirements.provider_policy_reasons:
        codes.append("policy.provider")
    return tuple(codes)


def _reason_string_codes(reasons: tuple[str, ...]) -> tuple[str, ...]:
    codes: list[str] = []
    for reason in reasons:
        lowered = reason.lower()
        if "forced engine" in lowered:
            codes.append("force_engine.requested")
        if "force_engine ignored" in lowered:
            codes.append("force_engine.ignored")
        if "human confirmation" in lowered or "requires confirmation" in lowered:
            codes.append("safety.confirmation_required")
        if "private" in lowered and "local" in lowered:
            codes.append("policy.local_only")
        if "safe" in lowered and "confirmation" in lowered:
            codes.append("profile.safe_confirmation")
        if "fresh" in lowered or "current information" in lowered:
            codes.append("route.research")
        if "coding" in lowered or "repository" in lowered:
            codes.append("route.coding")
    return tuple(codes)


def _rejection_codes(rejected_engines: tuple[EngineRejection, ...]) -> tuple[str, ...]:
    codes: list[str] = []
    for rejection in rejected_engines:
        lowered = rejection.reason.lower()
        if "provider" in lowered and "denied" in lowered:
            codes.append("rejection.provider_denied")
        elif "provider" in lowered and "not allowed" in lowered:
            codes.append("rejection.provider_not_allowed")
        elif "backend" in lowered and "denied" in lowered:
            codes.append("rejection.backend_denied")
        elif "backend" in lowered and "not allowed" in lowered:
            codes.append("rejection.backend_not_allowed")
        elif "tools required" in lowered:
            codes.append("rejection.tools_missing")
        elif "modality" in lowered:
            codes.append("rejection.modality_missing")
        elif "cost_tier" in lowered:
            codes.append("rejection.cost_tier")
        elif "latency_tier" in lowered:
            codes.append("rejection.latency_tier")
        elif "unavailable" in lowered:
            codes.append("rejection.unavailable")
        elif "disabled" in lowered:
            codes.append("rejection.disabled")
        elif "undefined" in lowered:
            codes.append("rejection.undefined")
        elif "cycle" in lowered:
            codes.append("rejection.fallback_cycle")
    return tuple(codes)


def _selected_route_explanation(decision: RoutingDecision) -> str:
    if decision.selected_engine == "human_confirm":
        return "Selected human_confirm because the request needs explicit review before dispatch."
    if decision.requires_image_generation:
        reason = "image generation"
    elif decision.requires_vision:
        reason = "vision, OCR, or attachment handling"
    elif decision.requires_freshness:
        reason = "fresh research or current information"
    elif decision.requires_code_execution:
        reason = "coding or repository work"
    elif decision.requires_tools:
        reason = "tool-capable handling"
    elif decision.complexity_score >= 60:
        reason = "higher-complexity reasoning"
    else:
        reason = "the deterministic route matched this request"
    return f"Selected {decision.selected_engine} for {reason}."


def _policy_explanation(decision: RoutingDecision) -> str:
    requirements = decision.requirements
    parts: list[str] = []
    if requirements.profile_reasons:
        parts.append("Profile: " + "; ".join(requirements.profile_reasons))
    if requirements.provider_policy_reasons:
        parts.append("Provider policy: " + "; ".join(requirements.provider_policy_reasons))
    if requirements.allowed_providers:
        parts.append("Allowed providers: " + ", ".join(requirements.allowed_providers))
    if requirements.denied_providers:
        parts.append("Denied providers: " + ", ".join(requirements.denied_providers))
    if requirements.max_cost_tier:
        parts.append(f"Max cost tier: {requirements.max_cost_tier}")
    if requirements.max_latency_tier:
        parts.append(f"Max latency tier: {requirements.max_latency_tier}")
    return ". ".join(parts) + "." if parts else "No profile or provider policy constraints changed this route."


def _rejection_explanation(rejected_engines: tuple[EngineRejection, ...]) -> str:
    if not rejected_engines:
        return "No engines were rejected before selecting this route."
    shown = "; ".join(
        f"{rejection.engine}: {rejection.reason}" for rejection in rejected_engines[:4]
    )
    remaining = len(rejected_engines) - 4
    if remaining > 0:
        shown = f"{shown}; {remaining} more rejected"
    return f"Rejected engines: {shown}."


def _fallback_explanation(decision: RoutingDecision) -> str:
    if decision.fallback_used and decision.fallback_engine:
        return (
            f"Fallback selected {decision.fallback_engine} after the primary route "
            "was unavailable, incompatible, or blocked by policy."
        )
    if decision.fallback_used:
        return "Fallback handling was used and the request failed closed."
    if decision.fallback_engine:
        return f"No fallback was used; {decision.fallback_engine} remains the configured fallback."
    return "No fallback was used and no fallback engine is configured."


def _safety_explanation(decision: RoutingDecision) -> str:
    if decision.requires_confirmation:
        return "Human confirmation is required before dispatch."
    if decision.selected_engine == "human_confirm":
        return "The request failed closed to human confirmation."
    return "No human confirmation is required by the current safety policy."


def _privacy_explanation(decision: RoutingDecision) -> str:
    allowed = set(decision.requirements.allowed_providers)
    denied = decision.requirements.denied_providers
    if allowed and allowed <= LOCAL_PROVIDER_SET:
        return "Local-only routing is active; hosted providers are excluded."
    if denied:
        return "Provider denylist active: " + ", ".join(denied) + "."
    if decision.routing_profile.value == "private":
        return "Private profile is active; hosted providers are excluded."
    return "No raw prompt text is stored in this receipt; provider use follows the configured catalog and policy."


def _wrong_route_next_action(decision: RoutingDecision) -> str:
    expected = "code_agent" if decision.selected_engine != "code_agent" else "balanced_local"
    return (
        "If this route was wrong, label the proxy request id with "
        f"`model-router feedback <request_id> {expected}` or rerun "
        "`model-router decide --explain` with adjusted profile/provider policy."
    )
