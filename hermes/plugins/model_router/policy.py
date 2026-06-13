"""Policy-based engine selection for scored prompts."""

from __future__ import annotations

from pathlib import Path

from hermes.plugins.model_router.availability import validate_engine_availability
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.models import (
    EngineRejection,
    ModelEngine,
    PromptAnalysis,
    PromptFeatures,
    RouterConfig,
    RoutingDecision,
    RoutingHints,
    RoutingRequirements,
)
from hermes.plugins.model_router.scorer import score_prompt

FAIL_CLOSED_ENGINE = "human_confirm"
COST_TIER_ORDER = {
    "none": 0,
    "free": 0,
    "low": 1,
    "medium": 2,
    "paid": 2,
    "high": 3,
}
LATENCY_TIER_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "manual": 4,
}


def route_prompt(
    prompt: str,
    *,
    config: RouterConfig | None = None,
    config_path: str | Path | None = None,
    hints: dict | RoutingHints | None = None,
) -> RoutingDecision:
    analysis = score_prompt(prompt)
    try:
        routing_hints = _coerce_hints(hints)
    except ValueError as exc:
        return _fail_closed(
            analysis,
            f"fail-closed: invalid routing hints: {exc}",
            requirements=RoutingRequirements(),
        )

    requirements = _derive_requirements(analysis, routing_hints)
    try:
        router_config = config if config is not None else load_router_config(config_path)
    except RouterConfigError as exc:
        return _fail_closed(
            analysis,
            f"fail-closed: {exc}",
            requirements=requirements,
        )

    if routing_hints.force_engine:
        if analysis.features.requires_confirmation and (
            routing_hints.force_engine != FAIL_CLOSED_ENGINE
        ):
            target = "confirmation"
            target_reason = (
                f"force_engine ignored for high-risk request: "
                f"{routing_hints.force_engine}"
            )
        else:
            return _route_forced_engine(
                analysis,
                router_config,
                requirements,
                routing_hints.force_engine,
            )
    else:
        target, target_reason = _target_route(analysis, requirements)

    (
        selected,
        fallback_engine,
        fallback_reason,
        availability_valid,
        availability_reasons,
        rejected_engines,
        fallback_used,
    ) = _resolve_enabled_route(
        target,
        router_config,
        requirements,
    )
    reasons = [*analysis.reasons, target_reason]
    if fallback_reason:
        reasons.append(fallback_reason)

    requires_confirmation = (
        analysis.features.requires_confirmation or selected == FAIL_CLOSED_ENGINE
    )
    return RoutingDecision(
        selected_engine=selected,
        fallback_engine=fallback_engine,
        complexity_score=analysis.complexity_score.value,
        risk_score=analysis.risk_score.value,
        confidence_score=analysis.confidence_score,
        reasons=tuple(dict.fromkeys(reasons)),
        requires_confirmation=requires_confirmation,
        requires_tools=requirements.needs_tools,
        requires_freshness=analysis.features.requires_freshness,
        requires_code_execution=analysis.features.requires_code_execution,
        requires_vision=_requires_vision(analysis.features, requirements),
        requires_image_generation=analysis.features.requires_image_generation,
        config_valid=True,
        availability_valid=availability_valid,
        availability_reasons=availability_reasons,
        features=analysis.features,
        requirements=requirements,
        rejected_engines=rejected_engines,
        fallback_used=fallback_used,
    )


def _coerce_hints(hints: dict | RoutingHints | None) -> RoutingHints:
    if isinstance(hints, RoutingHints):
        return hints
    return RoutingHints.from_dict(hints)


def _derive_requirements(
    analysis: PromptAnalysis,
    hints: RoutingHints,
) -> RoutingRequirements:
    required_modalities = tuple(
        attachment for attachment in hints.attachments if attachment != "code"
    )
    return RoutingRequirements(
        needs_tools=(
            analysis.features.requires_code_execution
            or analysis.features.requires_freshness
            or analysis.features.requires_vision
            or analysis.features.requires_image_generation
            or bool(required_modalities)
        ),
        required_modalities=required_modalities,
        max_cost_tier=hints.max_cost_tier,
        max_latency_tier=hints.max_latency_tier
        or ("medium" if hints.latency_sensitive else None),
    )


def _route_forced_engine(
    analysis: PromptAnalysis,
    router_config: RouterConfig,
    requirements: RoutingRequirements,
    force_engine: str,
) -> RoutingDecision:
    (
        selected,
        fallback_engine,
        fallback_reason,
        availability_valid,
        availability_reasons,
        rejected_engines,
        fallback_used,
    ) = _resolve_enabled_engine(force_engine, router_config, requirements)
    reasons = [*analysis.reasons, f"forced engine {force_engine}"]
    if selected == FAIL_CLOSED_ENGINE and router_config.get_engine(force_engine) is None:
        reasons.append(f"unknown forced engine {force_engine}")
    if fallback_reason:
        reasons.append(fallback_reason)
    return RoutingDecision(
        selected_engine=selected,
        fallback_engine=fallback_engine,
        complexity_score=analysis.complexity_score.value,
        risk_score=analysis.risk_score.value,
        confidence_score=analysis.confidence_score,
        reasons=tuple(dict.fromkeys(reasons)),
        requires_confirmation=analysis.features.requires_confirmation
        or selected == FAIL_CLOSED_ENGINE,
        requires_tools=requirements.needs_tools,
        requires_freshness=analysis.features.requires_freshness,
        requires_code_execution=analysis.features.requires_code_execution,
        requires_vision=_requires_vision(analysis.features, requirements),
        requires_image_generation=analysis.features.requires_image_generation,
        config_valid=True,
        availability_valid=availability_valid,
        availability_reasons=availability_reasons,
        features=analysis.features,
        requirements=requirements,
        rejected_engines=rejected_engines,
        fallback_used=fallback_used,
    )


def _requires_vision(
    features: PromptFeatures,
    requirements: RoutingRequirements,
) -> bool:
    return features.requires_vision or any(
        modality in requirements.required_modalities
        for modality in ("image", "pdf", "audio")
    )


def _target_route(
    analysis: PromptAnalysis,
    requirements: RoutingRequirements,
) -> tuple[str, str]:
    features = analysis.features
    if features.requires_confirmation:
        return "confirmation", "high-risk action requires human confirmation"
    if features.ambiguous and features.sensitive_domain:
        return "confirmation", "ambiguous high-impact request"
    if features.requires_image_generation:
        return "image_generation", "image generation required"
    if requirements.required_modalities:
        return "vision", "attachment modality requires vision or extraction"
    if features.requires_vision and not features.requires_code_execution:
        return "vision", "multimodal vision or OCR required"
    if features.requires_code_execution or features.coding_intent:
        return "coding", "coding or repository work"
    if features.requires_freshness:
        return "research", "fresh research or current information required"
    if (
        analysis.complexity_score.value >= 60
        or features.multi_step_reasoning
        or features.long_context
    ):
        return "reasoning", "complex planning or long-context reasoning"
    if analysis.confidence_score < 60:
        return "reasoning", "low confidence routes upward"
    if (
        features.simple_transform
        and analysis.complexity_score.value < 35
        and analysis.risk_score.value < 20
    ):
        return "simple", "simple rewrite/extraction/formatting"
    return "balanced", "general task"


def _resolve_enabled_route(
    target: str,
    router_config: RouterConfig,
    requirements: RoutingRequirements,
) -> tuple[
    str,
    str | None,
    str | None,
    bool,
    tuple[str, ...],
    tuple[EngineRejection, ...],
    bool,
]:
    engine_name = router_config.target_engine(target)
    if engine_name is None:
        return (
            FAIL_CLOSED_ENGINE,
            FAIL_CLOSED_ENGINE,
            f"fallback to {FAIL_CLOSED_ENGINE}: route {target!r} is undefined",
            False,
            (f"route {target!r} is undefined",),
            (EngineRejection(target, "route is undefined"),),
            True,
        )
    return _resolve_enabled_engine(engine_name, router_config, requirements)


def _resolve_enabled_engine(
    target_engine: str,
    router_config: RouterConfig,
    requirements: RoutingRequirements,
) -> tuple[
    str,
    str | None,
    str | None,
    bool,
    tuple[str, ...],
    tuple[EngineRejection, ...],
    bool,
]:
    visited: set[str] = set()
    current = target_engine
    availability_reasons: list[str] = []
    rejected_engines: list[EngineRejection] = []
    fallback_cause = "disabled"

    while current and current not in visited:
        visited.add(current)
        engine = router_config.get_engine(current)
        if engine is None:
            rejected_engines.append(EngineRejection(current, "engine is undefined"))
            return (
                FAIL_CLOSED_ENGINE,
                FAIL_CLOSED_ENGINE,
                f"fallback to {FAIL_CLOSED_ENGINE}: engine {current!r} is undefined",
                False,
                (f"engine {current!r} is undefined",),
                tuple(rejected_engines),
                True,
            )
        if engine.enabled:
            availability = validate_engine_availability(engine)
            availability_reasons.extend(
                f"{current}: {reason}" for reason in availability.reasons
            )
            if not availability.available:
                rejected_engines.append(EngineRejection(current, "engine unavailable"))
                if engine.fallback is None:
                    return (
                        FAIL_CLOSED_ENGINE,
                        FAIL_CLOSED_ENGINE,
                        f"fallback to {FAIL_CLOSED_ENGINE}: {current} unavailable",
                        False,
                        tuple(availability_reasons),
                        tuple(rejected_engines),
                        True,
                    )
                fallback = engine.fallback
                availability_reasons.append(
                    f"{current} unavailable; trying fallback {fallback}"
                )
                fallback_cause = "unavailable"
                current = fallback
                continue
            constraint_reason = _engine_constraint_reason(engine, requirements)
            if constraint_reason is not None:
                rejected_engines.append(EngineRejection(current, constraint_reason))
                if engine.fallback is None:
                    return (
                        FAIL_CLOSED_ENGINE,
                        FAIL_CLOSED_ENGINE,
                        f"fallback to {FAIL_CLOSED_ENGINE}: {current} rejected",
                        False,
                        tuple(availability_reasons),
                        tuple(rejected_engines),
                        True,
                    )
                fallback = engine.fallback
                availability_reasons.append(
                    f"{current} rejected ({constraint_reason}); trying fallback "
                    f"{fallback}"
                )
                fallback_cause = "rejected"
                current = fallback
                continue
            if current != target_engine:
                return (
                    current,
                    current,
                    f"fallback to {current}: {target_engine} {fallback_cause}",
                    True,
                    tuple(availability_reasons),
                    tuple(rejected_engines),
                    True,
                )
            return (
                current,
                engine.fallback,
                None,
                True,
                tuple(availability_reasons),
                tuple(rejected_engines),
                False,
            )
        rejected_engines.append(EngineRejection(current, "engine disabled"))
        if engine.fallback is None:
            return (
                FAIL_CLOSED_ENGINE,
                FAIL_CLOSED_ENGINE,
                f"fallback to {FAIL_CLOSED_ENGINE}: {current} disabled",
                False,
                (f"{current} disabled with no fallback",),
                tuple(rejected_engines),
                True,
            )
        current = engine.fallback

    if current is not None:
        rejected_engines.append(EngineRejection(current, "fallback cycle detected"))
    return (
        FAIL_CLOSED_ENGINE,
        FAIL_CLOSED_ENGINE,
        "fallback to human_confirm: fallback cycle detected",
        False,
        ("fallback cycle detected",),
        tuple(rejected_engines),
        True,
    )


def _engine_constraint_reason(
    engine: ModelEngine,
    requirements: RoutingRequirements,
) -> str | None:
    if requirements.needs_tools and not engine.supports_tools:
        return "tools required but engine does not support tools"
    for modality in requirements.required_modalities:
        if modality not in engine.modalities:
            return f"missing required modality {modality}"
    if _tier_exceeds(engine.cost_tier, requirements.max_cost_tier, COST_TIER_ORDER):
        return f"cost_tier {engine.cost_tier} exceeds {requirements.max_cost_tier}"
    if _tier_exceeds(
        engine.latency_tier,
        requirements.max_latency_tier,
        LATENCY_TIER_ORDER,
    ):
        return (
            f"latency_tier {engine.latency_tier} exceeds "
            f"{requirements.max_latency_tier}"
        )
    return None


def _tier_exceeds(
    value: str,
    max_value: str | None,
    order: dict[str, int],
) -> bool:
    if max_value is None:
        return False
    return order.get(value, 999) > order.get(max_value, 999)


def _fail_closed(
    analysis: PromptAnalysis,
    reason: str,
    *,
    requirements: RoutingRequirements,
    rejected_engines: tuple[EngineRejection, ...] = (),
) -> RoutingDecision:
    return RoutingDecision(
        selected_engine=FAIL_CLOSED_ENGINE,
        fallback_engine=None,
        complexity_score=analysis.complexity_score.value,
        risk_score=max(analysis.risk_score.value, 80),
        confidence_score=min(analysis.confidence_score, 50),
        reasons=tuple(dict.fromkeys([*analysis.reasons, reason])),
        requires_confirmation=True,
        requires_tools=analysis.features.requires_tools,
        requires_freshness=analysis.features.requires_freshness,
        requires_code_execution=analysis.features.requires_code_execution,
        requires_vision=analysis.features.requires_vision,
        requires_image_generation=analysis.features.requires_image_generation,
        config_valid=False,
        availability_valid=False,
        availability_reasons=(reason,),
        features=_with_confirmation(analysis.features),
        requirements=requirements,
        rejected_engines=rejected_engines,
        fallback_used=True,
    )


def _with_confirmation(features: PromptFeatures) -> PromptFeatures:
    data = features.to_dict()
    data["requires_confirmation"] = True
    return PromptFeatures(**data)
