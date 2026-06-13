"""Policy-based engine selection for scored prompts."""

from __future__ import annotations

from pathlib import Path

from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.models import (
    PromptAnalysis,
    PromptFeatures,
    RouterConfig,
    RoutingDecision,
)
from hermes.plugins.model_router.scorer import score_prompt

FAIL_CLOSED_ENGINE = "human_confirm"


def route_prompt(
    prompt: str,
    *,
    config: RouterConfig | None = None,
    config_path: str | Path | None = None,
) -> RoutingDecision:
    analysis = score_prompt(prompt)
    try:
        router_config = config if config is not None else load_router_config(config_path)
    except RouterConfigError as exc:
        return _fail_closed(analysis, f"fail-closed: {exc}")

    target, target_reason = _target_engine(analysis)
    selected, fallback_engine, fallback_reason = _resolve_enabled_engine(
        target,
        router_config,
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
        requires_tools=analysis.features.requires_tools,
        requires_freshness=analysis.features.requires_freshness,
        requires_code_execution=analysis.features.requires_code_execution,
        config_valid=True,
        features=analysis.features,
    )


def _target_engine(analysis: PromptAnalysis) -> tuple[str, str]:
    features = analysis.features
    if features.requires_confirmation:
        return FAIL_CLOSED_ENGINE, "high-risk action requires human confirmation"
    if features.ambiguous and features.sensitive_domain:
        return FAIL_CLOSED_ENGINE, "ambiguous high-impact request"
    if features.requires_code_execution or features.coding_intent:
        return "codex", "coding or repository work"
    if features.requires_freshness:
        return "web_research", "fresh research or current information required"
    if (
        analysis.complexity_score.value >= 60
        or features.multi_step_reasoning
        or features.long_context
    ):
        return "reasoning_local", "complex planning or long-context reasoning"
    if analysis.confidence_score < 60:
        return "reasoning_local", "low confidence routes upward"
    if (
        features.simple_transform
        and analysis.complexity_score.value < 35
        and analysis.risk_score.value < 20
    ):
        return "fast_local", "simple rewrite/extraction/formatting"
    return "balanced_local", "general task"


def _resolve_enabled_engine(
    target: str,
    router_config: RouterConfig,
) -> tuple[str, str | None, str | None]:
    visited: set[str] = set()
    current = target

    while current and current not in visited:
        visited.add(current)
        engine = router_config.get_engine(current)
        if engine is None:
            return (
                FAIL_CLOSED_ENGINE,
                FAIL_CLOSED_ENGINE,
                f"fallback to {FAIL_CLOSED_ENGINE}: engine {current!r} is undefined",
            )
        if engine.enabled:
            if current != target:
                return current, current, f"fallback to {current}: {target} disabled"
            return current, engine.fallback, None
        if engine.fallback is None:
            return (
                FAIL_CLOSED_ENGINE,
                FAIL_CLOSED_ENGINE,
                f"fallback to {FAIL_CLOSED_ENGINE}: {current} disabled",
            )
        current = engine.fallback

    return (
        FAIL_CLOSED_ENGINE,
        FAIL_CLOSED_ENGINE,
        "fallback to human_confirm: fallback cycle detected",
    )


def _fail_closed(analysis: PromptAnalysis, reason: str) -> RoutingDecision:
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
        config_valid=False,
        features=_with_confirmation(analysis.features),
    )


def _with_confirmation(features: PromptFeatures) -> PromptFeatures:
    data = features.to_dict()
    data["requires_confirmation"] = True
    return PromptFeatures(**data)
