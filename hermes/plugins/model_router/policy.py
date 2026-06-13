"""Policy-based engine selection for scored prompts."""

from __future__ import annotations

from pathlib import Path

from hermes.plugins.model_router.availability import (
    validate_engine_availability,
    validate_router_availability,
)
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.models import (
    EngineRejection,
    EngineAvailabilityResult,
    ModelEngine,
    PromptAnalysis,
    PromptFeatures,
    RouterConfig,
    RouterAvailabilityReport,
    RoutingAlternative,
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
_FAST_CONFIRMATION_PREFIXES = (
    "delete",
    "remove",
    "wipe",
    "destroy",
    "drop",
    "erase",
    "cancel",
    "terminate",
    "purge",
    "truncate",
    "shutdown",
    "uninstall",
    "revoke",
    "send",
    "post",
    "publish",
    "submit",
    "reply",
    "buy",
    "purchase",
    "order",
    "pay",
    "transfer",
    "wire",
    "subscribe",
    "schedule",
    "reschedule",
    "invite",
    "deploy",
    "merge",
    "commit",
    "push",
)
_FAST_CONFIRMATION_WORDS = frozenset(_FAST_CONFIRMATION_PREFIXES)
_FAST_CONFIRMATION_VERBS_WITH_OBJECT = frozenset({"email", "message"})
_FAST_BENIGN_MESSAGE_OBJECTS = frozenset(
    {
        "address",
        "board",
        "body",
        "content",
        "copy",
        "draft",
        "format",
        "header",
        "ideas",
        "marketing",
        "notification",
        "preferences",
        "settings",
        "subject",
        "summary",
        "template",
    }
)
_FAST_BOOK_ACTION_OBJECTS = frozenset(
    {
        "appointment",
        "call",
        "flight",
        "hotel",
        "meeting",
        "reservation",
        "ride",
        "room",
        "table",
        "ticket",
        "trip",
    }
)
_FAST_ARTICLES = frozenset({"a", "an", "the", "my", "our"})
_FAST_PUNCTUATION_STRIP = ".,!?;:"
_FAST_CODING_MARKERS = (
    " code ",
    " coding ",
    " repo ",
    " repository ",
    " implement ",
    " implementation ",
    " pytest ",
    " ruff ",
    " unit test",
    " tests",
    " debug ",
    " bug ",
    " fix the repo",
    " edit ",
    " pull request",
    " pr ",
)
_FAST_RESEARCH_MARKERS = (
    " research ",
    " look up ",
    " search ",
    " browse ",
    " cite ",
    " citation",
    " source",
    " trend",
    " web ",
)
_FAST_CURRENT_MARKERS = (
    " current ",
    " latest ",
    " recent ",
    " today ",
    " yesterday ",
    " now ",
    " news ",
    " up-to-date ",
    " fresh ",
)
_FAST_REASONING_MARKERS = (
    " architecture",
    " architect",
    " design ",
    " plan ",
    " multi-step",
    " strategy ",
    " roadmap ",
    " tradeoff",
    " trade-off",
    " edge case",
    " data flow",
    " rollout",
    " migration",
    " system",
    " distributed",
    " scalable",
    " consensus",
    " throughput",
    " backpressure",
    " exactly-once",
)
_FAST_SIMPLE_MARKERS = (
    " rewrite ",
    " rephrase ",
    " format ",
    " extract ",
    " clean up ",
    " copyedit ",
    " proofread ",
)
_FAST_SIMPLE_PREFIXES = (
    "rewrite",
    "rephrase",
    "format",
    "clean up",
    "copyedit",
    "proofread",
)
_FAST_VISION_MARKERS = (
    " image ",
    " picture ",
    " photo ",
    " screenshot",
    " screen shot",
    " chart ",
    " diagram ",
    " graph ",
    " ocr ",
    " vision ",
    " visual ",
    " scan ",
)
_FAST_IMAGE_NOUN_MARKERS = (
    " image ",
    " picture ",
    " photo ",
    " illustration ",
    " logo ",
    " icon ",
    " wallpaper ",
    " poster ",
    " diffusion",
    " stable diffusion",
)
_FAST_IMAGE_VERB_MARKERS = (
    " generate ",
    " create ",
    " make ",
    " draw ",
    " render ",
    " produce ",
    " design ",
)
_FAST_TOOL_TARGETS = frozenset(
    {"coding", "research", "vision", "image_generation", "confirmation"}
)
_FAST_TARGET_NAMES = (
    "simple",
    "balanced",
    "reasoning",
    "coding",
    "research",
    "vision",
    "image_generation",
    "confirmation",
)
_FAST_SIMPLE_INDEX = 0
_FAST_BALANCED_INDEX = 1
_FAST_REASONING_INDEX = 2
_FAST_CODING_INDEX = 3
_FAST_RESEARCH_INDEX = 4
_FAST_VISION_INDEX = 5
_FAST_IMAGE_GENERATION_INDEX = 6
_FAST_CONFIRMATION_INDEX = 7
_FAST_TARGET_REQUIREMENTS = {
    "simple": RoutingRequirements(),
    "balanced": RoutingRequirements(),
    "reasoning": RoutingRequirements(),
    "coding": RoutingRequirements(needs_tools=True),
    "research": RoutingRequirements(needs_tools=True),
    "vision": RoutingRequirements(needs_tools=True),
    "image_generation": RoutingRequirements(needs_tools=True),
    "confirmation": RoutingRequirements(),
}


class ModelRouter:
    """Initialized, reusable model router for the runtime hot path."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        availability_report: RouterAvailabilityReport | None = None,
    ) -> None:
        self.config = config
        self._availability_results = (
            availability_report.engines if availability_report is not None else None
        )
        self._fast_target_engines = self._compile_fast_target_engines()
        self._fast_engine_data = {
            name: (
                engine.enabled,
                engine.fallback,
                engine.supports_tools,
                engine.modalities,
                engine.cost_tier,
                engine.latency_tier,
                self._fast_engine_available(name),
            )
            for name, engine in config.engines.items()
        }
        self._fast_engine_count = len(self._fast_engine_data)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        validate_availability: bool = True,
    ) -> "ModelRouter":
        return cls.from_config_object(
            load_router_config(config_path),
            validate_availability=validate_availability,
        )

    @classmethod
    def from_config_object(
        cls,
        config: RouterConfig,
        *,
        validate_availability: bool = True,
    ) -> "ModelRouter":
        availability_report = (
            validate_router_availability(config) if validate_availability else None
        )
        return cls(config, availability_report=availability_report)

    def route(
        self,
        prompt: str,
        hints: dict | RoutingHints | None = None,
        *,
        include_alternatives: bool = True,
    ) -> RoutingDecision:
        analysis = score_prompt(prompt, scoring_config=self.config.scoring)
        try:
            routing_hints = _coerce_hints(hints)
        except ValueError as exc:
            return _fail_closed(
                analysis,
                f"fail-closed: invalid routing hints: {exc}",
                requirements=RoutingRequirements(),
            )

        requirements = _derive_requirements(analysis, routing_hints)

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
                    self.config,
                    requirements,
                    routing_hints.force_engine,
                    routing_hints,
                    self._availability_results,
                    include_alternatives=include_alternatives,
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
            self.config,
            requirements,
            self._availability_results,
        )
        if include_alternatives:
            alternatives, alternative_rejections = _rank_alternatives(
                self.config,
                selected,
                analysis,
                requirements,
                routing_hints,
                self._availability_results,
            )
        else:
            alternatives = ()
            alternative_rejections = ()
        reasons = [*analysis.reasons, target_reason]
        if fallback_reason:
            reasons.append(fallback_reason)

        requires_confirmation = (
            analysis.features.requires_confirmation or selected == FAIL_CLOSED_ENGINE
        )
        if requires_confirmation and selected == FAIL_CLOSED_ENGINE:
            alternatives = ()
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
            rejected_engines=tuple(
                dict.fromkeys((*rejected_engines, *alternative_rejections))
            ),
            alternatives=alternatives,
            fallback_used=fallback_used,
        )

    def route_fast(
        self,
        prompt: str,
        hints: dict | RoutingHints | None = None,
    ) -> str:
        """Return only the selected engine through the precompiled hot path.

        This path is for latency-sensitive callers that need a safe engine choice,
        not a scored receipt. It still sends high-risk prompts to human_confirm.
        """
        target_index = _fast_target_route_index(prompt)
        if hints is None:
            return self._fast_target_engines[target_index]

        target = _FAST_TARGET_NAMES[target_index]

        try:
            routing_hints = _coerce_hints(hints)
        except ValueError:
            return FAIL_CLOSED_ENGINE

        required_modalities = tuple(
            attachment
            for attachment in routing_hints.attachments
            if attachment != "code"
        )
        if target != "confirmation" and required_modalities:
            target = "vision"

        max_latency_tier = routing_hints.max_latency_tier
        if max_latency_tier is None and routing_hints.latency_sensitive:
            max_latency_tier = "medium"

        if routing_hints.force_engine:
            if (
                target == "confirmation"
                and routing_hints.force_engine != FAIL_CLOSED_ENGINE
            ):
                return FAIL_CLOSED_ENGINE
            return self._resolve_engine_fast(
                routing_hints.force_engine,
                needs_tools=target in _FAST_TOOL_TARGETS or bool(required_modalities),
                required_modalities=required_modalities,
                max_cost_tier=routing_hints.max_cost_tier,
                max_latency_tier=max_latency_tier,
            )

        return self._resolve_target_fast(
            target,
            needs_tools=target in _FAST_TOOL_TARGETS or bool(required_modalities),
            required_modalities=required_modalities,
            max_cost_tier=routing_hints.max_cost_tier,
            max_latency_tier=max_latency_tier,
        )

    def _compile_fast_target_engines(self) -> tuple[str, ...]:
        return tuple(
            _resolve_enabled_route(
                target,
                self.config,
                _FAST_TARGET_REQUIREMENTS.get(target, RoutingRequirements()),
                self._availability_results,
            )[0]
            for target in _FAST_TARGET_NAMES
        )

    def _fast_engine_available(self, engine_name: str) -> bool:
        if self._availability_results is None:
            return True
        result = self._availability_results.get(engine_name)
        return bool(result is not None and result.available)

    def _resolve_target_fast(
        self,
        target: str,
        *,
        needs_tools: bool,
        required_modalities: tuple[str, ...],
        max_cost_tier: str | None,
        max_latency_tier: str | None,
    ) -> str:
        engine_name = self.config.routing_targets.get(target)
        if engine_name is None:
            return FAIL_CLOSED_ENGINE
        return self._resolve_engine_fast(
            engine_name,
            needs_tools=needs_tools,
            required_modalities=required_modalities,
            max_cost_tier=max_cost_tier,
            max_latency_tier=max_latency_tier,
        )

    def _resolve_engine_fast(
        self,
        engine_name: str,
        *,
        needs_tools: bool,
        required_modalities: tuple[str, ...],
        max_cost_tier: str | None,
        max_latency_tier: str | None,
    ) -> str:
        current = engine_name
        for _ in range(self._fast_engine_count + 1):
            data = self._fast_engine_data.get(current)
            if data is None:
                return FAIL_CLOSED_ENGINE
            (
                enabled,
                fallback,
                supports_tools,
                modalities,
                cost_tier,
                latency_tier,
                available,
            ) = data
            if (
                enabled
                and available
                and not (needs_tools and not supports_tools)
                and not any(
                    modality not in modalities for modality in required_modalities
                )
                and not _tier_exceeds(cost_tier, max_cost_tier, COST_TIER_ORDER)
                and not _tier_exceeds(
                    latency_tier,
                    max_latency_tier,
                    LATENCY_TIER_ORDER,
                )
            ):
                return current
            if fallback is None:
                return FAIL_CLOSED_ENGINE
            current = fallback
        return FAIL_CLOSED_ENGINE


def route_prompt(
    prompt: str,
    *,
    config: RouterConfig | None = None,
    config_path: str | Path | None = None,
    hints: dict | RoutingHints | None = None,
) -> RoutingDecision:
    try:
        router = (
            ModelRouter.from_config_object(config)
            if config is not None
            else ModelRouter.from_config(config_path)
        )
    except RouterConfigError as exc:
        analysis = score_prompt(prompt)
        try:
            routing_hints = _coerce_hints(hints)
            requirements = _derive_requirements(analysis, routing_hints)
        except ValueError:
            requirements = RoutingRequirements()
        return _fail_closed(
            analysis,
            f"fail-closed: {exc}",
            requirements=requirements,
        )
    return router.route(prompt, hints=hints)


def _fast_target_route_index(prompt: str) -> int:
    raw_text = (prompt or "").lower()
    prompt_length = len(prompt or "")

    if _fast_has_confirmation_word(raw_text):
        return _FAST_CONFIRMATION_INDEX

    if raw_text.startswith(_FAST_SIMPLE_PREFIXES):
        return _FAST_SIMPLE_INDEX
    if (
        "repo" in raw_text
        or "run tests" in raw_text
        or "pytest" in raw_text
        or "ruff" in raw_text
        or "fix the repo" in raw_text
    ):
        return _FAST_CODING_INDEX

    text = f" {raw_text} "
    image_request = _fast_has_any(text, _FAST_IMAGE_NOUN_MARKERS)
    if image_request and _fast_has_any(text, _FAST_IMAGE_VERB_MARKERS):
        return _FAST_IMAGE_GENERATION_INDEX
    if _fast_has_any(text, _FAST_VISION_MARKERS):
        return _FAST_VISION_INDEX
    if _fast_has_any(text, _FAST_CODING_MARKERS):
        return _FAST_CODING_INDEX
    if (
        "latest" in raw_text
        or "current" in raw_text
        or "today" in raw_text
        or "news" in raw_text
    ) and _fast_has_any(text, _FAST_RESEARCH_MARKERS):
        return _FAST_RESEARCH_INDEX
    if _fast_has_any(text, _FAST_RESEARCH_MARKERS) and _fast_has_any(
        text,
        _FAST_CURRENT_MARKERS,
    ):
        return _FAST_RESEARCH_INDEX
    if prompt_length >= 4000 or _fast_has_any(text, _FAST_REASONING_MARKERS):
        return _FAST_REASONING_INDEX
    if _fast_has_any(text, _FAST_SIMPLE_MARKERS):
        return _FAST_SIMPLE_INDEX
    return _FAST_BALANCED_INDEX


def _fast_has_any(text: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        if marker in text:
            return True
    return False


def _fast_has_confirmation_word(text: str) -> bool:
    apply_pending = False
    book_pending = False
    message_pending = False
    for raw_token in text.split():
        token = raw_token.strip(_FAST_PUNCTUATION_STRIP)
        if not token:
            continue
        if apply_pending:
            if token == "for":
                return True
            apply_pending = False
        if message_pending:
            if token not in _FAST_BENIGN_MESSAGE_OBJECTS:
                return True
            message_pending = False
        if book_pending:
            if token in _FAST_ARTICLES:
                continue
            if token in _FAST_BOOK_ACTION_OBJECTS:
                return True
            book_pending = False
        if token in _FAST_CONFIRMATION_WORDS:
            return True
        if token == "apply":
            apply_pending = True
            continue
        if token in _FAST_CONFIRMATION_VERBS_WITH_OBJECT:
            message_pending = True
            continue
        if token == "book":
            book_pending = True
    return False


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
        needs_tools=analysis.features.requires_tools or bool(required_modalities),
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
    hints: RoutingHints,
    availability_results: dict[str, EngineAvailabilityResult] | None,
    *,
    include_alternatives: bool = True,
) -> RoutingDecision:
    (
        selected,
        fallback_engine,
        fallback_reason,
        availability_valid,
        availability_reasons,
        rejected_engines,
        fallback_used,
    ) = _resolve_enabled_engine(
        force_engine,
        router_config,
        requirements,
        availability_results,
    )
    if include_alternatives:
        alternatives, alternative_rejections = _rank_alternatives(
            router_config,
            selected,
            analysis,
            requirements,
            hints,
            availability_results,
        )
    else:
        alternatives = ()
        alternative_rejections = ()
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
        rejected_engines=tuple(
            dict.fromkeys((*rejected_engines, *alternative_rejections))
        ),
        alternatives=() if selected == FAIL_CLOSED_ENGINE else alternatives,
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
    availability_results: dict[str, EngineAvailabilityResult] | None,
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
    return _resolve_enabled_engine(
        engine_name,
        router_config,
        requirements,
        availability_results,
    )


def _resolve_enabled_engine(
    target_engine: str,
    router_config: RouterConfig,
    requirements: RoutingRequirements,
    availability_results: dict[str, EngineAvailabilityResult] | None,
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
            availability = _engine_availability(engine, availability_results)
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


def _engine_availability(
    engine: ModelEngine,
    availability_results: dict[str, EngineAvailabilityResult] | None,
) -> EngineAvailabilityResult:
    if availability_results is None:
        return EngineAvailabilityResult(
            engine=engine.name,
            available=True,
            reasons=("availability validation disabled",),
        )
    return availability_results.get(engine.name) or validate_engine_availability(engine)


def _rank_alternatives(
    router_config: RouterConfig,
    selected_engine: str,
    analysis: PromptAnalysis,
    requirements: RoutingRequirements,
    hints: RoutingHints,
    availability_results: dict[str, EngineAvailabilityResult] | None,
) -> tuple[tuple[RoutingAlternative, ...], tuple[EngineRejection, ...]]:
    alternatives: list[RoutingAlternative] = []
    rejections: list[EngineRejection] = []
    for engine in router_config.engines.values():
        if engine.name == selected_engine or not engine.enabled:
            continue
        if engine.name in {FAIL_CLOSED_ENGINE, "intent_router"}:
            continue
        availability = _engine_availability(engine, availability_results)
        if not availability.available:
            rejections.append(EngineRejection(engine.name, "engine unavailable"))
            continue
        constraint_reason = _engine_constraint_reason(engine, requirements)
        if constraint_reason is not None:
            rejections.append(EngineRejection(engine.name, constraint_reason))
            continue
        alternatives.append(_rank_engine(engine, analysis, hints))

    alternatives.sort(
        key=lambda alternative: (
            alternative.rank_score,
            alternative.capability,
            100 - alternative.cost,
            100 - alternative.latency,
            alternative.engine,
        ),
        reverse=True,
    )
    return tuple(alternatives), tuple(rejections)


def _rank_engine(
    engine: ModelEngine,
    analysis: PromptAnalysis,
    hints: RoutingHints,
) -> RoutingAlternative:
    if hints.latency_sensitive:
        capability_weight = 0.20
        trust_weight = 0.20
        cost_weight = 0.10
        latency_weight = 0.50
    elif analysis.risk_score.value >= 50 or analysis.features.sensitive_domain:
        capability_weight = 0.25
        trust_weight = 0.50
        cost_weight = 0.15
        latency_weight = 0.10
    elif analysis.complexity_score.value >= 60 or analysis.features.long_context:
        capability_weight = 0.55
        trust_weight = 0.25
        cost_weight = 0.15
        latency_weight = 0.05
    else:
        capability_weight = 0.45
        trust_weight = 0.30
        cost_weight = 0.20
        latency_weight = 0.05

    cost_score = 100 - engine.cost
    latency_score = 100 - engine.latency
    rank_score = round(
        engine.capability * capability_weight
        + engine.trust * trust_weight
        + cost_score * cost_weight
        + latency_score * latency_weight
    )
    reasons = [
        f"capability {engine.capability}/100",
        f"trust {engine.trust}/100",
        f"cost {engine.cost}/100",
        f"latency {engine.latency}/100",
    ]
    if hints.latency_sensitive:
        reasons.append("latency-sensitive ranking")
    elif analysis.risk_score.value >= 50 or analysis.features.sensitive_domain:
        reasons.append("risk-sensitive ranking")
    elif analysis.complexity_score.value >= 60 or analysis.features.long_context:
        reasons.append("complexity-sensitive ranking")
    return RoutingAlternative(
        engine=engine.name,
        rank_score=max(0, min(rank_score, 100)),
        capability=engine.capability,
        trust=engine.trust,
        cost=engine.cost,
        latency=engine.latency,
        reasons=tuple(reasons),
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
