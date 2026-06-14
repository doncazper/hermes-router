"""Fast deterministic prompt scoring for model routing."""

from __future__ import annotations

import math
import re

from hermes.plugins.model_router.models import (
    ComplexityScore,
    PromptAnalysis,
    PromptFeatures,
    PromptSignal,
    RiskScore,
    ScoringConfig,
)

DEFAULT_SCORING_WEIGHTS: dict[str, dict[str, int]] = {
    "complexity": {
        "medium_prompt": 10,
        "long_prompt": 15,
        "very_long_prompt": 15,
        "coding_intent": 25,
        "research_intent": 20,
        "current_info_intent": 20,
        "multi_step_reasoning": 25,
        "architecture": 25,
        "tool_intent": 10,
        "structured_output": 8,
        "ambiguous": 12,
        "long_context": 20,
        "sensitive_domain": 8,
        "vision_intent": 15,
        "image_generation_intent": 18,
    },
    "risk": {
        "destructive_action": 70,
        "send_action": 60,
        "purchase_action": 60,
        "external_action": 40,
        "file_shell_github": 25,
        "sensitive_domain": 25,
        "legal_domain": 5,
        "medical_domain": 5,
        "financial_domain": 5,
        "ambiguous_high_impact": 15,
        "production": 20,
        "security": 15,
        "pii": 20,
    },
    "confidence": {
        "ambiguous": 25,
        "short_high_risk": 15,
        "weak_feature_match": 10,
        "empty_prompt": 40,
    },
}


def score_prompt(
    prompt: str,
    *,
    scoring_config: ScoringConfig | None = None,
) -> PromptAnalysis:
    config = scoring_config or ScoringConfig()
    weights = _merged_weights(config)
    text = prompt or ""
    normalized = " ".join(text.lower().split())
    prompt_length = len(text)
    estimated_tokens = math.ceil(prompt_length / 4) if prompt_length else 0

    simple_transform = _matches(
        normalized,
        r"\b(rewrite|rephrase|format|extract|clean up|copyedit|proofread)\b",
    )
    coding_intent = _matches(
        normalized,
        r"\b(code|coding|repo|repository|implement|implementation|pytest|ruff|"
        r"unit test|tests?|debug|bug|fix the repo|edit .*file|pull request|pr)\b",
    )
    research_intent = _matches(
        normalized,
        r"\b(research|look up|search|browse|cite|citation|sources?|trend|trends)\b",
    )
    current_info_intent = _matches(
        normalized,
        r"\b(current|latest|recent|today|yesterday|now|news|202[0-9]|"
        r"up-to-date|fresh)\b",
    )
    multi_step_reasoning = _matches(
        normalized,
        r"\b(architecture|architect|design|plan|multi-step|strategy|roadmap|"
        r"trade-?offs?|edge cases?|data flow|rollout|migration|system)\b",
    )
    architecture_intent = _matches(
        normalized,
        r"\b(distributed|architecture|scalab|concurren|consensus|throughput|"
        r"backpressure|exactly-once)\b",
    )
    file_intent = _matches(
        normalized,
        r"\b(file|files|folder|directory|write|edit|create|patch)\b",
    )
    email_intent = _matches(normalized, r"\b(emails|mail|inbox)\b")
    calendar_intent = _matches(
        normalized,
        r"\b(calendar|invite|appointment|schedule|reschedule)\b",
    )
    shell_intent = _matches(
        normalized,
        r"\b(shell|terminal|command|execute|run tests?|pytest|ruff|npm|git)\b",
    )
    github_intent = _matches(
        normalized,
        r"\b(github|pull request|pr|issue|branch|commit|merge|git)\b",
    )
    # A generation verb must be close to the image noun (its object), not just
    # somewhere earlier in the prompt; a bare ".*" matched "generate a summary of
    # this image's metadata". "stable diffusion" (the tool) implies generation on
    # its own; bare "diffusion" does not (e.g. "heat diffusion").
    image_generation_intent = _matches(
        normalized,
        r"\b(generate|create|make|draw|render|produce|design)\b"
        r"(?:\s+\w+){0,2}\s+"
        r"(image|picture|photo|illustration|logo|icon|wallpaper|poster)s?\b"
        r"|\bstable diffusion\b",
    )
    vision_intent = _matches(
        normalized,
        r"\b(image|picture|photo|screenshot|screen shot|chart|diagram|graph|"
        r"ocr|vision|visual|scan|extract text from|describe .*image|"
        r"look at .*screenshot)\b",
    )
    tool_intent = _matches(
        normalized,
        r"\b(run|execute|open|browse|search|send|schedule|download|upload|"
        r"install|call|use tool|github|calendar)\b",
    )

    legal_domain = _matches(
        normalized,
        r"\b(legal|lawyer|lawsuit|contract|settlement|liability|compliance|"
        r"regulation|regulations)\b",
    )
    medical_domain = _matches(
        normalized,
        r"\b(medical|doctor|diagnosis|diagnose|treatment|symptom|prescription|"
        r"clinical|health|patient)\b",
    )
    financial_domain = _matches(
        normalized,
        r"\b(financial|finance|tax|taxes|investment|invest|liability|loan|"
        r"insurance|bank|trading|stock|portfolio|payment|invoice|refund)\b",
    )
    sensitive_domain = legal_domain or medical_domain or financial_domain
    production_risk = _matches(
        normalized,
        r"\b(production|prod|customer data|live system|main branch)\b",
    )
    security_risk = _matches(
        normalized,
        r"\b(exploit|malware|xss|csrf|breach|vulnerabilit|sql injection)\b",
    )
    pii_risk = _matches(
        normalized,
        r"\b(ssns?|passports?|passwords?|secrets?|social security|"
        r"credit cards?|api keys?|private keys?)\b",
    )

    destructive_action = _matches(
        normalized,
        r"\b(delete|remove|wipe|destroy|drop|erase|cancel|terminate|purge|"
        r"truncate|shutdown|uninstall|revoke)\b",
    )
    send_action = _matches(
        normalized,
        r"\b(send|post|publish|submit|reply)\b|"
        r"\b(message|email)\s+"
        r"(?!format|marketing|draft|summary|template|copy|ideas|address|board|"
        r"body|subject|header|content|notification|settings|preferences)\w+",
    )
    purchase_action = _matches(
        normalized,
        r"\b(buy|purchase|pay|transfer|wire|subscribe)\b|"
        r"\bbook\s+(?:(?:a|an|the|my|our)\s+)?"
        r"(flight|hotel|room|ticket|appointment|reservation|meeting|call|table|"
        r"ride|trip)\b",
    ) or _order_is_purchase(normalized)
    high_impact_external_action = _matches(
        normalized,
        r"\b(schedule|reschedule|invite|deploy|merge|commit|push)\b|"
        r"\bapply\s+for\b",
    )
    external_action = (
        destructive_action
        or send_action
        or purchase_action
        or high_impact_external_action
    )
    if send_action and _matches(normalized, r"\b(email|message|mail|inbox)\b"):
        email_intent = True
    structured_output = _matches(
        normalized,
        r"\b(json|yaml|csv|table|schema|structured|bullets?|checklist)\b",
    )

    word_count = len(re.findall(r"\b\w+\b", normalized))
    ambiguous = (
        not simple_transform
        and word_count <= 4
        and _matches(
            normalized,
            r"\b(handle|help|fix|manage|do|deal with|this|that|it)\b",
        )
    )
    long_context = estimated_tokens >= 1000 or prompt_length >= 4000
    requires_freshness = bool(research_intent and current_info_intent)
    requires_code_execution = bool(coding_intent and (shell_intent or file_intent))
    requires_vision = bool(vision_intent and not image_generation_intent)
    requires_image_generation = bool(image_generation_intent)
    requires_tools = bool(
        tool_intent
        or file_intent
        or email_intent
        or calendar_intent
        or shell_intent
        or github_intent
        or send_action
        or purchase_action
        or requires_freshness
        or requires_code_execution
        or requires_vision
        or requires_image_generation
    )

    complexity_signals, complexity_reasons = _complexity_signals(
        weights=weights["complexity"],
        estimated_tokens=estimated_tokens,
        simple_transform=simple_transform,
        coding_intent=coding_intent,
        research_intent=research_intent,
        current_info_intent=current_info_intent,
        multi_step_reasoning=multi_step_reasoning,
        architecture_intent=architecture_intent,
        tool_intent=requires_tools,
        structured_output=structured_output,
        ambiguous=ambiguous,
        long_context=long_context,
        sensitive_domain=sensitive_domain,
        vision_intent=requires_vision,
        image_generation_intent=requires_image_generation,
    )
    risk_signals, risk_reasons = _risk_signals(
        weights=weights["risk"],
        destructive_action=destructive_action,
        send_action=send_action,
        purchase_action=purchase_action,
        external_action=external_action,
        file_intent=file_intent,
        shell_intent=shell_intent,
        github_intent=github_intent,
        sensitive_domain=sensitive_domain,
        legal_domain=legal_domain,
        medical_domain=medical_domain,
        financial_domain=financial_domain,
        ambiguous=ambiguous,
        production_risk=production_risk,
        security_risk=security_risk,
        pii_risk=pii_risk,
    )

    complexity = min(
        100,
        10 + _saturate_weight_sum(complexity_signals, config.saturation_k),
    )
    risk = _saturate_weight_sum(risk_signals, config.saturation_k)
    if destructive_action:
        risk = max(risk, 70)
    if high_impact_external_action:
        risk = max(risk, 70)
    if send_action or purchase_action:
        risk = max(risk, 60)

    requires_confirmation = (
        risk >= 70
        or destructive_action
        or send_action
        or purchase_action
        or high_impact_external_action
    )
    confidence = _confidence_score(
        weights=weights["confidence"],
        prompt_length=prompt_length,
        ambiguous=ambiguous,
        high_risk=requires_confirmation,
        has_clear_signal=any(
            (
                simple_transform,
                coding_intent,
                research_intent,
                multi_step_reasoning,
                requires_tools,
                requires_vision,
                requires_image_generation,
                sensitive_domain,
                external_action,
                structured_output,
            )
        ),
    )

    features = PromptFeatures(
        prompt_length=prompt_length,
        estimated_tokens=estimated_tokens,
        simple_transform=simple_transform,
        coding_intent=coding_intent,
        research_intent=research_intent,
        current_info_intent=current_info_intent,
        multi_step_reasoning=multi_step_reasoning,
        tool_intent=tool_intent,
        file_intent=file_intent,
        email_intent=email_intent,
        calendar_intent=calendar_intent,
        shell_intent=shell_intent,
        github_intent=github_intent,
        vision_intent=vision_intent,
        image_generation_intent=image_generation_intent,
        legal_domain=legal_domain,
        medical_domain=medical_domain,
        financial_domain=financial_domain,
        sensitive_domain=sensitive_domain,
        destructive_action=destructive_action,
        external_action=external_action,
        purchase_action=purchase_action,
        send_action=send_action,
        structured_output=structured_output,
        ambiguous=ambiguous,
        long_context=long_context,
        requires_tools=requires_tools,
        requires_freshness=requires_freshness,
        requires_code_execution=requires_code_execution,
        requires_vision=requires_vision,
        requires_image_generation=requires_image_generation,
        requires_confirmation=requires_confirmation,
    )
    reasons = tuple(dict.fromkeys([*complexity_reasons, *risk_reasons]))

    return PromptAnalysis(
        complexity_score=ComplexityScore(
            value=complexity,
            reasons=tuple(complexity_reasons),
        ),
        risk_score=RiskScore(
            value=risk,
            requires_confirmation=requires_confirmation,
            reasons=tuple(risk_reasons),
        ),
        confidence_score=confidence,
        features=features,
        reasons=reasons,
        signals=tuple([*complexity_signals, *risk_signals]),
    )


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


# "order" is verb (purchase) and noun (sorting/idiom). Treat it as a purchase
# only when it is not one of these common non-purchase usages.
_ORDER_NON_PURCHASE = re.compile(
    r"\b(in|alphabetical|chronological|numerical|numeric|reverse|ascending|"
    r"descending|sort|sorted|sorting|random|word|priority|of|that)\s+order\b"
    r"|\border\s+(?:of|to|by|in)\b",
    re.IGNORECASE,
)


def _order_is_purchase(text: str) -> bool:
    return bool(re.search(r"\border\b", text)) and not _ORDER_NON_PURCHASE.search(text)


def _merged_weights(config: ScoringConfig) -> dict[str, dict[str, int]]:
    merged = {
        dimension: dict(weights)
        for dimension, weights in DEFAULT_SCORING_WEIGHTS.items()
    }
    for dimension, overrides in config.weights.items():
        merged.setdefault(dimension, {})
        merged[dimension].update(overrides)
    return merged


def _signal(
    *,
    dimension: str,
    feature: str,
    weights: dict[str, int],
    detail: str,
) -> PromptSignal:
    return PromptSignal(
        dimension=dimension,
        feature=feature,
        weight=weights.get(feature, 0),
        detail=detail,
    )


def _complexity_signals(
    **signals: bool | int | dict[str, int],
) -> tuple[list[PromptSignal], list[str]]:
    weights = signals["weights"]
    assert isinstance(weights, dict)
    found: list[PromptSignal] = []
    reasons: list[str] = []
    estimated_tokens = int(signals["estimated_tokens"])

    if signals["simple_transform"]:
        reasons.append("simple rewrite/extraction/formatting")
    if estimated_tokens > 200:
        found.append(
            _signal(
                dimension="complexity",
                feature="medium_prompt",
                weights=weights,
                detail="medium prompt length",
            )
        )
        reasons.append("medium prompt length")
    if estimated_tokens > 800:
        found.append(
            _signal(
                dimension="complexity",
                feature="long_prompt",
                weights=weights,
                detail="long prompt",
            )
        )
        reasons.append("long prompt")
    if estimated_tokens > 1600:
        found.append(
            _signal(
                dimension="complexity",
                feature="very_long_prompt",
                weights=weights,
                detail="very long prompt",
            )
        )
        reasons.append("very long prompt")
    if signals["coding_intent"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="coding_intent",
                weights=weights,
                detail="coding or repository intent",
            )
        )
        reasons.append("coding or repository intent")
    if signals["research_intent"] or signals["current_info_intent"]:
        feature = (
            "current_info_intent"
            if signals["current_info_intent"]
            else "research_intent"
        )
        found.append(
            _signal(
                dimension="complexity",
                feature=feature,
                weights=weights,
                detail="research or current-information intent",
            )
        )
        reasons.append("research or current-information intent")
    if signals["multi_step_reasoning"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="multi_step_reasoning",
                weights=weights,
                detail="multi-step planning or architecture",
            )
        )
        reasons.append("multi-step planning or architecture")
    if signals["architecture_intent"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="architecture",
                weights=weights,
                detail="architecture or distributed-systems intent",
            )
        )
        reasons.append("architecture or distributed-systems intent")
    if signals["tool_intent"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="tool_intent",
                weights=weights,
                detail="tool use likely",
            )
        )
        reasons.append("tool use likely")
    if signals["structured_output"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="structured_output",
                weights=weights,
                detail="structured output requested",
            )
        )
        reasons.append("structured output requested")
    if signals["ambiguous"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="ambiguous",
                weights=weights,
                detail="ambiguous request",
            )
        )
        reasons.append("ambiguous request")
    if signals["long_context"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="long_context",
                weights=weights,
                detail="long-context need",
            )
        )
        reasons.append("long-context need")
    if signals["sensitive_domain"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="sensitive_domain",
                weights=weights,
                detail="sensitive domain",
            )
        )
        reasons.append("sensitive domain")
    if signals["vision_intent"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="vision_intent",
                weights=weights,
                detail="vision or OCR intent",
            )
        )
        reasons.append("vision or OCR intent")
    if signals["image_generation_intent"]:
        found.append(
            _signal(
                dimension="complexity",
                feature="image_generation_intent",
                weights=weights,
                detail="image generation intent",
            )
        )
        reasons.append("image generation intent")

    return found, reasons


def _risk_signals(**signals: bool | dict[str, int]) -> tuple[list[PromptSignal], list[str]]:
    weights = signals["weights"]
    assert isinstance(weights, dict)
    found: list[PromptSignal] = []
    reasons: list[str] = []

    def add(feature: str, reason: str) -> None:
        found.append(
            _signal(
                dimension="risk",
                feature=feature,
                weights=weights,
                detail=reason,
            )
        )
        reasons.append(reason)

    if signals["destructive_action"]:
        add("destructive_action", "destructive external action")
    if signals["send_action"]:
        add("send_action", "sending or publishing action")
    if signals["purchase_action"]:
        add("purchase_action", "purchase or payment action")
    if signals["external_action"] and not (
        signals["destructive_action"]
        or signals["send_action"]
        or signals["purchase_action"]
    ):
        add("external_action", "external action")
    if signals["file_intent"] or signals["shell_intent"] or signals["github_intent"]:
        add("file_shell_github", "file, shell, or GitHub operation")
    if signals["sensitive_domain"]:
        add("sensitive_domain", "sensitive legal, medical, or financial domain")
    if signals["legal_domain"]:
        add("legal_domain", "legal sensitivity")
    if signals["medical_domain"]:
        add("medical_domain", "medical sensitivity")
    if signals["financial_domain"]:
        add("financial_domain", "financial sensitivity")
    if signals["production_risk"]:
        add("production", "production or live-system risk")
    if signals["security_risk"]:
        add("security", "security-sensitive request")
    if signals["pii_risk"]:
        add("pii", "private data or credential sensitivity")
    if signals["ambiguous"] and (
        signals["sensitive_domain"] or signals["external_action"]
    ):
        add("ambiguous_high_impact", "ambiguous high-impact request")

    return found, reasons


def _saturate_weight_sum(signals: list[PromptSignal], saturation_k: int) -> int:
    total = sum(signal.weight for signal in signals)
    if total <= 0:
        return 0
    return round((total / (total + saturation_k)) * 100)


def _confidence_score(
    *,
    weights: dict[str, int],
    prompt_length: int,
    ambiguous: bool,
    high_risk: bool,
    has_clear_signal: bool,
) -> int:
    score = 90
    if ambiguous:
        score -= weights.get("ambiguous", 25)
    if high_risk and prompt_length < 80:
        score -= weights.get("short_high_risk", 15)
    if not has_clear_signal:
        score -= weights.get("weak_feature_match", 10)
    if prompt_length == 0:
        score -= weights.get("empty_prompt", 40)
    return max(0, min(score, 100))
