"""Fast deterministic prompt scoring for model routing."""

from __future__ import annotations

import math
import re

from hermes.plugins.model_router.models import (
    ComplexityScore,
    PromptAnalysis,
    PromptFeatures,
    RiskScore,
)


def score_prompt(prompt: str) -> PromptAnalysis:
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
    file_intent = _matches(
        normalized,
        r"\b(file|files|folder|directory|write|edit|create|patch)\b",
    )
    email_intent = _matches(normalized, r"\b(email|emails|mail|inbox)\b")
    calendar_intent = _matches(
        normalized,
        r"\b(calendar|meeting|invite|appointment|schedule|reschedule)\b",
    )
    shell_intent = _matches(
        normalized,
        r"\b(shell|terminal|command|execute|run tests?|pytest|ruff|npm|git)\b",
    )
    github_intent = _matches(
        normalized,
        r"\b(github|pull request|pr|issue|branch|commit|merge|git)\b",
    )
    image_generation_intent = _matches(
        normalized,
        r"\b(generate|create|make|draw|render|produce|design)\b"
        r".*\b(image|picture|photo|illustration|logo|icon|wallpaper|poster|"
        r"diffusion|stable diffusion)\b",
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
        r"install|call|use tool|github|calendar|email)\b",
    )

    legal_domain = _matches(
        normalized,
        r"\b(legal|lawyer|lawsuit|contract|settlement|liability|compliance|"
        r"regulation|regulations)\b",
    )
    medical_domain = _matches(
        normalized,
        r"\b(medical|doctor|diagnosis|diagnose|treatment|symptom|prescription|"
        r"clinical|health)\b",
    )
    financial_domain = _matches(
        normalized,
        r"\b(financial|finance|tax|taxes|investment|invest|liability|loan|"
        r"insurance|bank|trading|stock|portfolio)\b",
    )
    sensitive_domain = legal_domain or medical_domain or financial_domain

    destructive_action = _matches(
        normalized,
        r"\b(delete|remove|wipe|destroy|drop|erase|cancel|terminate|purge)\b",
    )
    send_action = _matches(
        normalized,
        r"\b(send|message|post|publish|submit|reply)\b",
    )
    purchase_action = _matches(
        normalized,
        r"\b(buy|purchase|order|pay|transfer|wire|book|subscribe)\b",
    )
    external_action = (
        destructive_action
        or send_action
        or purchase_action
        or _matches(
            normalized,
            r"\b(schedule|reschedule|invite|apply|deploy|merge|commit|push)\b",
        )
    )
    structured_output = _matches(
        normalized,
        r"\b(json|yaml|csv|table|schema|structured|bullets?|checklist)\b",
    )

    word_count = len(re.findall(r"\b\w+\b", normalized))
    ambiguous = (
        not simple_transform
        and word_count <= 4
        and _matches(normalized, r"\b(handle|help|fix|manage|do|deal with|this|that|it)\b")
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
        or requires_freshness
        or requires_code_execution
        or requires_vision
        or requires_image_generation
    )

    complexity, complexity_reasons = _complexity_score(
        estimated_tokens=estimated_tokens,
        simple_transform=simple_transform,
        coding_intent=coding_intent,
        research_intent=research_intent,
        current_info_intent=current_info_intent,
        multi_step_reasoning=multi_step_reasoning,
        tool_intent=requires_tools,
        structured_output=structured_output,
        ambiguous=ambiguous,
        long_context=long_context,
        sensitive_domain=sensitive_domain,
        vision_intent=requires_vision,
        image_generation_intent=requires_image_generation,
    )
    risk, risk_reasons = _risk_score(
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
    )
    requires_confirmation = (
        risk >= 70 or destructive_action or send_action or purchase_action
    )
    confidence = _confidence_score(
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
    )


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _complexity_score(**signals: bool | int) -> tuple[int, list[str]]:
    score = 10
    reasons: list[str] = []
    estimated_tokens = int(signals["estimated_tokens"])

    if signals["simple_transform"]:
        reasons.append("simple rewrite/extraction/formatting")
    if estimated_tokens > 200:
        score += 10
        reasons.append("medium prompt length")
    if estimated_tokens > 800:
        score += 15
        reasons.append("long prompt")
    if estimated_tokens > 1600:
        score += 15
        reasons.append("very long prompt")
    if signals["coding_intent"]:
        score += 25
        reasons.append("coding or repository intent")
    if signals["research_intent"] or signals["current_info_intent"]:
        score += 20
        reasons.append("research or current-information intent")
    if signals["multi_step_reasoning"]:
        score += 25
        reasons.append("multi-step planning or architecture")
    if signals["tool_intent"]:
        score += 10
        reasons.append("tool use likely")
    if signals["structured_output"]:
        score += 8
        reasons.append("structured output requested")
    if signals["ambiguous"]:
        score += 12
        reasons.append("ambiguous request")
    if signals["long_context"]:
        score += 20
        reasons.append("long-context need")
    if signals["sensitive_domain"]:
        score += 8
        reasons.append("sensitive domain")
    if signals["vision_intent"]:
        score += 15
        reasons.append("vision or OCR intent")
    if signals["image_generation_intent"]:
        score += 18
        reasons.append("image generation intent")

    return min(score, 100), reasons


def _risk_score(**signals: bool) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if signals["destructive_action"]:
        score += 70
        reasons.append("destructive external action")
    if signals["send_action"]:
        score += 60
        reasons.append("sending or publishing action")
    if signals["purchase_action"]:
        score += 60
        reasons.append("purchase or payment action")
    if signals["external_action"] and not (
        signals["destructive_action"]
        or signals["send_action"]
        or signals["purchase_action"]
    ):
        score += 40
        reasons.append("external action")
    if signals["file_intent"] or signals["shell_intent"] or signals["github_intent"]:
        score += 25
        reasons.append("file, shell, or GitHub operation")
    if signals["sensitive_domain"]:
        score += 25
        reasons.append("sensitive legal, medical, or financial domain")
    if signals["legal_domain"]:
        score += 5
        reasons.append("legal sensitivity")
    if signals["medical_domain"]:
        score += 5
        reasons.append("medical sensitivity")
    if signals["financial_domain"]:
        score += 5
        reasons.append("financial sensitivity")
    if signals["ambiguous"] and (
        signals["sensitive_domain"] or signals["external_action"]
    ):
        score += 15
        reasons.append("ambiguous high-impact request")

    return min(score, 100), reasons


def _confidence_score(
    *,
    prompt_length: int,
    ambiguous: bool,
    high_risk: bool,
    has_clear_signal: bool,
) -> int:
    score = 90
    if ambiguous:
        score -= 25
    if high_risk and prompt_length < 80:
        score -= 15
    if not has_clear_signal:
        score -= 10
    if prompt_length == 0:
        score -= 40
    return max(0, min(score, 100))
