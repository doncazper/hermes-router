from pathlib import Path

import yaml

from hermes.plugins.model_router.config import REQUIRED_ENGINE_CATEGORIES
from hermes.plugins.model_router.policy import route_prompt


def _engine(
    name: str,
    *,
    enabled: bool = True,
    fallback: str | None = None,
    supports_tools: bool | None = None,
    modalities: list[str] | None = None,
    capability: int | None = None,
    trust: int | None = None,
    cost: int | None = None,
    latency: int | None = None,
) -> dict:
    data = {
        "provider": "local" if name != "human_confirm" else "human",
        "model": f"{name}-model",
        "adapter": name,
        "strengths": [name],
        "max_context": 8192,
        "cost_tier": "low",
        "latency_tier": "low",
        "enabled": enabled,
        "fallback": fallback,
        "supports_tools": supports_tools
        if supports_tools is not None
        else name
        in {
            "claude_code",
            "codex",
            "code_agent",
            "web_research",
            "multimodal_vision",
            "image_generation",
            "human_confirm",
        },
    }
    if modalities is not None:
        data["modalities"] = modalities
    if capability is not None:
        data["capability"] = capability
    if trust is not None:
        data["trust"] = trust
    if cost is not None:
        data["cost"] = cost
    if latency is not None:
        data["latency"] = latency
    return data


def _config_path(
    tmp_path: Path,
    overrides: dict[str, dict] | None = None,
    *,
    safety: dict | None = None,
    provider_policy: dict | None = None,
) -> Path:
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
    for name, patch in (overrides or {}).items():
        if name in engines:
            engines[name].update(patch)
        else:
            engines[name] = _engine(name, fallback="code_agent") | patch
    path = tmp_path / "model_router.yaml"
    data = {
        "routing_targets": {
            "simple": "fast_local",
            "balanced": "balanced_local",
            "reasoning": "reasoning_local",
            "coding": "code_agent",
            "research": "web_research",
            "vision": "multimodal_vision",
            "image_generation": "image_generation",
            "confirmation": "human_confirm",
        },
        "engines": engines,
    }
    if safety is not None:
        data["safety"] = safety
    if provider_policy is not None:
        data["provider_policy"] = provider_policy
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_simple_rewrite_routes_to_fast_local(tmp_path):
    decision = route_prompt("rewrite this text", config_path=_config_path(tmp_path))

    assert decision.selected_engine == "fast_local"
    assert decision.requires_confirmation is False


def test_normal_summarization_routes_to_balanced_local(tmp_path):
    decision = route_prompt(
        "Summarize the attached meeting notes into three bullets.",
        config_path=_config_path(tmp_path),
    )

    assert decision.selected_engine == "balanced_local"


def test_complex_architecture_prompt_routes_to_reasoning_local(tmp_path):
    prompt = (
        "Design a multi-step architecture plan for an agent plugin with data flow, "
        "edge cases, testing strategy, and rollout notes."
    )
    decision = route_prompt(prompt, config_path=_config_path(tmp_path))

    assert decision.selected_engine == "reasoning_local"


def test_repo_coding_implementation_routes_to_code_agent(tmp_path):
    decision = route_prompt(
        "Fix the repo, edit the Python files, and run tests.",
        config_path=_config_path(tmp_path),
    )

    assert decision.selected_engine == "code_agent"
    assert decision.requires_code_execution is True
    assert decision.requires_tools is True


def test_user_can_route_coding_to_claude_code(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "claude_code": {
                "provider": "anthropic",
                "model": "claude-code",
                "adapter": "claude_code",
                "strengths": ["repository edits", "tests"],
                "enabled": True,
                "fallback": "code_agent",
            }
        },
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["coding"] = "claude_code"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    decision = route_prompt(
        "Implement the feature in this repository and run tests.",
        config_path=path,
    )

    assert decision.selected_engine == "claude_code"
    assert decision.fallback_engine == "code_agent"


def test_current_research_routes_to_web_research(tmp_path):
    decision = route_prompt(
        "Research current GLP-1 supplement trends and include citations.",
        config_path=_config_path(tmp_path),
    )

    assert decision.selected_engine == "web_research"
    assert decision.requires_freshness is True


def test_screenshot_ocr_routes_to_multimodal_vision(tmp_path):
    decision = route_prompt(
        "Extract the text from this screenshot and summarize the chart.",
        config_path=_config_path(tmp_path),
    )

    assert decision.selected_engine == "multimodal_vision"
    assert decision.requires_vision is True
    assert decision.requires_tools is True


def test_image_generation_routes_to_image_generation_engine(tmp_path):
    decision = route_prompt(
        "Generate an image of a ModelRouter dashboard.",
        config_path=_config_path(tmp_path),
    )

    assert decision.selected_engine == "image_generation"
    assert decision.requires_image_generation is True
    assert decision.requires_tools is True


def test_ambiguous_high_impact_prompt_does_not_route_to_weak_engine(tmp_path):
    decision = route_prompt("Handle my taxes.", config_path=_config_path(tmp_path))

    assert decision.selected_engine in {"reasoning_local", "human_confirm"}
    assert decision.selected_engine != "fast_local"


def test_high_impact_external_actions_require_confirmation(tmp_path):
    path = _config_path(tmp_path)
    prompts = (
        "deploy to production",
        "merge this pull request",
        "push to main",
        "schedule a meeting",
        "apply for this job",
    )

    for prompt in prompts:
        decision = route_prompt(prompt, config_path=path)

        assert decision.selected_engine == "human_confirm"
        assert decision.requires_confirmation is True
        assert decision.risk_score >= 70


def test_confirmation_overrides_can_allow_send_actions(tmp_path):
    path = _config_path(
        tmp_path,
        safety={
            "require_human_confirmation": True,
            "confirmation_overrides": {"allow_send_actions": True},
        },
    )

    decision = route_prompt("send the project update", config_path=path)

    assert decision.selected_engine != "human_confirm"
    assert decision.requires_confirmation is False
    assert decision.features.send_action is True


def test_confirmation_overrides_stay_scoped(tmp_path):
    path = _config_path(
        tmp_path,
        safety={
            "require_human_confirmation": True,
            "confirmation_overrides": {"allow_send_actions": True},
        },
    )

    decision = route_prompt("delete all my emails", config_path=path)

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True


def test_confirmation_can_be_disabled_explicitly(tmp_path):
    decision = route_prompt(
        "deploy to production",
        config_path=_config_path(
            tmp_path,
            safety={"require_human_confirmation": False},
        ),
    )

    assert decision.selected_engine != "human_confirm"
    assert decision.requires_confirmation is False
    assert decision.features.high_impact_external_action is True


def test_invalid_safety_config_fails_closed(tmp_path):
    decision = route_prompt(
        "rewrite this text",
        config_path=_config_path(
            tmp_path,
            safety={"confirmation_overrides": {"allow_send_actions": "yes"}},
        ),
    )

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert decision.config_valid is False


def test_missing_config_fails_closed_to_human_confirm(tmp_path):
    decision = route_prompt("rewrite this text", config_path=tmp_path / "missing.yaml")

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert decision.config_valid is False
    assert any("fail-closed" in reason for reason in decision.reasons)


def test_disabled_tool_engine_fails_closed_when_fallback_cannot_run_tools(tmp_path):
    decision = route_prompt(
        "Fix the repo and run tests.",
        config_path=_config_path(
            tmp_path,
            {"code_agent": {"enabled": False, "fallback": "reasoning_local"}},
        ),
    )

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert any("fallback" in reason for reason in decision.reasons)
    assert any(
        rejection.engine == "reasoning_local" and "tools required" in rejection.reason
        for rejection in decision.rejected_engines
    )


def test_force_engine_hint_routes_to_known_engine(tmp_path):
    decision = route_prompt(
        "rewrite this text",
        config_path=_config_path(tmp_path),
        hints={"force_engine": "reasoning_local"},
    )

    assert decision.selected_engine == "reasoning_local"
    assert any("forced engine reasoning_local" in reason for reason in decision.reasons)


def test_force_engine_hint_cannot_bypass_confirmation(tmp_path):
    decision = route_prompt(
        "delete all my emails",
        config_path=_config_path(tmp_path),
        hints={"force_engine": "fast_local"},
    )

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert any("force_engine ignored" in reason for reason in decision.reasons)


def test_routing_profiles_do_not_bypass_human_confirm(tmp_path):
    path = _config_path(tmp_path)

    for profile in ("fast", "balanced", "quality", "private", "safe"):
        decision = route_prompt(
            "delete all my emails",
            config_path=path,
            hints={"profile": profile, "force_engine": "fast_local"},
        )

        assert decision.routing_profile == profile
        assert decision.selected_engine == "human_confirm"
        assert decision.requires_confirmation is True


def test_private_profile_excludes_hosted_provider_and_explains_receipt(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "hosted_reasoning": {
                "provider": "openai",
                "model": "hosted-reasoning",
                "adapter": "openai",
                "strengths": ["reasoning"],
                "enabled": True,
                "fallback": "reasoning_local",
                "cost_tier": "paid",
                "latency_tier": "medium",
            }
        },
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["reasoning"] = "hosted_reasoning"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=path,
        hints={"profile": "private"},
    )

    assert decision.routing_profile == "private"
    assert decision.selected_engine == "reasoning_local"
    assert decision.requirements.allowed_providers == ("local", "human")
    assert any(
        rejection.engine == "hosted_reasoning"
        and "private routing profile" in rejection.reason
        and "local-only" in rejection.reason
        for rejection in decision.rejected_engines
    )
    assert any("local-only provider policy" in reason for reason in decision.reasons)


def test_provider_allowlist_excludes_non_allowed_provider(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "hosted_reasoning": {
                "provider": "openai",
                "model": "hosted-reasoning",
                "adapter": "openai",
                "strengths": ["reasoning"],
                "enabled": True,
                "fallback": "reasoning_local",
            }
        },
        provider_policy={"version": 1, "provider_allowlist": ["local"]},
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["reasoning"] = "hosted_reasoning"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=path,
    )

    assert decision.selected_engine == "reasoning_local"
    assert decision.requirements.allowed_providers == ("local", "human")
    assert any("provider policy allowlist: local" in reason for reason in decision.reasons)
    assert any(
        rejection.engine == "hosted_reasoning"
        and rejection.reason == "provider openai not allowed by provider policy"
        for rejection in decision.rejected_engines
    )


def test_provider_denylist_excludes_denied_provider(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "hosted_reasoning": {
                "provider": "openai",
                "model": "hosted-reasoning",
                "adapter": "openai",
                "strengths": ["reasoning"],
                "enabled": True,
                "fallback": "reasoning_local",
            }
        },
        provider_policy={"version": 1, "provider_denylist": ["openai"]},
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["reasoning"] = "hosted_reasoning"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=path,
    )

    assert decision.selected_engine == "reasoning_local"
    assert decision.requirements.denied_providers == ("openai",)
    assert any("provider policy denylist: openai" in reason for reason in decision.reasons)
    assert any(
        rejection.engine == "hosted_reasoning"
        and rejection.reason == "provider openai denied by provider policy"
        for rejection in decision.rejected_engines
    )


def test_provider_max_tier_policy_cannot_be_loosened_by_hints(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "reasoning_local": {
                "cost_tier": "high",
                "fallback": "balanced_local",
            },
            "balanced_local": {
                "cost_tier": "low",
            },
        },
        provider_policy={"version": 1, "max_cost_tier": "low"},
    )

    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=path,
        hints={"max_cost_tier": "high"},
    )

    assert decision.selected_engine == "balanced_local"
    assert decision.requirements.max_cost_tier == "low"
    assert any("provider policy max cost tier: low" in reason for reason in decision.reasons)
    assert any(
        rejection.engine == "reasoning_local"
        and rejection.reason == "cost_tier high exceeds low"
        for rejection in decision.rejected_engines
    )


def test_provider_route_pools_apply_per_route_constraints(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "hosted_fast": {
                "provider": "openai",
                "model": "hosted-fast",
                "adapter": "openai",
                "strengths": ["rewrite"],
                "enabled": True,
                "fallback": "fast_local",
            },
            "hosted_reasoning": {
                "provider": "openai",
                "model": "hosted-reasoning",
                "adapter": "openai",
                "strengths": ["reasoning"],
                "enabled": True,
                "fallback": "reasoning_local",
            },
        },
        provider_policy={
            "version": 1,
            "route_pools": {
                "simple": {"local_only": True},
                "reasoning": {"hosted_allowed": True},
            },
        },
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["simple"] = "hosted_fast"
    data["routing_targets"]["reasoning"] = "hosted_reasoning"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    simple = route_prompt("rewrite this text", config_path=path)
    reasoning = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=path,
    )

    assert simple.selected_engine == "fast_local"
    assert simple.requirements.allowed_providers == ("local", "human")
    assert any("route pool applied for simple" in reason for reason in simple.reasons)
    assert reasoning.selected_engine == "hosted_reasoning"
    assert any(
        "route pool applied for reasoning" in reason for reason in reasoning.reasons
    )


def test_provider_fallback_chain_does_not_escape_denied_provider(tmp_path):
    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=_config_path(
            tmp_path,
            {
                "reasoning_local": {
                    "enabled": False,
                    "fallback": "hosted_reasoning",
                },
                "hosted_reasoning": {
                    "provider": "openai",
                    "model": "hosted-reasoning",
                    "adapter": "openai",
                    "strengths": ["reasoning"],
                    "enabled": True,
                    "fallback": "human_confirm",
                },
            },
            provider_policy={"version": 1, "provider_denylist": ["openai"]},
        ),
    )

    assert decision.selected_engine == "human_confirm"
    assert any(
        rejection.engine == "hosted_reasoning"
        and rejection.reason == "provider openai denied by provider policy"
        for rejection in decision.rejected_engines
    )


def test_forced_engine_cannot_bypass_provider_policy(tmp_path):
    decision = route_prompt(
        "rewrite this text",
        config_path=_config_path(
            tmp_path,
            {
                "hosted_reasoning": {
                    "provider": "openai",
                    "model": "hosted-reasoning",
                    "adapter": "openai",
                    "strengths": ["reasoning"],
                    "enabled": True,
                    "fallback": "fast_local",
                }
            },
        ),
        hints={
            "force_engine": "hosted_reasoning",
            "provider_denylist": ["openai"],
        },
    )

    assert decision.selected_engine == "fast_local"
    assert any("forced engine hosted_reasoning" in reason for reason in decision.reasons)
    assert any(
        rejection.engine == "hosted_reasoning"
        and rejection.reason == "provider openai denied by provider policy"
        for rejection in decision.rejected_engines
    )


def test_human_confirm_remains_reachable_under_restrictive_provider_policy(tmp_path):
    decision = route_prompt(
        "delete all my emails",
        config_path=_config_path(
            tmp_path,
            provider_policy={
                "version": 1,
                "provider_allowlist": ["openai"],
                "provider_denylist": ["human"],
            },
        ),
    )

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert "human" in decision.requirements.allowed_providers
    assert "human" not in decision.requirements.denied_providers


def test_fast_profile_applies_cost_and_latency_constraints(tmp_path):
    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=_config_path(
            tmp_path,
            {
                "reasoning_local": {
                    "cost_tier": "paid",
                    "latency_tier": "high",
                    "fallback": "balanced_local",
                },
                "balanced_local": {
                    "cost_tier": "low",
                    "latency_tier": "medium",
                },
            },
        ),
        hints={"profile": "fast"},
    )

    assert decision.routing_profile == "fast"
    assert decision.selected_engine == "balanced_local"
    assert decision.requirements.max_cost_tier == "low"
    assert decision.requirements.max_latency_tier == "medium"
    assert any(
        rejection.engine == "reasoning_local"
        and (
            "cost_tier" in rejection.reason
            or "latency_tier" in rejection.reason
        )
        for rejection in decision.rejected_engines
    )


def test_quality_profile_allows_configured_hosted_backend(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "hosted_reasoning": {
                "provider": "openai",
                "model": "hosted-reasoning",
                "adapter": "openai",
                "strengths": ["reasoning"],
                "enabled": True,
                "fallback": "reasoning_local",
                "cost_tier": "paid",
                "latency_tier": "medium",
            }
        },
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["reasoning"] = "hosted_reasoning"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=path,
        hints={"profile": "quality"},
    )

    assert decision.routing_profile == "quality"
    assert decision.selected_engine == "hosted_reasoning"
    assert decision.requirements.allowed_providers == ()


def test_safe_profile_confirms_ambiguous_sensitive_prompt(tmp_path):
    decision = route_prompt(
        "Handle my taxes.",
        config_path=_config_path(tmp_path),
        hints={"profile": "safe"},
    )

    assert decision.routing_profile == "safe"
    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert any("profile safe" in reason for reason in decision.reasons)


def test_unknown_force_engine_fails_closed(tmp_path):
    decision = route_prompt(
        "rewrite this text",
        config_path=_config_path(tmp_path),
        hints={"force_engine": "missing_engine"},
    )

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert any("unknown forced engine" in reason for reason in decision.reasons)


def test_toolless_target_is_rejected_with_reason(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "fast_local": {
                "fallback": "web_research",
                "supports_tools": False,
            },
            "web_research": {
                "supports_tools": True,
            },
        },
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["research"] = "fast_local"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    decision = route_prompt(
        "Research current GLP-1 supplement trends and include citations.",
        config_path=path,
    )

    assert decision.selected_engine == "web_research"
    assert any(
        rejection.engine == "fast_local" and "tools required" in rejection.reason
        for rejection in decision.rejected_engines
    )
    assert all(alternative.engine != "fast_local" for alternative in decision.alternatives)


def test_image_attachment_hint_routes_to_vision(tmp_path):
    decision = route_prompt(
        "summarize this attachment",
        config_path=_config_path(tmp_path),
        hints={"attachments": ["image"]},
    )

    assert decision.selected_engine == "multimodal_vision"
    assert decision.requirements.required_modalities == ("image",)


def test_latency_sensitive_hint_rejects_high_latency_target(tmp_path):
    decision = route_prompt(
        (
            "Design a multi-step architecture plan with data flow, edge cases, "
            "testing strategy, and rollout notes."
        ),
        config_path=_config_path(
            tmp_path,
            {
                "reasoning_local": {
                    "latency_tier": "high",
                    "fallback": "balanced_local",
                },
                "balanced_local": {
                    "latency_tier": "medium",
                },
            },
        ),
        hints={"latency_sensitive": True},
    )

    assert decision.selected_engine == "balanced_local"
    assert decision.requirements.max_latency_tier == "medium"
    assert any(
        rejection.engine == "reasoning_local" and "latency_tier" in rejection.reason
        for rejection in decision.rejected_engines
    )


def test_ranked_alternatives_are_returned_sorted_by_rank(tmp_path):
    decision = route_prompt(
        "rewrite this text",
        config_path=_config_path(
            tmp_path,
            {
                "balanced_local": {
                    "capability": 70,
                    "trust": 70,
                    "cost": 20,
                    "latency": 20,
                },
                "reasoning_local": {
                    "capability": 95,
                    "trust": 85,
                    "cost": 60,
                    "latency": 70,
                },
            },
        ),
    )

    ranks = [alternative.rank_score for alternative in decision.alternatives]
    assert ranks == sorted(ranks, reverse=True)
    assert decision.alternatives
    assert all(alternative.engine != decision.selected_engine for alternative in decision.alternatives)
    assert all(
        alternative.engine not in {"human_confirm", "intent_router"}
        for alternative in decision.alternatives
    )


def test_target_engine_remains_selected_even_when_alternative_ranks_higher(tmp_path):
    decision = route_prompt(
        "rewrite this text",
        config_path=_config_path(
            tmp_path,
            {
                "fast_local": {
                    "capability": 30,
                    "trust": 40,
                    "cost": 5,
                    "latency": 5,
                },
                "reasoning_local": {
                    "capability": 100,
                    "trust": 100,
                    "cost": 50,
                    "latency": 70,
                },
            },
        ),
    )

    assert decision.selected_engine == "fast_local"
    assert decision.alternatives[0].engine == "reasoning_local"


def test_latency_sensitive_hint_changes_alternative_order(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "balanced_local": {
                "capability": 80,
                "trust": 80,
                "cost": 30,
                "latency": 70,
            },
            "reasoning_local": {
                "capability": 70,
                "trust": 70,
                "cost": 30,
                "latency": 10,
            },
        },
    )

    normal = route_prompt("rewrite this text", config_path=path)
    fast = route_prompt(
        "rewrite this text",
        config_path=path,
        hints={"latency_sensitive": True},
    )

    assert normal.selected_engine == "fast_local"
    assert fast.selected_engine == "fast_local"
    assert normal.alternatives[0].engine == "balanced_local"
    assert fast.alternatives[0].engine == "reasoning_local"
