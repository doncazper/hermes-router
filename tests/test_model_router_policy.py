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


def _config_path(tmp_path: Path, overrides: dict[str, dict] | None = None) -> Path:
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
    path.write_text(
        yaml.safe_dump(
            {
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
        ),
        encoding="utf-8",
    )
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
        "Design a multi-step architecture plan for a Hermes plugin with data flow, "
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
        "Generate an image of a Hermes router dashboard.",
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
