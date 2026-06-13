from pathlib import Path

import yaml

from hermes.plugins.model_router.config import REQUIRED_ENGINE_CATEGORIES
from hermes.plugins.model_router.policy import route_prompt


def _engine(
    name: str,
    *,
    enabled: bool = True,
    fallback: str | None = None,
) -> dict:
    return {
        "provider": "local" if name != "human_confirm" else "human",
        "model": f"{name}-model",
        "adapter": name,
        "strengths": [name],
        "max_context": 8192,
        "cost_tier": "low",
        "latency_tier": "low",
        "enabled": enabled,
        "fallback": fallback,
    }


def _config_path(tmp_path: Path, overrides: dict[str, dict] | None = None) -> Path:
    fallbacks = {
        "fast_local": "balanced_local",
        "balanced_local": "reasoning_local",
        "reasoning_local": "human_confirm",
        "code_agent": "reasoning_local",
        "web_research": "reasoning_local",
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


def test_ambiguous_high_impact_prompt_does_not_route_to_weak_engine(tmp_path):
    decision = route_prompt("Handle my taxes.", config_path=_config_path(tmp_path))

    assert decision.selected_engine in {"reasoning_local", "human_confirm"}
    assert decision.selected_engine != "fast_local"


def test_missing_config_fails_closed_to_human_confirm(tmp_path):
    decision = route_prompt("rewrite this text", config_path=tmp_path / "missing.yaml")

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert decision.config_valid is False
    assert any("fail-closed" in reason for reason in decision.reasons)


def test_disabled_engine_follows_fallback_chain(tmp_path):
    decision = route_prompt(
        "Fix the repo and run tests.",
        config_path=_config_path(
            tmp_path,
            {"code_agent": {"enabled": False, "fallback": "reasoning_local"}},
        ),
    )

    assert decision.selected_engine == "reasoning_local"
    assert decision.fallback_engine == "reasoning_local"
    assert any("fallback" in reason for reason in decision.reasons)
