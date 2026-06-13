from pathlib import Path

import yaml

from hermes.plugins.model_router import ModelRouter
from hermes.plugins.model_router.config import REQUIRED_ENGINE_CATEGORIES
from hermes.plugins.model_router.policy import route_prompt


def _engine(name: str, *, fallback: str | None = None) -> dict:
    return {
        "provider": "local" if name != "human_confirm" else "human",
        "model": f"{name}-model",
        "adapter": name,
        "strengths": [name],
        "max_context": 8192,
        "cost_tier": "low",
        "latency_tier": "low",
        "enabled": True,
        "fallback": fallback,
        "supports_tools": name
        in {
            "code_agent",
            "web_research",
            "multimodal_vision",
            "image_generation",
            "human_confirm",
        },
        "modalities": ["image"] if name == "multimodal_vision" else [],
    }


def _config_path(tmp_path: Path, *, scoring: dict | None = None) -> Path:
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
        "engines": {
            name: _engine(name, fallback=fallbacks[name])
            for name in REQUIRED_ENGINE_CATEGORIES
        },
    }
    if scoring is not None:
        data["scoring"] = scoring
    path = tmp_path / "model_router.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_initialized_router_matches_route_prompt_for_existing_cases(tmp_path):
    path = _config_path(tmp_path)
    router = ModelRouter.from_config(path)

    assert router.route("rewrite this text").selected_engine == route_prompt(
        "rewrite this text",
        config_path=path,
    ).selected_engine
    assert router.route("fix the repo and run tests").selected_engine == route_prompt(
        "fix the repo and run tests",
        config_path=path,
    ).selected_engine


def test_initialized_router_routes_after_config_file_is_removed(tmp_path):
    path = _config_path(tmp_path)
    router = ModelRouter.from_config(path)
    path.unlink()

    first = router.route("rewrite this text")
    second = router.route("fix the repo and run tests")

    assert first.selected_engine == "fast_local"
    assert second.selected_engine == "code_agent"


def test_invalid_scoring_config_fails_closed_through_compatibility_wrapper(tmp_path):
    path = _config_path(tmp_path, scoring={"saturation_k": 0})

    decision = route_prompt("rewrite this text", config_path=path)

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert decision.config_valid is False
    assert any("scoring" in reason for reason in decision.reasons)
