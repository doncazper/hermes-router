import inspect
from pathlib import Path

import yaml

import hermes.plugins.model_router as public_api
import hermes.plugins.model_router.policy as policy
from hermes.plugins.model_router import ModelRouter
from hermes.plugins.model_router.config import REQUIRED_ENGINE_CATEGORIES
from hermes.plugins.model_router.models import RoutingDecision


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


def _config_path(tmp_path: Path) -> Path:
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
                "engines": {
                    name: _engine(name, fallback=fallbacks[name])
                    for name in REQUIRED_ENGINE_CATEGORIES
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_public_api_exports_stable_router_surface():
    assert public_api.__all__ == [
        "ModelRouter",
        "build_dispatch_plan",
        "route_prompt",
        "score_prompt",
    ]
    assert public_api.ModelRouter is ModelRouter


def test_route_fast_is_production_api_and_route_is_diagnostic_api(tmp_path):
    router = ModelRouter.from_config(_config_path(tmp_path))

    fast_result = router.route_fast("fix the repo and run tests")
    diagnostic_result = router.route("fix the repo and run tests")

    assert isinstance(fast_result, str)
    assert fast_result == "code_agent"
    assert isinstance(diagnostic_result, RoutingDecision)
    assert diagnostic_result.selected_engine == fast_result
    assert diagnostic_result.reasons
    assert diagnostic_result.features is not None


def test_route_fast_does_not_call_rich_scorer(monkeypatch, tmp_path):
    router = ModelRouter.from_config(_config_path(tmp_path))

    def explode(*args, **kwargs):
        raise AssertionError("route_fast must not call score_prompt")

    monkeypatch.setattr("hermes.plugins.model_router.policy.score_prompt", explode)

    assert router.route_fast("rewrite this text") == "fast_local"


def test_route_fast_source_has_no_hot_path_logging_or_scorer_call():
    hot_path_objects = (
        policy.ModelRouter.route_fast,
        policy.ModelRouter._resolve_target_fast,
        policy.ModelRouter._resolve_engine_fast,
        policy._fast_target_route_index,
        policy._fast_has_confirmation_word,
    )
    source = "\n".join(inspect.getsource(obj) for obj in hot_path_objects)

    assert "score_prompt" not in source
    assert "logging" not in source
    assert "logger" not in source


def test_route_diagnostics_can_skip_alternatives(tmp_path):
    router = ModelRouter.from_config(_config_path(tmp_path))

    with_alternatives = router.route("rewrite this text")
    without_alternatives = router.route("rewrite this text", include_alternatives=False)

    assert with_alternatives.selected_engine == without_alternatives.selected_engine
    assert with_alternatives.alternatives
    assert without_alternatives.alternatives == ()
