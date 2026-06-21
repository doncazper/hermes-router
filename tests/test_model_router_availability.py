from pathlib import Path

import yaml

from hermes.plugins.model_router.availability import (
    validate_engine_availability,
    validate_router_availability,
)
from hermes.plugins.model_router.config import REQUIRED_ENGINE_CATEGORIES, load_router_config
from hermes.plugins.model_router.models import RouterAvailabilityReport
from hermes.plugins.model_router.policy import route_prompt


def _engine(
    name: str,
    *,
    fallback: str | None = None,
    availability: dict | None = None,
) -> dict:
    data = {
        "provider": "local" if name != "human_confirm" else "human",
        "model": f"{name}-model",
        "adapter": name,
        "strengths": [name],
        "max_context": 8192,
        "cost_tier": "low",
        "latency_tier": "low",
        "enabled": True,
        "fallback": fallback,
    }
    if availability is not None:
        data["availability"] = availability
    return data


def _config_path(tmp_path: Path, engines: dict[str, dict]) -> Path:
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
    base_engines = {
        name: _engine(name, fallback=fallbacks[name])
        for name in REQUIRED_ENGINE_CATEGORIES
    }
    base_engines.update(engines)
    data = {
        "routing_targets": {
            "simple": "fast_local",
            "balanced": "balanced_local",
            "reasoning": "reasoning_local",
            "coding": "claude_code",
            "research": "web_research",
            "vision": "multimodal_vision",
            "image_generation": "image_generation",
            "confirmation": "human_confirm",
        },
        "engines": base_engines,
    }
    path = tmp_path / "model_router.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_missing_required_command_marks_engine_unavailable(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "claude_code": _engine(
                "claude_code",
                fallback="code_agent",
                availability={
                    "required_commands": ["definitely-not-a-real-modelrouter-command"],
                },
            )
        },
    )
    config = load_router_config(path)

    result = validate_engine_availability(config.engines["claude_code"])

    assert result.available is False
    assert any("missing command" in reason for reason in result.reasons)


def test_missing_required_env_marks_engine_unavailable_without_value_leak(
    tmp_path,
    monkeypatch,
):
    env_name = "HERMES_ROUTER_TEST_SECRET"
    monkeypatch.delenv(env_name, raising=False)
    path = _config_path(
        tmp_path,
        {
            "claude_code": _engine(
                "claude_code",
                fallback="code_agent",
                availability={"required_env": [env_name]},
            )
        },
    )
    config = load_router_config(path)

    result = validate_engine_availability(config.engines["claude_code"])

    assert result.available is False
    assert env_name in " ".join(result.reasons)
    assert "SECRET=" not in " ".join(result.reasons)


def test_routing_skips_unavailable_target_and_uses_available_fallback(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "claude_code": _engine(
                "claude_code",
                fallback="code_agent",
                availability={"status": "unavailable"},
            )
        },
    )

    decision = route_prompt("Fix the repo and run tests.", config_path=path)

    assert decision.selected_engine == "code_agent"
    assert decision.fallback_engine == "code_agent"
    assert decision.availability_valid is True
    assert any("claude_code unavailable" in reason for reason in decision.reasons)
    assert any("unavailable" in reason for reason in decision.availability_reasons)


def test_routing_fails_closed_when_no_available_fallback_exists(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "claude_code": _engine(
                "claude_code",
                fallback="code_agent",
                availability={"status": "unavailable"},
            ),
            "code_agent": _engine(
                "code_agent",
                fallback=None,
                availability={"status": "unavailable"},
            ),
        },
    )

    decision = route_prompt("Fix the repo and run tests.", config_path=path)

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert decision.availability_valid is False
    assert any("unavailable" in reason for reason in decision.availability_reasons)


def test_router_availability_report_serializes_engine_statuses(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "claude_code": _engine(
                "claude_code",
                fallback="code_agent",
                availability={"status": "unavailable"},
            )
        },
    )
    config = load_router_config(path)

    report = validate_router_availability(config)
    payload = report.to_dict()

    assert payload["all_available"] is False
    assert payload["engines"]["claude_code"]["available"] is False
    assert payload["engines"]["code_agent"]["available"] is True


def test_empty_availability_report_is_not_available():
    report = RouterAvailabilityReport(engines={})

    assert report.all_available is False
    assert report.to_dict()["all_available"] is False
