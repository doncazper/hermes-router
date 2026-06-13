"""Configuration loading for the model router engine catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hermes.plugins.model_router.models import ModelEngine, RouterConfig, ScoringConfig

REQUIRED_ENGINE_CATEGORIES = (
    "intent_router",
    "fast_local",
    "balanced_local",
    "reasoning_local",
    "code_agent",
    "web_research",
    "multimodal_vision",
    "image_generation",
    "human_confirm",
)

REQUIRED_ROUTING_TARGETS = (
    "simple",
    "balanced",
    "reasoning",
    "coding",
    "research",
    "vision",
    "image_generation",
    "confirmation",
)

DEFAULT_ROUTING_TARGETS = {
    "simple": "fast_local",
    "balanced": "balanced_local",
    "reasoning": "reasoning_local",
    "coding": "code_agent",
    "research": "web_research",
    "vision": "multimodal_vision",
    "image_generation": "image_generation",
    "confirmation": "human_confirm",
}


class RouterConfigError(ValueError):
    """Raised when the model router catalog cannot be trusted."""


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "model_router.yaml"


def load_router_config(config_path: str | Path | None = None) -> RouterConfig:
    path = Path(config_path) if config_path is not None else default_config_path()
    path = path.expanduser()
    if not path.exists():
        raise RouterConfigError(f"model router config missing: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RouterConfigError(f"model router config invalid YAML: {exc}") from exc
    except OSError as exc:
        raise RouterConfigError(f"model router config unreadable: {exc}") from exc

    if not isinstance(data, dict):
        raise RouterConfigError("model router config must be a mapping")

    engines_data = data.get("engines")
    if not isinstance(engines_data, dict):
        raise RouterConfigError("model router config requires an engines mapping")

    missing = [
        category
        for category in REQUIRED_ENGINE_CATEGORIES
        if category not in engines_data
    ]
    if missing:
        raise RouterConfigError(
            "model router config missing required engines: " + ", ".join(missing)
        )

    engines: dict[str, ModelEngine] = {}
    for name, engine_data in engines_data.items():
        if not isinstance(name, str) or not name.strip():
            raise RouterConfigError("engine names must be non-empty strings")
        try:
            engines[name] = ModelEngine.from_dict(name, _require_mapping(engine_data))
        except ValueError as exc:
            raise RouterConfigError(str(exc)) from exc

    for engine in engines.values():
        if engine.fallback is not None and engine.fallback not in engines:
            raise RouterConfigError(
                f"engine {engine.name!r} fallback {engine.fallback!r} is not defined"
            )

    routing_targets = _load_routing_targets(data, engines)
    try:
        scoring = ScoringConfig.from_dict(data.get("scoring"))
    except ValueError as exc:
        raise RouterConfigError(f"model router scoring config invalid: {exc}") from exc

    return RouterConfig(
        engines=engines,
        routing_targets=routing_targets,
        source_path=str(path),
        scoring=scoring,
    )


def _load_routing_targets(
    data: dict[str, Any],
    engines: dict[str, ModelEngine],
) -> dict[str, str]:
    raw_targets = data.get("routing_targets", DEFAULT_ROUTING_TARGETS)
    if not isinstance(raw_targets, dict):
        raise RouterConfigError("model router config routing_targets must be a mapping")

    targets = {**DEFAULT_ROUTING_TARGETS}
    for key, value in raw_targets.items():
        if not isinstance(key, str) or not key.strip():
            raise RouterConfigError("routing target names must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise RouterConfigError(f"routing target {key!r} must name an engine")
        targets[key] = value

    missing = [key for key in REQUIRED_ROUTING_TARGETS if key not in targets]
    if missing:
        raise RouterConfigError(
            "model router config missing routing targets: " + ", ".join(missing)
        )

    undefined = [
        f"{target} -> {engine}"
        for target, engine in targets.items()
        if engine not in engines
    ]
    if undefined:
        raise RouterConfigError(
            "routing targets reference undefined engines: " + ", ".join(undefined)
        )

    return targets


def _require_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("engine entry must be a mapping")
    return value
