"""Configuration loading for the model router engine catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hermes.plugins.model_router.models import ModelEngine, RouterConfig

REQUIRED_ENGINE_CATEGORIES = (
    "fast_local",
    "balanced_local",
    "reasoning_local",
    "codex",
    "web_research",
    "human_confirm",
)


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

    return RouterConfig(engines=engines, source_path=str(path))


def _require_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("engine entry must be a mapping")
    return value
