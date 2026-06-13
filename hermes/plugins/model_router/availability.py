"""Safe availability validation for configured model-router engines."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from hermes.plugins.model_router.models import (
    EngineAvailabilityResult,
    ModelEngine,
    RouterAvailabilityReport,
    RouterConfig,
)


def validate_engine_availability(engine: ModelEngine) -> EngineAvailabilityResult:
    spec = engine.availability
    reasons: list[str] = []

    if spec.status == "unavailable":
        return EngineAvailabilityResult(
            engine=engine.name,
            available=False,
            reasons=("marked unavailable in config",),
        )

    for env_name in spec.required_env:
        if not os.environ.get(env_name):
            reasons.append(f"missing env var {env_name}")

    for command in spec.required_commands:
        if shutil.which(command) is None:
            reasons.append(f"missing command {command}")

    for raw_path in spec.required_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            reasons.append(f"missing path {raw_path}")

    if reasons:
        return EngineAvailabilityResult(
            engine=engine.name,
            available=False,
            reasons=tuple(reasons),
        )

    if spec.status == "available":
        reasons.append("marked available in config")
    elif spec.required_env or spec.required_commands or spec.required_paths:
        reasons.append("availability checks passed")
    else:
        reasons.append("no availability requirements declared")

    return EngineAvailabilityResult(
        engine=engine.name,
        available=True,
        reasons=tuple(reasons),
    )


def validate_router_availability(config: RouterConfig) -> RouterAvailabilityReport:
    return RouterAvailabilityReport(
        engines={
            name: validate_engine_availability(engine)
            for name, engine in config.engines.items()
            if engine.enabled
        }
    )
