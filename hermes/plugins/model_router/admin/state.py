"""Shared admin state entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hermes.plugins.model_router.admin.supervisor import ProxyProcessSupervisor
from hermes.plugins.model_router.pricing_catalog import DEFAULT_PRICING_CATALOG_NAME


def settings_paths(config_dir: str | Path) -> dict[str, Path]:
    """Return canonical settings/admin paths for a config directory."""

    base = Path(config_dir).expanduser()
    return {
        "config_dir": base,
        "model_router_config": base / "model_router.yaml",
        "proxy_config": base / "routing_proxy.yaml",
        "events": base / "logs" / "routing-events.jsonl",
        "feedback": base / "routing-feedback.jsonl",
        "settings_proxy_log": base / "logs" / "settings-proxy.log",
        "models": base / "models",
        "benchmarks": base / "benchmarks.json",
        "workflow_benchmarks": base / "workflow-benchmarks.json",
        "pricing": base / DEFAULT_PRICING_CATALOG_NAME,
    }


def build_admin_state(
    paths: Mapping[str, Path],
    supervisor: ProxyProcessSupervisor | None = None,
) -> dict[str, Any]:
    """Build shared admin state for web UI, future TUI, installer, and API.

    The first extraction keeps rendering helpers in ``settings_ui`` while moving
    the stable public control-plane entry point here. Later milestones can move
    more private helpers behind this boundary without changing callers.
    """

    from hermes.plugins.model_router import settings_ui as legacy_settings_ui

    return legacy_settings_ui._build_settings_state_impl(paths, supervisor)
