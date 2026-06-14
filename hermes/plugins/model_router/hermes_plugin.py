"""Hermes Agent plugin registration for Hermes Router."""

from __future__ import annotations

import argparse
import json
from typing import Any

from hermes.plugins.model_router.cli import configure_parser
from hermes.plugins.model_router.config import RouterConfigError
from hermes.plugins.model_router.policy import ModelRouter

PLUGIN_NAME = "hermes-router"
SLASH_USAGE = "Usage: /router status | /router decide <prompt>"


def register(ctx: Any) -> None:
    """Register Hermes CLI and slash-command diagnostics."""
    ctx.register_cli_command(
        name="router",
        help="Inspect Hermes Router decisions",
        setup_fn=_setup_cli,
        handler_fn=_handle_cli,
    )
    ctx.register_command(
        "router",
        handler=_handle_slash_command,
        description="Inspect Hermes Router status or route a prompt",
        args_hint="status | decide <prompt>",
    )


def _setup_cli(subparser: argparse.ArgumentParser) -> None:
    configure_parser(subparser)


def _handle_cli(args: argparse.Namespace) -> int:
    handler = getattr(args, "func", None)
    if handler is None:
        print("Usage: hermes router <decide|dispatch-plan|validate-config|setup>")
        return 2
    return int(handler(args) or 0)


def _handle_slash_command(raw_args: str) -> str:
    try:
        raw = (raw_args or "").strip()
        if not raw or raw in {"help", "-h", "--help"}:
            return SLASH_USAGE

        command, _, rest = raw.partition(" ")
        command = command.lower()
        rest = rest.strip()
        if command == "status":
            return _json_response(_status_payload())
        if command in {"decide", "route"}:
            if not rest:
                return _json_response(
                    {"ok": False, "error": "prompt is required", "usage": SLASH_USAGE}
                )
            return _json_response(_route_fast_payload(rest))
        return _json_response(
            {
                "ok": False,
                "error": f"unknown router command: {command}",
                "usage": SLASH_USAGE,
            }
        )
    except Exception as exc:
        return _json_response({"ok": False, "error": str(exc)})


def _status_payload() -> dict[str, Any]:
    try:
        router = ModelRouter.from_config(validate_availability=False)
    except RouterConfigError as exc:
        return {
            "ok": False,
            "plugin": PLUGIN_NAME,
            "api": "route_fast",
            "config_valid": False,
            "error": str(exc),
        }
    return {
        "ok": True,
        "plugin": PLUGIN_NAME,
        "api": "route_fast",
        "config_valid": True,
        "config_source": router.config.source_path,
        "engines": len(router.config.engines),
        "routing_targets": router.config.routing_targets,
    }


def _route_fast_payload(prompt: str) -> dict[str, Any]:
    try:
        router = ModelRouter.from_config(validate_availability=False)
    except RouterConfigError as exc:
        return {
            "ok": False,
            "plugin": PLUGIN_NAME,
            "api": "route_fast",
            "config_valid": False,
            "error": str(exc),
        }
    return {
        "ok": True,
        "plugin": PLUGIN_NAME,
        "api": "route_fast",
        "config_valid": True,
        "config_source": router.config.source_path,
        "selected_engine": router.route_fast(prompt),
    }


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)
