"""Guided runtime install/connect plans for operator surfaces.

This module is intentionally CLI/admin oriented. It must not be imported by
route_fast, route, or normal proxy forwarding paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyConfigError,
    load_proxy_config,
)
from hermes.plugins.model_router.model_registry import build_model_registry
from hermes.plugins.model_router.runtime_adapters import runtime_state_for_backend


SUPPORTED_CONNECT_RUNTIMES = ("lmstudio", "ollama", "llamacpp")


class RuntimeInstallError(ValueError):
    """Raised when a guided runtime operation cannot be planned safely."""


@dataclass(frozen=True)
class RuntimeConnectRequest:
    runtime_id: str
    config_path: Path
    backend: str = "fast"
    endpoint: str | None = None
    model: str | None = None
    write: bool = False
    confirmed: bool = False
    timeout_seconds: float = 0.25


def runtime_status_report(
    config_path: str | Path,
    *,
    timeout_seconds: float = 0.25,
) -> dict[str, Any]:
    """Return runtime adapter status for all configured backends."""

    config_path = Path(config_path).expanduser()
    config = load_proxy_config(config_path)
    runtimes: list[dict[str, Any]] = []
    runtime_states: dict[str, dict[str, Any]] = {}
    for backend in config.backends.values():
        state = runtime_state_for_backend(
            backend,
            timeout_seconds=timeout_seconds,
        )
        runtime_states[backend.name] = state
        runtimes.append(_status_row(backend, state))
    ready = sum(1 for row in runtimes if row.get("health_status") == "ready")
    registry = build_model_registry(
        proxy_config=config,
        runtime_models=runtime_states,
    )
    imported_models = [
        model.to_dict()
        for model in registry.models
        if "runtime_import" in model.source.split("+")
    ]
    return {
        "ok": True,
        "config_path": str(config_path),
        "runtime_count": len(runtimes),
        "ready_count": ready,
        "imported_model_count": len(imported_models),
        "imported_models": imported_models,
        "runtimes": runtimes,
        "notes": [
            "Runtime status is advisory operator state; routing uses configured backend policy.",
            "No install, download, config write, or lifecycle action was run.",
        ],
    }


def runtime_doctor_report(
    config_path: str | Path,
    *,
    timeout_seconds: float = 0.25,
) -> dict[str, Any]:
    """Return runtime status plus actionable remediation guidance."""

    status = runtime_status_report(config_path, timeout_seconds=timeout_seconds)
    guidance: list[dict[str, Any]] = []
    for row in status["runtimes"]:
        runtime_id = str(row.get("runtime_id") or row.get("provider") or "")
        action = _next_action_for_status(row)
        if action == "view details":
            continue
        guidance.append(
            {
                "backend": row.get("backend"),
                "runtime_id": runtime_id,
                "next_action": action,
                "message": _doctor_message(runtime_id, row),
                "install_hint": row.get("install_hint"),
                "official_sources": _official_sources(runtime_id),
            }
        )
    return {
        **status,
        "ok": True,
        "doctor_ok": not guidance,
        "guidance": guidance,
    }


def build_runtime_connect_plan(request: RuntimeConnectRequest) -> dict[str, Any]:
    """Build and optionally apply a runtime connect/config plan."""

    runtime_id = _normalize_runtime_id(request.runtime_id)
    endpoint = _endpoint_for_runtime(runtime_id, request.endpoint)
    backend_name = request.backend.strip() or "fast"
    model = (request.model or "").strip()
    config_path = request.config_path.expanduser()
    config_data = _read_proxy_yaml(config_path)
    backends = config_data.get("backends")
    if not isinstance(backends, dict):
        raise RuntimeInstallError("routing_proxy.yaml must contain a backends mapping")
    existing_backend = backends.get(backend_name)
    backend_exists = isinstance(existing_backend, dict)
    existing_model = (
        str(existing_backend.get("model") or "")
        if isinstance(existing_backend, Mapping)
        else ""
    )
    effective_model = model or existing_model or _placeholder_model(runtime_id)
    patch = _backend_patch(runtime_id, endpoint, effective_model)
    health_backend = ProxyBackendConfig(
        name=backend_name,
        base_url=endpoint,
        model=effective_model,
    )
    health = runtime_state_for_backend(
        health_backend,
        timeout_seconds=request.timeout_seconds,
    )
    actions = _connect_actions(
        runtime_id,
        endpoint,
        backend_name,
        patch,
        config_path=config_path,
    )
    warnings = _connect_warnings(
        runtime_id=runtime_id,
        endpoint=endpoint,
        backend_exists=backend_exists,
        model=model,
        write=request.write,
        confirmed=request.confirmed,
    )
    plan: dict[str, Any] = {
        "ok": True,
        "runtime_id": runtime_id,
        "config_path": str(config_path),
        "backend": backend_name,
        "endpoint": endpoint,
        "model": effective_model,
        "backend_exists": backend_exists,
        "dry_run": not (request.write and request.confirmed),
        "write_requested": request.write,
        "confirmed": request.confirmed,
        "config_written": False,
        "restart_recommended": False,
        "health": _safe_health_summary(health),
        "actions": actions,
        "config_patch": patch,
        "config_diff": _config_patch_preview(backend_name, patch),
        "guidance": _connect_guidance(runtime_id, endpoint),
        "official_sources": _official_sources(runtime_id),
        "warnings": warnings,
        "notes": [
            "Preview only by default; pass --write --yes to update routing_proxy.yaml.",
            "No models were downloaded and no runtime was installed or started.",
            "Runtime connection does not affect route_fast, route, or proxy forwarding hot paths.",
        ],
    }
    if request.write and not request.confirmed:
        plan["ok"] = False
        plan["error"] = "Runtime connect config write requires --yes."
        return plan
    if request.write and request.confirmed:
        if not backend_exists:
            plan["ok"] = False
            plan["error"] = (
                f"Backend {backend_name!r} is not configured; create or choose "
                "a backend before applying this MVP connect plan."
            )
            return plan
        if not model and not existing_model:
            plan["ok"] = False
            plan["error"] = "Config write requires --model or an existing backend model."
            return plan
        backup_path = _write_backend_patch(config_path, config_data, backend_name, patch)
        plan["config_written"] = True
        plan["restart_recommended"] = True
        plan["backup_path"] = str(backup_path)
        plan["dry_run"] = False
    return plan


def _status_row(
    backend: ProxyBackendConfig,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    health = state.get("health") if isinstance(state.get("health"), Mapping) else {}
    capabilities = (
        state.get("capabilities") if isinstance(state.get("capabilities"), Mapping) else {}
    )
    return {
        "backend": backend.name,
        "configured_model": backend.model,
        "runtime_id": state.get("runtime_id") or state.get("provider"),
        "runtime_kind": state.get("runtime_kind"),
        "runtime_mode": state.get("runtime_mode"),
        "endpoint": state.get("endpoint") or state.get("endpoint_url"),
        "detected": state.get("detected"),
        "version": state.get("version"),
        "health_status": state.get("health_status") or health.get("status"),
        "health_ok": health.get("ok") is True,
        "health_detail": health.get("detail"),
        "install_hint": state.get("install_hint"),
        "missing_dependency": state.get("missing_dependency"),
        "last_checked_at": state.get("last_checked_at"),
        "capabilities": {
            key: value
            for key, value in capabilities.items()
            if key
            in {
                "detect_runtime",
                "health",
                "discover_models",
                "list_loaded_models",
                "start_server",
                "stop_server",
                "load_model",
                "unload_model",
            }
        },
    }


def _normalize_runtime_id(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "lm-studio": "lmstudio",
        "llama.cpp": "llamacpp",
        "llama-cpp": "llamacpp",
        "llama-server": "llamacpp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_CONNECT_RUNTIMES:
        raise RuntimeInstallError(
            "runtime must be one of: " + ", ".join(SUPPORTED_CONNECT_RUNTIMES)
        )
    return normalized


def _endpoint_for_runtime(runtime_id: str, endpoint: str | None) -> str:
    value = (endpoint or "").strip()
    if not value:
        value = {
            "lmstudio": "http://127.0.0.1:1234/v1",
            "ollama": "http://127.0.0.1:11434/v1",
            "llamacpp": "http://127.0.0.1:8080/v1",
        }[runtime_id]
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeInstallError("--endpoint must be an http(s) URL")
    return value.rstrip("/")


def _placeholder_model(runtime_id: str) -> str:
    return {
        "lmstudio": "<model-id-from-lm-studio>",
        "ollama": "<ollama-model-tag>",
        "llamacpp": "<llama-cpp-model-id>",
    }[runtime_id]


def _backend_patch(runtime_id: str, endpoint: str, model: str) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "base_url": endpoint,
        "model": model,
        "timeout_seconds": 300,
    }
    if runtime_id in {"ollama", "llamacpp"}:
        patch["strip_tools"] = runtime_id == "ollama"
    return patch


def _read_proxy_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise RuntimeInstallError(f"routing proxy config missing: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeInstallError(f"routing proxy config unreadable: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeInstallError(f"routing proxy config invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeInstallError("routing proxy config must be a mapping")
    return data


def _write_backend_patch(
    config_path: Path,
    config_data: dict[str, Any],
    backend_name: str,
    patch: Mapping[str, Any],
) -> Path:
    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    backends = config_data.setdefault("backends", {})
    if not isinstance(backends, dict) or not isinstance(backends.get(backend_name), dict):
        raise RuntimeInstallError(f"backend {backend_name!r} is not configured")
    updated = dict(backends[backend_name])
    updated.update(patch)
    backends[backend_name] = updated
    text = yaml.safe_dump(config_data, sort_keys=False)
    config_path.write_text(text, encoding="utf-8")
    try:
        load_proxy_config(config_path)
    except ProxyConfigError as exc:
        config_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
        raise RuntimeInstallError(f"updated config failed validation: {exc}") from exc
    return backup_path


def _connect_actions(
    runtime_id: str,
    endpoint: str,
    backend: str,
    patch: Mapping[str, Any],
    *,
    config_path: Path,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if runtime_id == "lmstudio":
        actions.append(
            _external_link_action(
                "runtime.lmstudio.open_download",
                "Open LM Studio download page",
                "https://lmstudio.ai/download",
                "Install/open LM Studio, load a model, and enable the local server.",
            )
        )
    elif runtime_id == "ollama":
        actions.extend(
            [
                _external_link_action(
                    "runtime.ollama.open_download",
                    "Open Ollama download page",
                    "https://ollama.com/download",
                    "Install Ollama with the official app or installer.",
                ),
                _shell_action(
                    "runtime.ollama.serve",
                    "Start Ollama server",
                    ("ollama", "serve"),
                    "Run only if Ollama is installed and not already running.",
                ),
                _shell_action(
                    "runtime.ollama.pull",
                    "Pull selected model separately",
                    ("ollama", "pull", str(patch["model"])),
                    "Model pulls are separate from connect and require confirmation.",
                ),
            ]
        )
    else:
        actions.extend(
            [
                _external_link_action(
                    "runtime.llamacpp.open_docs",
                    "Open llama.cpp server docs",
                    "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md",
                    "Install or build llama.cpp using the official docs.",
                ),
                _shell_action(
                    "runtime.llamacpp.example_start",
                    "Example llama-server command",
                    (
                        "llama-server",
                        "-m",
                        "/path/to/model.gguf",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        _port_from_endpoint(endpoint),
                    ),
                    "Replace the model path before running.",
                ),
            ]
        )
    actions.append(
        {
            "id": f"runtime.{runtime_id}.write_config",
            "kind": "config_patch",
            "label": f"Connect backend {backend}",
            "preview": _config_patch_preview(backend, patch),
            "mutates": True,
            "requires_confirmation": True,
            "command": [
                "model-router",
                "runtimes",
                "connect",
                runtime_id,
                "--config",
                str(config_path),
                "--backend",
                backend,
                "--model",
                str(patch["model"]),
                "--write",
                "--yes",
            ],
        }
    )
    return actions


def _external_link_action(
    action_id: str,
    label: str,
    url: str,
    description: str,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "kind": "external_link",
        "label": label,
        "url": url,
        "description": description,
        "mutates": False,
        "requires_confirmation": False,
    }


def _shell_action(
    action_id: str,
    label: str,
    command: tuple[str, ...],
    description: str,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "kind": "shell_command",
        "label": label,
        "command": list(command),
        "preview": " ".join(command),
        "description": description,
        "mutates": True,
        "requires_confirmation": True,
    }


def _config_patch_preview(backend: str, patch: Mapping[str, Any]) -> str:
    return yaml.safe_dump({"backends": {backend: dict(patch)}}, sort_keys=False)


def _connect_guidance(runtime_id: str, endpoint: str) -> list[str]:
    if runtime_id == "lmstudio":
        return [
            "Open LM Studio, download/load a model, and enable the local server.",
            f"ModelRouter will connect to {endpoint} after you confirm a config write.",
            "Use the exact model id shown by LM Studio's /v1/models endpoint.",
        ]
    if runtime_id == "ollama":
        return [
            "Install or start Ollama, then verify port 11434 is reachable.",
            "Run model pulls separately, for example `ollama pull qwen3:4b`.",
            f"ModelRouter will connect to {endpoint} after you confirm a config write.",
        ]
    return [
        "Install or build llama.cpp so `llama-server` is available.",
        "Start llama-server with an explicit GGUF model path.",
        f"ModelRouter will connect to {endpoint} after you confirm a config write.",
    ]


def _connect_warnings(
    *,
    runtime_id: str,
    endpoint: str,
    backend_exists: bool,
    model: str,
    write: bool,
    confirmed: bool,
) -> list[str]:
    warnings: list[str] = []
    parsed = urlparse(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        warnings.append("Endpoint is not localhost; review network exposure and auth.")
    if not backend_exists:
        warnings.append("Selected backend does not exist; MVP will not create it.")
    if write and not confirmed:
        warnings.append("Config write requested without --yes; no file will be changed.")
    if write and confirmed and not model:
        warnings.append("No --model supplied; existing backend model will be reused if present.")
    if runtime_id == "ollama":
        warnings.append("Ollama model pulls are not part of connect and must be run separately.")
    if runtime_id == "llamacpp":
        warnings.append("llama.cpp model downloads/builds are not part of connect.")
    return warnings


def _next_action_for_status(row: Mapping[str, Any]) -> str:
    if row.get("install_hint") or row.get("missing_dependency"):
        return "install guide"
    if row.get("detected") is False:
        return "configure"
    health_status = str(row.get("health_status") or "").lower()
    if health_status in {"unreachable", "error"}:
        return "connect"
    if health_status in {"unknown", "unsupported", ""}:
        return "configure"
    return "view details"


def _doctor_message(runtime_id: str, row: Mapping[str, Any]) -> str:
    hint = row.get("install_hint")
    if hint:
        return str(hint)
    if runtime_id == "lmstudio":
        return "Start LM Studio's local server and verify the configured model id."
    if runtime_id == "ollama":
        return "Start Ollama with `ollama serve` or install it from the official download page."
    if runtime_id == "llamacpp":
        return "Start the configured llama.cpp server or update the backend endpoint."
    return "Start or configure the runtime endpoint, then run runtimes status again."


def _official_sources(runtime_id: str) -> list[str]:
    return {
        "lmstudio": [
            "https://lmstudio.ai/download",
            "https://lmstudio.ai/docs/app",
        ],
        "ollama": [
            "https://ollama.com/download",
            "https://github.com/ollama/ollama",
        ],
        "llamacpp": [
            "https://github.com/ggml-org/llama.cpp",
            "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md",
        ],
    }.get(runtime_id, [])


def _safe_health_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    health = state.get("health") if isinstance(state.get("health"), Mapping) else {}
    return {
        "runtime_id": state.get("runtime_id") or state.get("provider"),
        "detected": state.get("detected"),
        "endpoint": state.get("endpoint") or state.get("endpoint_url"),
        "health_status": state.get("health_status") or health.get("status"),
        "health_ok": health.get("ok") is True,
        "health_detail": health.get("detail"),
        "install_hint": state.get("install_hint"),
        "missing_dependency": state.get("missing_dependency"),
        "last_checked_at": state.get("last_checked_at"),
    }


def _port_from_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.port:
        return str(parsed.port)
    return "443" if parsed.scheme == "https" else "80"
