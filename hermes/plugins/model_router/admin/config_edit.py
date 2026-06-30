"""Shared proxy config edit entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shlex
import tempfile
from typing import Any

import yaml

from hermes.plugins.model_router.product import PRESETS, initialize_product_config
from hermes.plugins.model_router.proxy_config import (
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)


def save_proxy_config_patch(
    config_path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply an explicit admin config patch and return redacted config state."""

    path = Path(config_path).expanduser()
    if _payload_bool(payload, "apply_preset", default=False):
        preset = str(payload.get("preset", "")).strip()
        if preset not in PRESETS:
            raise ValueError("preset must be one of: " + ", ".join(PRESETS))
        result = initialize_product_config(
            preset=preset,
            config_dir=path.parent,
            force=True,
            interactive=False,
        )
        config = load_proxy_config(path)
        return {
            "config_path": str(path),
            "preset": preset,
            "init": result.to_dict(),
            "proxy": _redacted_proxy_state(config),
            "backend_policy": _backend_policy_state(config),
            "backends": _redacted_backend_states(config),
            "observability": _observability_state(config),
        }
    if not path.exists():
        raise ValueError(f"routing proxy config missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ProxyConfigError("routing proxy config must be a mapping")
    _patch_proxy_config_data(data, payload)
    text = yaml.safe_dump(data, sort_keys=False)
    _validate_proxy_config_text(path, text)
    path.write_text(text, encoding="utf-8")
    config = load_proxy_config(path)
    return {
        "config_path": str(path),
        "proxy": _redacted_proxy_state(config),
        "backend_policy": _backend_policy_state(config),
        "backends": _redacted_backend_states(config),
        "observability": _observability_state(config),
    }


def _redacted_proxy_state(config: RoutingProxyConfig | None) -> dict[str, Any]:
    if config is None:
        return {
            "routing_mode": "decision",
            "decision_layer_enabled": True,
            "default_backend": None,
            "default_model": None,
            "respect_client_model": False,
            "unknown_model_behavior": "fallback_to_default",
            "safety_gate_mode": "decision_only",
        }
    proxy = config.proxy
    routing_mode = str(getattr(proxy, "routing_mode", "decision") or "decision")
    return {
        "host": proxy.host,
        "port": proxy.port,
        "routing_profile": proxy.routing_profile,
        "routing_mode": routing_mode,
        "decision_layer_enabled": routing_mode == "decision",
        "default_backend": getattr(proxy, "default_backend", None),
        "default_model": getattr(proxy, "default_model", None),
        "respect_client_model": bool(
            getattr(proxy, "respect_client_model", routing_mode != "decision")
        ),
        "unknown_model_behavior": str(
            getattr(proxy, "unknown_model_behavior", "fallback_to_default")
            or "fallback_to_default"
        ),
        "safety_gate_mode": str(
            getattr(proxy, "safety_gate_mode", "decision_only") or "decision_only"
        ),
        "endpoint": f"http://{proxy.host}:{proxy.port}/v1",
        "model_ids": list(proxy.model_ids),
        "api_key_configured": bool(proxy.api_key or proxy.api_key_env),
        "api_key_env": proxy.api_key_env,
    }


def _backend_policy_state(config: RoutingProxyConfig | None) -> dict[str, Any]:
    if config is None:
        return {"backend_allowlist": [], "backend_denylist": []}
    return config.backend_policy.to_dict()


def _redacted_backend_states(config: RoutingProxyConfig | None) -> list[dict[str, Any]]:
    if config is None:
        return []
    rows: list[dict[str, Any]] = []
    for backend in config.backends.values():
        runtime = backend.runtime
        rows.append(
            {
                "name": backend.name,
                "base_url": backend.base_url,
                "model": backend.model,
                "timeout_seconds": backend.timeout_seconds,
                "strip_tools": backend.strip_tools,
                "api_key_configured": bool(backend.api_key or backend.api_key_env),
                "api_key_env": backend.api_key_env,
                "runtime": {
                    "enabled": runtime.enabled,
                    "kind": runtime.kind,
                    "command": list(runtime.command),
                    "readiness_url": runtime.readiness_url,
                    "readiness_timeout_seconds": runtime.readiness_timeout_seconds,
                    "idle_timeout_seconds": runtime.idle_timeout_seconds,
                    "shutdown_timeout_seconds": runtime.shutdown_timeout_seconds,
                    "log_path": runtime.log_path,
                    "status": "managed-by-proxy" if runtime.enabled else "unmanaged",
                },
            }
        )
    return rows


def _observability_state(config: RoutingProxyConfig | None) -> dict[str, Any]:
    if config is None:
        return {"enabled": False, "log_path": "", "prompt_capture": "redacted_preview"}
    return {
        "enabled": config.observability.enabled,
        "log_path": config.observability.log_path,
        "prompt_capture": config.observability.prompt_capture,
        "max_bytes": config.observability.max_bytes,
        "backups": config.observability.backups,
    }


def _patch_proxy_config_data(data: dict[str, Any], payload: Mapping[str, Any]) -> None:
    proxy_patch = _mapping(payload.get("proxy"))
    proxy = data.setdefault("proxy", {})
    if proxy_patch:
        _patch_string(proxy, proxy_patch, "host")
        _patch_int(proxy, proxy_patch, "port")
        _patch_string(proxy, proxy_patch, "routing_profile")
        if "model_ids" in proxy_patch:
            proxy["model_ids"] = _string_list(proxy_patch["model_ids"])

    obs_patch = _mapping(payload.get("observability"))
    observability = data.setdefault("observability", {})
    if obs_patch:
        _patch_bool(observability, obs_patch, "enabled")
        _patch_string(observability, obs_patch, "log_path")
        _patch_string(observability, obs_patch, "prompt_capture")

    backend_policy_patch = _mapping(payload.get("backend_policy"))
    if backend_policy_patch:
        backend_policy = data.setdefault("backend_policy", {})
        backend_policy["version"] = 1
        if "backend_allowlist" in backend_policy_patch:
            backend_policy["backend_allowlist"] = _string_list(
                backend_policy_patch["backend_allowlist"]
            )
        if "backend_denylist" in backend_policy_patch:
            backend_policy["backend_denylist"] = _string_list(
                backend_policy_patch["backend_denylist"]
            )

    backend_patches = _mapping(payload.get("backends"))
    backends = data.setdefault("backends", {})
    for backend_name, raw_patch in backend_patches.items():
        if backend_name not in backends or not isinstance(backends[backend_name], dict):
            continue
        patch = _mapping(raw_patch)
        backend = backends[backend_name]
        _patch_string(backend, patch, "model")
        _patch_string(backend, patch, "base_url")
        _patch_float(backend, patch, "timeout_seconds")
        _patch_bool(backend, patch, "strip_tools")
        _patch_string(backend, patch, "api_key_env")
        runtime_patch = _mapping(patch.get("runtime"))
        if runtime_patch:
            runtime = backend.setdefault("runtime", {})
            _patch_bool(runtime, runtime_patch, "enabled")
            _patch_string(runtime, runtime_patch, "kind")
            if "command" in runtime_patch:
                runtime["command"] = _argv_list(runtime_patch["command"])
            _patch_string(runtime, runtime_patch, "readiness_url")
            _patch_float(runtime, runtime_patch, "readiness_timeout_seconds")
            _patch_float(runtime, runtime_patch, "idle_timeout_seconds")
            _patch_float(runtime, runtime_patch, "shutdown_timeout_seconds")
            _patch_string(runtime, runtime_patch, "log_path")


def _validate_proxy_config_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        load_proxy_config(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _patch_string(target: dict[str, Any], patch: Mapping[str, Any], key: str) -> None:
    if key not in patch:
        return
    value = patch[key]
    if value is None:
        return
    text = str(value).strip()
    if text:
        target[key] = text


def _patch_int(target: dict[str, Any], patch: Mapping[str, Any], key: str) -> None:
    if key in patch and str(patch[key]).strip():
        target[key] = int(patch[key])


def _patch_float(target: dict[str, Any], patch: Mapping[str, Any], key: str) -> None:
    if key in patch and str(patch[key]).strip():
        target[key] = float(patch[key])


def _patch_bool(target: dict[str, Any], patch: Mapping[str, Any], key: str) -> None:
    if key in patch:
        target[key] = _payload_bool(patch, key, default=False)


def _payload_bool(payload: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _argv_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return shlex.split(str(value))
