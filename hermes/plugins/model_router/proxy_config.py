"""Configuration for the optional OpenAI-compatible routing proxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import os
from pathlib import Path
from typing import Any

import yaml

from hermes.plugins.model_router.routing_log import (
    DEFAULT_LOG_PATH,
    PROMPT_CAPTURE_MODES,
    PROMPT_CAPTURE_REDACTED,
)


DEFAULT_PROXY_MODEL_ID = "model-router"
DEFAULT_PROXY_CONFIG_PACKAGE = "hermes.plugins.model_router.data"
DEFAULT_PROXY_CONFIG_NAME = "routing_proxy.example.yaml"
DEFAULT_PROXY_CONFIG_SOURCE = (
    f"resource://{DEFAULT_PROXY_CONFIG_PACKAGE}/{DEFAULT_PROXY_CONFIG_NAME}"
)


class ProxyConfigError(ValueError):
    """Raised when the routing proxy config cannot be trusted."""


@dataclass(frozen=True)
class ProxyServerConfig:
    host: str = "127.0.0.1"
    port: int = 8082
    api_key: str | None = None
    api_key_env: str | None = None
    model_ids: tuple[str, ...] = (DEFAULT_PROXY_MODEL_ID,)

    @property
    def resolved_api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return self.api_key


@dataclass(frozen=True)
class ProxyBackendConfig:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 300.0
    strip_tools: bool = False

    @property
    def resolved_api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return self.api_key


@dataclass(frozen=True)
class ProxyObservabilityConfig:
    enabled: bool = False
    log_path: str = DEFAULT_LOG_PATH
    prompt_capture: str = PROMPT_CAPTURE_REDACTED
    max_bytes: int = 10_485_760
    backups: int = 5


@dataclass(frozen=True)
class ProxyHealthConfig:
    backend_timeout_seconds: float = 1.0


@dataclass(frozen=True)
class RoutingProxyConfig:
    proxy: ProxyServerConfig
    router_config: str | None
    backends: dict[str, ProxyBackendConfig]
    engine_backends: dict[str, str]
    fallback_backends: dict[str, tuple[str, ...]]
    source_path: str
    observability: ProxyObservabilityConfig = field(
        default_factory=ProxyObservabilityConfig
    )
    health: ProxyHealthConfig = field(default_factory=ProxyHealthConfig)

    def backend_for_engine(self, engine: str) -> ProxyBackendConfig | None:
        backend_name = self.engine_backends.get(engine)
        if backend_name is None:
            return None
        return self.backends.get(backend_name)

    def fallback_chain_for_backend(self, backend_name: str) -> tuple[ProxyBackendConfig, ...]:
        return tuple(
            self.backends[name]
            for name in self.fallback_backends.get(backend_name, ())
            if name in self.backends
        )


def default_proxy_config_resource() -> resources.abc.Traversable:
    return resources.files(DEFAULT_PROXY_CONFIG_PACKAGE).joinpath(
        DEFAULT_PROXY_CONFIG_NAME,
    )


def default_proxy_config_source() -> str:
    return DEFAULT_PROXY_CONFIG_SOURCE


def default_proxy_config_text() -> str:
    return default_proxy_config_resource().read_text(encoding="utf-8")


def load_proxy_config(config_path: str | Path | None = None) -> RoutingProxyConfig:
    source_path: str
    try:
        if config_path is None:
            source_path = default_proxy_config_source()
            data = yaml.safe_load(default_proxy_config_text())
        else:
            path = Path(config_path).expanduser()
            source_path = str(path)
            if not path.exists():
                raise ProxyConfigError(f"routing proxy config missing: {path}")
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProxyConfigError(f"routing proxy config invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ProxyConfigError(f"routing proxy config unreadable: {exc}") from exc

    if not isinstance(data, dict):
        raise ProxyConfigError("routing proxy config must be a mapping")

    proxy = _load_proxy_server(data.get("proxy"))
    observability = _load_observability(data.get("observability"))
    health = _load_health(data.get("health"))
    router_config = _optional_string(data, "router_config")
    backends = _load_backends(data.get("backends"))
    _validate_resolved_env_secrets(proxy, backends)
    engine_backends = _load_engine_backends(data.get("engine_backends"), backends)
    fallback_backends = _load_fallback_backends(data.get("fallback_backends"), backends)
    _validate_no_fallback_cycles(fallback_backends)

    return RoutingProxyConfig(
        proxy=proxy,
        router_config=router_config,
        backends=backends,
        engine_backends=engine_backends,
        fallback_backends=fallback_backends,
        source_path=source_path,
        observability=observability,
        health=health,
    )


def _load_proxy_server(data: Any) -> ProxyServerConfig:
    if data is None:
        return ProxyServerConfig()
    if not isinstance(data, dict):
        raise ProxyConfigError("proxy must be a mapping")
    api_key = _optional_string(data, "api_key")
    api_key_env = _optional_string(data, "api_key_env")
    if api_key and api_key_env:
        raise ProxyConfigError("proxy may define api_key or api_key_env, not both")
    model_ids = _string_tuple(data, "model_ids", default=(DEFAULT_PROXY_MODEL_ID,))
    return ProxyServerConfig(
        host=_string(data, "host", default="127.0.0.1"),
        port=_positive_int(data, "port", default=8082),
        api_key=api_key,
        api_key_env=api_key_env,
        model_ids=model_ids,
    )


def _load_backends(data: Any) -> dict[str, ProxyBackendConfig]:
    if not isinstance(data, dict) or not data:
        raise ProxyConfigError("routing proxy config requires non-empty backends")
    backends: dict[str, ProxyBackendConfig] = {}
    for name, raw_backend in data.items():
        if not isinstance(name, str) or not name.strip():
            raise ProxyConfigError("backend names must be non-empty strings")
        if not isinstance(raw_backend, dict):
            raise ProxyConfigError(f"backend {name!r} must be a mapping")
        api_key = _optional_string(raw_backend, "api_key")
        api_key_env = _optional_string(raw_backend, "api_key_env")
        if api_key and api_key_env:
            raise ProxyConfigError(
                f"backend {name!r} may define api_key or api_key_env, not both"
            )
        backends[name] = ProxyBackendConfig(
            name=name,
            base_url=_string(raw_backend, "base_url"),
            model=_string(raw_backend, "model"),
            api_key=api_key,
            api_key_env=api_key_env,
            timeout_seconds=_positive_float(
                raw_backend,
                "timeout_seconds",
                default=300.0,
            ),
            strip_tools=_bool(raw_backend, "strip_tools", default=False),
        )
    return backends


def _load_observability(data: Any) -> ProxyObservabilityConfig:
    if data is None:
        return ProxyObservabilityConfig()
    if not isinstance(data, dict):
        raise ProxyConfigError("observability must be a mapping")
    prompt_capture = _string(
        data,
        "prompt_capture",
        default=PROMPT_CAPTURE_REDACTED,
    )
    if prompt_capture not in PROMPT_CAPTURE_MODES:
        raise ProxyConfigError(
            "observability prompt_capture must be one of: "
            + ", ".join(PROMPT_CAPTURE_MODES)
        )
    return ProxyObservabilityConfig(
        enabled=_bool(data, "enabled", default=False),
        log_path=_string(data, "log_path", default=DEFAULT_LOG_PATH),
        prompt_capture=prompt_capture,
        max_bytes=_non_negative_int(data, "max_bytes", default=10_485_760),
        backups=_non_negative_int(data, "backups", default=5),
    )


def _load_health(data: Any) -> ProxyHealthConfig:
    if data is None:
        return ProxyHealthConfig()
    if not isinstance(data, dict):
        raise ProxyConfigError("health must be a mapping")
    return ProxyHealthConfig(
        backend_timeout_seconds=_positive_float(
            data,
            "backend_timeout_seconds",
            default=1.0,
        )
    )


def _load_engine_backends(data: Any, backends: dict[str, ProxyBackendConfig]) -> dict[str, str]:
    if not isinstance(data, dict) or not data:
        raise ProxyConfigError(
            "routing proxy config requires non-empty engine_backends mapping"
        )
    engine_backends: dict[str, str] = {}
    for engine, backend_name in data.items():
        if not isinstance(engine, str) or not engine.strip():
            raise ProxyConfigError("engine_backends keys must be non-empty strings")
        if not isinstance(backend_name, str) or not backend_name.strip():
            raise ProxyConfigError(f"engine_backends {engine!r} must name a backend")
        if backend_name not in backends:
            raise ProxyConfigError(
                f"engine_backends {engine!r} references undefined backend {backend_name!r}"
            )
        engine_backends[engine] = backend_name
    return engine_backends


def _load_fallback_backends(
    data: Any,
    backends: dict[str, ProxyBackendConfig],
) -> dict[str, tuple[str, ...]]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ProxyConfigError("fallback_backends must be a mapping")
    fallback_backends: dict[str, tuple[str, ...]] = {}
    for backend_name, raw_chain in data.items():
        if backend_name not in backends:
            raise ProxyConfigError(
                f"fallback_backends references undefined backend {backend_name!r}"
            )
        if not isinstance(raw_chain, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_chain
        ):
            raise ProxyConfigError(
                f"fallback_backends {backend_name!r} must be a list of backend names"
            )
        chain = tuple(raw_chain)
        for fallback_name in chain:
            if fallback_name not in backends:
                raise ProxyConfigError(
                    f"fallback_backends {backend_name!r} references undefined backend {fallback_name!r}"
                )
        fallback_backends[backend_name] = chain
    return fallback_backends


def _validate_no_fallback_cycles(fallback_backends: dict[str, tuple[str, ...]]) -> None:
    def visit(name: str, trail: tuple[str, ...]) -> None:
        if name in trail:
            raise ProxyConfigError(
                "fallback_backends contains a cycle: "
                + " -> ".join((*trail, name))
            )
        for next_name in fallback_backends.get(name, ()):
            visit(next_name, (*trail, name))

    for backend_name in fallback_backends:
        visit(backend_name, ())


def _validate_resolved_env_secrets(
    proxy: ProxyServerConfig,
    backends: dict[str, ProxyBackendConfig],
) -> None:
    if proxy.api_key_env and not proxy.resolved_api_key:
        raise ProxyConfigError(
            f"proxy api_key_env {proxy.api_key_env!r} is not set"
        )
    for backend in backends.values():
        if backend.api_key_env and not backend.resolved_api_key:
            raise ProxyConfigError(
                f"backend {backend.name!r} api_key_env {backend.api_key_env!r} is not set"
            )


def _string(data: dict[str, Any], key: str, default: str | None = None) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ProxyConfigError(f"{key} must be a non-empty string")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProxyConfigError(f"{key} must be a non-empty string")
    return value


def _positive_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProxyConfigError(f"{key} must be a positive integer")
    return value


def _non_negative_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProxyConfigError(f"{key} must be a non-negative integer")
    return value


def _positive_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ProxyConfigError(f"{key} must be a positive number")
    return float(value)


def _bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ProxyConfigError(f"{key} must be a boolean")
    return value


def _string_tuple(
    data: dict[str, Any],
    key: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ProxyConfigError(f"{key} must be a list of non-empty strings")
    return tuple(dict.fromkeys(value))
