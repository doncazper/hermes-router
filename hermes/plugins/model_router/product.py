"""Product setup, validation, and doctor helpers for the local proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import yaml

from hermes.plugins.model_router.config import (
    RouterConfigError,
    default_config_text,
    load_router_config,
)
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)


DEFAULT_CONFIG_DIR = "~/.model-router"
DEFAULT_PROXY_PORT = 8082
PRODUCT_DATA_PACKAGE = "hermes.plugins.model_router.data"
PRESETS = (
    "lmstudio",
    "ollama",
    "llamacpp",
    "localai",
    "hosted-openai-compatible",
)


@dataclass(frozen=True)
class InitResult:
    ok: bool
    config_dir: str
    preset: str
    model_router_config: str
    routing_proxy_config: str
    log_dir: str
    written: tuple[str, ...]
    skipped: tuple[str, ...]
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendHealth:
    backend: str
    reachable: bool
    ok: bool
    status_code: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    proxy_config_valid: bool
    router_config_valid: bool
    proxy_config: str
    router_config: str | None
    backends: tuple[BackendHealth, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "proxy_config_valid": self.proxy_config_valid,
            "router_config_valid": self.router_config_valid,
            "proxy_config": self.proxy_config,
            "router_config": self.router_config,
            "backends": [backend.to_dict() for backend in self.backends],
            "errors": list(self.errors),
        }


def initialize_product_config(
    *,
    preset: str | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    proxy_port: int = DEFAULT_PROXY_PORT,
    force: bool = False,
    interactive: bool = False,
    input_func=input,
) -> InitResult:
    selected_preset = preset or (
        _ask_preset(input_func) if interactive else "lmstudio"
    )
    _validate_preset(selected_preset)

    expanded_dir = Path(config_dir).expanduser()
    model_config = expanded_dir / "model_router.yaml"
    proxy_config = expanded_dir / "routing_proxy.yaml"
    log_dir = expanded_dir / "logs"
    written: list[str] = []
    skipped: list[str] = []
    messages: list[str] = []

    if interactive:
        proxy_port = _ask_int(
            input_func,
            "Proxy port",
            default=proxy_port,
        )

    proxy_data = _template_data(selected_preset)
    proxy_data.setdefault("proxy", {})["port"] = proxy_port
    proxy_data["router_config"] = str(model_config)
    proxy_data.setdefault("observability", {})["log_path"] = str(
        log_dir / "routing-events.jsonl"
    )

    if interactive:
        _customize_backends(proxy_data, input_func)

    expanded_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    messages.append(f"Created config directory: {expanded_dir}")
    messages.append(f"Created log directory: {log_dir}")

    _write_or_skip(
        model_config,
        default_config_text(),
        force=force,
        written=written,
        skipped=skipped,
    )
    _write_or_skip(
        proxy_config,
        yaml.safe_dump(proxy_data, sort_keys=False),
        force=force,
        written=written,
        skipped=skipped,
    )

    ok = not skipped
    if skipped:
        messages.append("Existing files skipped; pass --force to overwrite.")
    else:
        messages.append("Configuration ready.")
    messages.append(f"Run: model-router-proxy --config {proxy_config}")
    messages.append(f"Agent endpoint: http://127.0.0.1:{proxy_port}/v1")

    return InitResult(
        ok=ok,
        config_dir=str(expanded_dir),
        preset=selected_preset,
        model_router_config=str(model_config),
        routing_proxy_config=str(proxy_config),
        log_dir=str(log_dir),
        written=tuple(written),
        skipped=tuple(skipped),
        messages=tuple(messages),
    )


def validate_proxy_config(config_path: str | Path) -> RoutingProxyConfig:
    return load_proxy_config(config_path)


def doctor_proxy_config(
    config_path: str | Path,
    *,
    timeout_seconds: float | None = None,
) -> DoctorReport:
    errors: list[str] = []
    backends: tuple[BackendHealth, ...] = ()
    proxy_config_valid = False
    router_config_valid = False
    router_config_source: str | None = None
    proxy_config_source = str(Path(config_path).expanduser())
    try:
        config = load_proxy_config(config_path)
        proxy_config_valid = True
        proxy_config_source = config.source_path
    except ProxyConfigError as exc:
        errors.append(str(exc))
        return DoctorReport(
            ok=False,
            proxy_config_valid=False,
            router_config_valid=False,
            proxy_config=proxy_config_source,
            router_config=None,
            backends=(),
            errors=tuple(errors),
        )

    try:
        router_config = load_router_config(config.router_config)
        router_config_valid = True
        router_config_source = router_config.source_path
    except RouterConfigError as exc:
        errors.append(str(exc))
        router_config_source = config.router_config

    timeout = timeout_seconds or config.health.backend_timeout_seconds
    backends = tuple(
        check_backend_health(backend, timeout_seconds=timeout)
        for backend in config.backends.values()
    )
    ok = proxy_config_valid and router_config_valid and all(
        backend.reachable for backend in backends
    )
    return DoctorReport(
        ok=ok,
        proxy_config_valid=proxy_config_valid,
        router_config_valid=router_config_valid,
        proxy_config=proxy_config_source,
        router_config=router_config_source,
        backends=backends,
        errors=tuple(errors),
    )


def check_backend_health(
    backend: ProxyBackendConfig,
    *,
    timeout_seconds: float,
) -> BackendHealth:
    url = urljoin(backend.base_url.rstrip("/") + "/", "models")
    headers = {}
    api_key = backend.resolved_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
        return BackendHealth(
            backend=backend.name,
            reachable=True,
            ok=200 <= status_code < 500,
            status_code=status_code,
            detail=f"reachable: HTTP {status_code}",
        )
    except HTTPError as exc:
        return BackendHealth(
            backend=backend.name,
            reachable=True,
            ok=False,
            status_code=int(exc.code),
            detail=f"reachable but returned HTTP {exc.code}",
        )
    except (OSError, URLError) as exc:
        return BackendHealth(
            backend=backend.name,
            reachable=False,
            ok=False,
            status_code=None,
            detail=str(exc),
        )


def preset_template_names() -> tuple[str, ...]:
    return tuple(f"proxy_template_{preset}.yaml" for preset in PRESETS)


def _template_data(preset: str) -> dict[str, Any]:
    template = resources.files(PRODUCT_DATA_PACKAGE).joinpath(
        f"proxy_template_{preset}.yaml"
    )
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"proxy template for {preset} must be a mapping")
    return data


def _write_or_skip(
    path: Path,
    content: str,
    *,
    force: bool,
    written: list[str],
    skipped: list[str],
) -> None:
    if path.exists() and not force:
        skipped.append(str(path))
        return
    path.write_text(content, encoding="utf-8")
    written.append(str(path))


def _ask_preset(input_func) -> str:
    print("Provider preset:")
    for index, name in enumerate(PRESETS, start=1):
        print(f"{index}. {name}")
    answer = input_func("Choose preset [1]: ").strip()
    if not answer:
        return "lmstudio"
    if answer.isdigit():
        index = int(answer)
        if 1 <= index <= len(PRESETS):
            return PRESETS[index - 1]
    _validate_preset(answer)
    return answer


def _ask_int(input_func, label: str, *, default: int) -> int:
    answer = input_func(f"{label} [{default}]: ").strip()
    if not answer:
        return default
    try:
        parsed = int(answer)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _customize_backends(data: dict[str, Any], input_func) -> None:
    backends = data.get("backends", {})
    if not isinstance(backends, dict):
        return
    for name, backend in backends.items():
        if not isinstance(backend, dict):
            continue
        current_url = backend.get("base_url", "")
        current_model = backend.get("model", "")
        url = input_func(f"{name} base_url [{current_url}]: ").strip()
        model = input_func(f"{name} model [{current_model}]: ").strip()
        if url:
            backend["base_url"] = url
        if model:
            backend["model"] = model


def _validate_preset(preset: str) -> None:
    if preset not in PRESETS:
        raise ValueError("preset must be one of: " + ", ".join(PRESETS))
