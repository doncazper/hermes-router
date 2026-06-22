"""Product setup, validation, and doctor helpers for the local proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources
import json
from pathlib import Path
import shutil
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
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
OLLAMA_RECOMMENDED_MODELS = (
    "qwen3:0.6b",
    "qwen3:4b",
    "qwen3:14b",
    "qwen2.5-coder:7b",
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
    detection: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FirstRunSignals:
    ollama_installed: bool
    ollama_running: bool
    lmstudio_running: bool
    ollama_models: tuple[str, ...] = ()
    lmstudio_models: tuple[str, ...] = ()
    recommended_preset: str = "lmstudio"
    notes: tuple[str, ...] = ()

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
    proxy_endpoint: str | None = None
    telemetry_log_path: str | None = None
    remediation: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "proxy_config_valid": self.proxy_config_valid,
            "router_config_valid": self.router_config_valid,
            "proxy_config": self.proxy_config,
            "router_config": self.router_config,
            "backends": [backend.to_dict() for backend in self.backends],
            "errors": list(self.errors),
            "proxy_endpoint": self.proxy_endpoint,
            "telemetry_log_path": self.telemetry_log_path,
            "remediation": list(self.remediation),
        }


def initialize_product_config(
    *,
    preset: str | None = None,
    auto_detect: bool = False,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    proxy_port: int = DEFAULT_PROXY_PORT,
    force: bool = False,
    interactive: bool = False,
    input_func=input,
) -> InitResult:
    if preset and auto_detect:
        raise ValueError("pass either --preset or --auto, not both")

    signals = detect_first_run_environment() if auto_detect else None
    selected_preset = _select_preset(
        preset=preset,
        interactive=interactive,
        auto_detect=auto_detect,
        signals=signals,
        input_func=input_func,
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
    messages.extend(_init_guidance(selected_preset, signals))
    messages.append(f"Run: model-router-proxy --config {proxy_config}")
    messages.append(f"Agent endpoint: http://127.0.0.1:{proxy_port}/v1")
    messages.append(
        "Telemetry: model-router telemetry summary "
        f"--events {log_dir / 'routing-events.jsonl'} "
        f"--feedback {expanded_dir / 'routing-feedback.jsonl'}"
    )

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
        detection=signals.to_dict() if signals else {},
    )


def detect_first_run_environment(timeout_seconds: float = 0.25) -> FirstRunSignals:
    ollama_installed = shutil.which("ollama") is not None
    ollama_models = _fetch_model_ids(OLLAMA_BASE_URL, timeout_seconds=timeout_seconds)
    lmstudio_models = _fetch_model_ids(
        LMSTUDIO_BASE_URL,
        timeout_seconds=timeout_seconds,
    )
    ollama_running = ollama_models is not None
    lmstudio_running = lmstudio_models is not None
    recommended_preset = _recommended_preset_from_signals(
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        lmstudio_running=lmstudio_running,
    )
    notes = _signal_notes(
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        lmstudio_running=lmstudio_running,
        recommended_preset=recommended_preset,
    )
    return FirstRunSignals(
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        lmstudio_running=lmstudio_running,
        ollama_models=tuple(sorted(ollama_models or ())),
        lmstudio_models=tuple(sorted(lmstudio_models or ())),
        recommended_preset=recommended_preset,
        notes=notes,
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
            proxy_endpoint=None,
            telemetry_log_path=None,
            remediation=("Fix routing proxy config before running doctor again.",),
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
        backend.ok for backend in backends
    )
    remediation = _doctor_remediation(config, backends, router_config_valid)
    return DoctorReport(
        ok=ok,
        proxy_config_valid=proxy_config_valid,
        router_config_valid=router_config_valid,
        proxy_config=proxy_config_source,
        router_config=router_config_source,
        backends=backends,
        errors=tuple(errors),
        proxy_endpoint=f"http://{config.proxy.host}:{config.proxy.port}/v1",
        telemetry_log_path=(
            config.observability.log_path if config.observability.enabled else None
        ),
        remediation=remediation,
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
            body = response.read()
        model_ok, model_detail = _backend_model_detail(backend, body)
        ok = 200 <= status_code < 500 and model_ok
        detail = f"reachable: HTTP {status_code}"
        if model_detail:
            detail += f"; {model_detail}"
        return BackendHealth(
            backend=backend.name,
            reachable=True,
            ok=ok,
            status_code=status_code,
            detail=detail,
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


def _backend_model_detail(
    backend: ProxyBackendConfig,
    body: bytes,
) -> tuple[bool, str | None]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True, "model list unavailable"
    if not isinstance(payload, dict):
        return True, "model list unavailable"
    data = payload.get("data")
    if not isinstance(data, list):
        return True, "model list unavailable"
    model_ids = {
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if backend.model in model_ids:
        return True, f"configured model {backend.model!r} listed"
    return False, f"configured model {backend.model!r} not listed"


def preset_template_names() -> tuple[str, ...]:
    return tuple(f"proxy_template_{preset}.yaml" for preset in PRESETS)


def _select_preset(
    *,
    preset: str | None,
    interactive: bool,
    auto_detect: bool,
    signals: FirstRunSignals | None,
    input_func,
) -> str:
    if preset:
        return preset
    if auto_detect and signals is not None:
        return signals.recommended_preset
    if interactive:
        return _ask_preset(input_func)
    return "lmstudio"


def _fetch_model_ids(base_url: str, *, timeout_seconds: float) -> tuple[str, ...] | None:
    url = urljoin(base_url.rstrip("/") + "/", "models")
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
    except (HTTPError, OSError, URLError):
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return ()
    return tuple(
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )


def _recommended_preset_from_signals(
    *,
    ollama_installed: bool,
    ollama_running: bool,
    lmstudio_running: bool,
) -> str:
    if ollama_running:
        return "ollama"
    if lmstudio_running:
        return "lmstudio"
    if ollama_installed:
        return "ollama"
    return "lmstudio"


def _signal_notes(
    *,
    ollama_installed: bool,
    ollama_running: bool,
    lmstudio_running: bool,
    recommended_preset: str,
) -> tuple[str, ...]:
    notes = [f"Recommended preset: {recommended_preset}."]
    if ollama_running:
        notes.append(f"Ollama is reachable at {OLLAMA_BASE_URL}.")
    elif ollama_installed:
        notes.append("Ollama is installed but not reachable; start it with `ollama serve`.")
    else:
        notes.append("Ollama command not found.")
    if lmstudio_running:
        notes.append(f"LM Studio-style server is reachable at {LMSTUDIO_BASE_URL}.")
    else:
        notes.append(f"No LM Studio-style server detected at {LMSTUDIO_BASE_URL}.")
    return tuple(notes)


def _init_guidance(
    selected_preset: str,
    signals: FirstRunSignals | None,
) -> tuple[str, ...]:
    messages: list[str] = []
    if signals is not None:
        messages.extend(signals.notes)
    if selected_preset == "ollama":
        if signals is not None and not signals.ollama_running:
            messages.append("Start Ollama before running the proxy: ollama serve")
        models = set(signals.ollama_models) if signals is not None else set()
        missing = (
            tuple(model for model in OLLAMA_RECOMMENDED_MODELS if model not in models)
            if signals is not None and signals.ollama_running
            else OLLAMA_RECOMMENDED_MODELS
        )
        if missing:
            messages.append("Recommended Ollama model pulls:")
            messages.extend(f"- ollama pull {model}" for model in missing)
    if selected_preset == "lmstudio":
        messages.append(
            "LM Studio: start the local server at http://127.0.0.1:1234/v1 "
            "and edit backend model ids to the exact names LM Studio advertises."
        )
        messages.append(
            "Replace lmstudio-fast-model, lmstudio-balanced-model, "
            "lmstudio-reasoning-model, and lmstudio-code-model as needed."
        )
    return tuple(messages)


def _doctor_remediation(
    config: RoutingProxyConfig,
    backends: tuple[BackendHealth, ...],
    router_config_valid: bool,
) -> tuple[str, ...]:
    messages: list[str] = []
    if not router_config_valid:
        messages.append("Fix the router_config path before starting the proxy.")
    backend_by_name = config.backends
    unreachable_base_urls = {
        backend_by_name[health.backend].base_url
        for health in backends
        if not health.reachable and health.backend in backend_by_name
    }
    if any(base_url.startswith(OLLAMA_BASE_URL) for base_url in unreachable_base_urls):
        messages.append("Ollama backend unreachable; start Ollama with `ollama serve`.")
    if any(base_url.startswith(LMSTUDIO_BASE_URL) for base_url in unreachable_base_urls):
        messages.append(
            "LM Studio backend unreachable; start the LM Studio local server "
            "at http://127.0.0.1:1234/v1."
        )
    for health in backends:
        backend = backend_by_name.get(health.backend)
        if backend is None:
            continue
        if "not listed" not in health.detail:
            continue
        if backend.base_url.startswith(OLLAMA_BASE_URL):
            messages.append(
                f"Backend {backend.name} model missing; run `ollama pull {backend.model}`."
            )
        elif backend.base_url.startswith(LMSTUDIO_BASE_URL):
            messages.append(
                f"Backend {backend.name} model {backend.model!r} is not listed; "
                "edit routing_proxy.yaml to use the exact LM Studio model id."
            )
        else:
            messages.append(
                f"Backend {backend.name} model {backend.model!r} is not listed by "
                f"{backend.base_url}; update the model or upstream."
            )
    if config.observability.enabled:
        messages.append(
            "Telemetry is enabled; inspect dogfood data with "
            "`model-router telemetry summary`."
        )
    messages.append(
        f"Agent base URL: http://{config.proxy.host}:{config.proxy.port}/v1 "
        "with model `model-router`."
    )
    messages.append(f"Start proxy: model-router-proxy --config {config.source_path}")
    return tuple(dict.fromkeys(messages))


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
