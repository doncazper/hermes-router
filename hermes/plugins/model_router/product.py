"""Product setup, validation, and doctor helpers for the local proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources
import json
from pathlib import Path
import shutil
import socket
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import yaml

from hermes.plugins.model_router.config import (
    RouterConfigError,
    default_config_text,
    load_router_config,
)
from hermes.plugins.model_router.model_advisor import detect_hardware_profile
from hermes.plugins.model_router.maturity import feature_maturity_state
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)
from hermes.plugins.model_router.setup_assistant import (
    DiscoveredModel,
    default_model_dirs,
    mlx_lm_download_suggestions,
    plan_model_downloads,
    scan_local_environment,
)


DEFAULT_CONFIG_DIR = "~/.model-router"
DEFAULT_PROXY_PORT = 8082
PRODUCT_DATA_PACKAGE = "hermes.plugins.model_router.data"
PRESETS = (
    "lmstudio",
    "ollama",
    "llamacpp",
    "mlx-lm",
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
    apple_silicon: bool = False
    mlx_lm_available: bool = False
    llama_server_available: bool = False
    ollama_models: tuple[str, ...] = ()
    lmstudio_models: tuple[str, ...] = ()
    mlx_lm_models: tuple[str, ...] = ()
    gguf_models: tuple[str, ...] = ()
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
    maturity: dict[str, Any] = field(default_factory=feature_maturity_state)

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
            "maturity": self.maturity,
        }


def initialize_product_config(
    *,
    preset: str | None = None,
    auto_detect: bool = False,
    auto_models: bool = False,
    model_dirs: Sequence[str | Path] | None = None,
    profile: str = "balanced",
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

    if auto_models:
        _apply_proxy_auto_models(
            proxy_data,
            preset=selected_preset,
            config_dir=expanded_dir,
            model_dirs=model_dirs,
            profile=profile,
            messages=messages,
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
    discovery = scan_local_environment()
    hardware = detect_hardware_profile()
    ollama_installed = discovery.commands.get("ollama", False)
    ollama_models = _fetch_model_ids(OLLAMA_BASE_URL, timeout_seconds=timeout_seconds)
    lmstudio_models = _fetch_model_ids(
        LMSTUDIO_BASE_URL,
        timeout_seconds=timeout_seconds,
    )
    ollama_running = ollama_models is not None
    lmstudio_running = lmstudio_models is not None
    mlx_lm_models = tuple(
        model.repo_id for model in discovery.models if _is_mlx_lm_compatible_model(model)
    )
    gguf_models = tuple(
        model.repo_id for model in discovery.models if _is_llamacpp_compatible_model(model)
    )
    recommended_preset = _recommended_preset_from_signals(
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        lmstudio_running=lmstudio_running,
        apple_silicon=hardware.apple_silicon,
        mlx_lm_available=discovery.commands.get("mlx_lm.server", False),
        llama_server_available=discovery.commands.get("llama-server", False),
        mlx_lm_models=mlx_lm_models,
        gguf_models=gguf_models,
    )
    notes = _signal_notes(
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        lmstudio_running=lmstudio_running,
        apple_silicon=hardware.apple_silicon,
        mlx_lm_available=discovery.commands.get("mlx_lm.server", False),
        llama_server_available=discovery.commands.get("llama-server", False),
        mlx_lm_models=mlx_lm_models,
        gguf_models=gguf_models,
        recommended_preset=recommended_preset,
    )
    return FirstRunSignals(
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        lmstudio_running=lmstudio_running,
        apple_silicon=hardware.apple_silicon,
        mlx_lm_available=discovery.commands.get("mlx_lm.server", False),
        llama_server_available=discovery.commands.get("llama-server", False),
        ollama_models=tuple(sorted(ollama_models or ())),
        lmstudio_models=tuple(sorted(lmstudio_models or ())),
        mlx_lm_models=tuple(sorted(mlx_lm_models)),
        gguf_models=tuple(sorted(gguf_models)),
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
        _backend_ok_for_doctor(config.backends[backend.backend], backend)
        for backend in backends
        if backend.backend in config.backends
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
    apple_silicon: bool = False,
    mlx_lm_available: bool = False,
    llama_server_available: bool = False,
    mlx_lm_models: tuple[str, ...] = (),
    gguf_models: tuple[str, ...] = (),
) -> str:
    if ollama_running:
        return "ollama"
    if lmstudio_running:
        return "lmstudio"
    if ollama_installed:
        return "ollama"
    if apple_silicon and (mlx_lm_available or mlx_lm_models):
        return "mlx-lm"
    if llama_server_available or gguf_models:
        return "llamacpp"
    return "lmstudio"


def _signal_notes(
    *,
    ollama_installed: bool,
    ollama_running: bool,
    lmstudio_running: bool,
    apple_silicon: bool = False,
    mlx_lm_available: bool = False,
    llama_server_available: bool = False,
    mlx_lm_models: tuple[str, ...] = (),
    gguf_models: tuple[str, ...] = (),
    recommended_preset: str,
) -> tuple[str, ...]:
    notes = [f"Recommended preset: {recommended_preset}."]
    notes.append("Apple Silicon detected." if apple_silicon else "Non-Apple-Silicon machine detected.")
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
    if mlx_lm_available:
        notes.append("mlx_lm.server command detected.")
    if llama_server_available:
        notes.append("llama-server command detected.")
    if mlx_lm_models:
        notes.append(f"Local MLX-compatible models detected: {len(mlx_lm_models)}.")
    if gguf_models:
        notes.append(f"Local GGUF models detected: {len(gguf_models)}.")
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
    if selected_preset == "mlx-lm":
        messages.append(
            "MLX-LM: replace every REPLACE_WITH_MLX_* placeholder with an "
            "exact MLX/Hugging Face repo id or local model path."
        )
        messages.append(
            "Managed MLX-LM runtimes start on first routed chat request and "
            "idle out after 900 seconds."
        )
        messages.append(
            "MLX-LM /v1/responses translation is deferred; use chat-compatible "
            "clients or an upstream that supports Responses API."
        )
    return tuple(messages)


_MLX_BACKEND_ROUTES = {
    "fast": "fast_local",
    "balanced": "balanced_local",
    "reasoning": "reasoning_local",
    "code": "code_agent",
}


def _apply_proxy_auto_models(
    proxy_data: dict[str, Any],
    *,
    preset: str,
    config_dir: Path,
    model_dirs: Sequence[str | Path] | None,
    profile: str,
    messages: list[str],
) -> None:
    if preset not in {"mlx-lm", "llamacpp"}:
        messages.append(
            "Auto model selection currently supports managed mlx-lm and llamacpp "
            "presets."
        )
        return
    scan_dirs = _proxy_model_scan_dirs(config_dir, model_dirs)
    discovery = scan_local_environment(model_dirs=scan_dirs)
    suggestions = _proxy_download_suggestions(
        preset=preset,
        discovery=discovery,
        profile=profile,
        local_root=config_dir / "models",
    )
    messages.append(
        "Scanned for local runtime-compatible models in: "
        + ", ".join(str(path) for path in scan_dirs)
    )
    backends = proxy_data.get("backends", {})
    if not isinstance(backends, dict):
        return
    for backend_name, route in _MLX_BACKEND_ROUTES.items():
        backend = backends.get(backend_name)
        if not isinstance(backend, dict):
            continue
        model = _select_proxy_model(discovery.models, route, preset)
        if model is not None:
            _apply_selected_proxy_model(
                backend,
                backend_name=backend_name,
                preset=preset,
                model=model,
            )
            messages.append(_selected_proxy_model_message(backend_name, route, model, preset))
            continue
        suggestion = suggestions.get(route)
        if suggestion is None:
            continue
        messages.append(
            f"No compatible local {preset} model found for backend {backend_name} "
            f"({route}). Recommended download: {' '.join(suggestion.command)}"
        )


def _proxy_download_suggestions(
    *,
    preset: str,
    discovery,
    profile: str,
    local_root: Path,
) -> dict[str, Any]:
    if preset == "mlx-lm":
        return {
            suggestion.route: suggestion
            for suggestion in mlx_lm_download_suggestions(local_root=local_root)
        }
    plan = plan_model_downloads(
        discovery=discovery,
        profile=profile,
        routes=tuple(_MLX_BACKEND_ROUTES.values()),
        local_root=local_root,
    )
    return {suggestion.route: suggestion for suggestion in plan.suggestions}


def _proxy_model_scan_dirs(
    config_dir: Path,
    model_dirs: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    configured = (
        tuple(Path(path).expanduser() for path in model_dirs)
        if model_dirs is not None
        else default_model_dirs()
    )
    config_models = config_dir / "models"
    if config_models in configured:
        return configured
    return (*configured, config_models)


def _select_proxy_model(
    models: tuple[DiscoveredModel, ...],
    route: str,
    preset: str,
) -> DiscoveredModel | None:
    predicate = (
        _is_mlx_lm_compatible_model
        if preset == "mlx-lm"
        else _is_llamacpp_compatible_model
    )
    compatible = [model for model in models if predicate(model)]
    role_matches = [model for model in compatible if route in model.roles]
    if not role_matches:
        return None
    return sorted(role_matches, key=lambda model: _proxy_model_rank(model, route))[0]


def _is_mlx_lm_compatible_model(model: DiscoveredModel) -> bool:
    text = f"{model.repo_id} {model.path}".lower()
    if any(token in text for token in ("gguf", "embedding", "embed", "bge", "flux")):
        return False
    return model.repo_id.startswith("mlx-community/") or "mlx" in text


def _is_llamacpp_compatible_model(model: DiscoveredModel) -> bool:
    text = f"{model.repo_id} {model.path}".lower()
    return "gguf" in text or _first_gguf_file(Path(model.path)) is not None


def _proxy_model_rank(model: DiscoveredModel, route: str) -> tuple[int, int, int, str]:
    text = model.repo_id.lower()
    role_penalty = 0 if route in model.roles else 10
    if route == "fast_local":
        route_penalty = 0 if any(token in text for token in ("0.6b", "1b")) else 3
    elif route == "balanced_local":
        route_penalty = 0 if any(token in text for token in ("4b", "7b")) else 3
    elif route == "reasoning_local":
        route_penalty = 0 if any(token in text for token in ("deepseek", "reason", "8b")) else 3
    elif route == "code_agent":
        route_penalty = 0 if any(token in text for token in ("coder", "code")) else 3
    else:
        route_penalty = 1
    source_penalty = 0 if model.source in {"huggingface_cache", "local_directory"} else 1
    return (role_penalty, route_penalty, source_penalty, model.repo_id)


def _model_reference_for_mlx_lm(model: DiscoveredModel) -> str:
    if model.source == "huggingface_cache":
        return model.repo_id
    return model.path


def _selected_proxy_model_message(
    backend_name: str,
    route: str,
    model: DiscoveredModel,
    preset: str,
) -> str:
    if preset == "mlx-lm":
        model_ref = _model_reference_for_mlx_lm(model)
    else:
        model_ref = str(_first_gguf_file(Path(model.path)) or model.path)
    return (
        f"Auto-selected {model_ref} for {preset} backend {backend_name} "
        f"({route})."
    )


def _apply_selected_proxy_model(
    backend: dict[str, Any],
    *,
    backend_name: str,
    preset: str,
    model: DiscoveredModel,
) -> None:
    if preset == "mlx-lm":
        _replace_proxy_backend_model(backend, _model_reference_for_mlx_lm(model))
        return

    model_file = _first_gguf_file(Path(model.path))
    if model_file is None:
        return
    model_id = model_file.stem
    backend["model"] = model_id
    port = _backend_port(backend) or _default_managed_port(backend_name)
    backend["runtime"] = {
        "enabled": True,
        "kind": "llama-server",
        "command": [
            "llama-server",
            "-m",
            str(model_file),
            "--port",
            str(port),
        ],
        "readiness_url": f"http://127.0.0.1:{port}/v1/models",
        "readiness_timeout_seconds": 30,
        "idle_timeout_seconds": 900,
        "shutdown_timeout_seconds": 5,
        "log_path": f"~/.model-router/logs/llama-{backend_name}.log",
    }


def _first_gguf_file(path: Path) -> Path | None:
    if path.is_file() and path.name.lower().endswith(".gguf"):
        return path
    if not path.is_dir():
        return None
    try:
        candidates = sorted(
            child for child in path.iterdir() if child.name.lower().endswith(".gguf")
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def _backend_port(backend: dict[str, Any]) -> int | None:
    base_url = backend.get("base_url")
    if not isinstance(base_url, str):
        return None
    return urlparse(base_url).port


def _default_managed_port(backend_name: str) -> int:
    return {"fast": 8080, "balanced": 8081, "code": 8083, "reasoning": 8084}.get(
        backend_name,
        8080,
    )


def _replace_proxy_backend_model(backend: dict[str, Any], model_ref: str) -> None:
    previous = backend.get("model")
    backend["model"] = model_ref
    runtime = backend.get("runtime")
    if not isinstance(runtime, dict):
        return
    command = runtime.get("command")
    if not isinstance(command, list):
        return
    runtime["command"] = [
        model_ref
        if isinstance(item, str)
        and (item == previous or item.startswith("REPLACE_WITH_"))
        else item
        for item in command
    ]


def _doctor_remediation(
    config: RoutingProxyConfig,
    backends: tuple[BackendHealth, ...],
    router_config_valid: bool,
) -> tuple[str, ...]:
    messages: list[str] = []
    maturity = feature_maturity_state()
    feature_summary = ", ".join(
        f"{feature['feature_id']}={feature['maturity']}"
        for feature in maturity["features"]
    )
    messages.append(f"Feature maturity: {feature_summary}.")
    if config.proxy.routing_mode == "manual":
        messages.append(
            "Manual/basic router mode is beta; run decision and manual dogfood "
            "checks before release."
        )
    else:
        messages.append(
            "Decision router mode is the stable default; manual/basic mode should "
            "be dogfooded separately before release."
        )
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
    disabled_runtime_backends = [
        backend.name for backend in backend_by_name.values() if not backend.runtime.enabled
    ]
    if disabled_runtime_backends:
        messages.append(
            "Managed runtimes disabled for backends: "
            + ", ".join(sorted(disabled_runtime_backends))
            + "; start those upstream servers separately."
        )
    for backend in backend_by_name.values():
        runtime = backend.runtime
        if not runtime.enabled:
            continue
        messages.append(
            f"Backend {backend.name} managed runtime enabled ({runtime.kind}); "
            "starts on first routed request."
        )
        if _runtime_has_placeholder(backend):
            messages.append(
                f"Backend {backend.name} has placeholder MLX/runtime model values; "
                "replace REPLACE_WITH_* entries before dogfooding."
            )
        if not _runtime_command_available(runtime.command):
            messages.append(
                f"Backend {backend.name} runtime command missing: {runtime.command[0]}"
            )
        if runtime.kind == "mlx-lm":
            messages.append(
                "MLX-LM managed runtimes are chat/models-first; /v1/responses "
                "requires an upstream that supports Responses API."
            )
    for health in backends:
        backend = backend_by_name.get(health.backend)
        if backend is None:
            continue
        if backend.runtime.enabled and not health.reachable:
            if _runtime_port_open(backend.runtime.readiness_url):
                messages.append(
                    f"Backend {backend.name} runtime port is in use but readiness "
                    "is not HTTP-compatible; stop the conflicting process or fix "
                    "the runtime command."
                )
            messages.append(
                f"Backend {backend.name} readiness is not responding at "
                f"{backend.runtime.readiness_url}; the proxy will try to start it "
                "on first route."
            )
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


def _backend_ok_for_doctor(
    backend: ProxyBackendConfig,
    health: BackendHealth,
) -> bool:
    if health.ok:
        return True
    if not backend.runtime.enabled:
        return False
    if _runtime_has_placeholder(backend):
        return False
    if not _runtime_command_available(backend.runtime.command):
        return False
    if _runtime_port_open(backend.runtime.readiness_url):
        return False
    return True


def _runtime_has_placeholder(backend: ProxyBackendConfig) -> bool:
    values = (backend.model, *backend.runtime.command)
    return any("REPLACE_WITH_" in value for value in values)


def _runtime_command_available(command: tuple[str, ...]) -> bool:
    if not command:
        return False
    executable = command[0]
    path = Path(executable).expanduser()
    if "/" in executable:
        return path.exists()
    return shutil.which(executable) is not None


def _runtime_port_open(url: str, *, timeout_seconds: float = 0.05) -> bool:
    parsed = urlparse(url)
    if parsed.hostname is None or parsed.port is None:
        return False
    try:
        with socket.create_connection(
            (parsed.hostname, parsed.port),
            timeout=timeout_seconds,
        ):
            return True
    except OSError:
        return False


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
