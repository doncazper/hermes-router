"""Optional runtime adapter foundation for ModelRouter admin/operator surfaces.

Adapters coordinate proven runtimes; they are not a custom inference engine and
must not be required by route_fast or normal proxy forwarding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from hermes.plugins.model_router.proxy_config import ProxyBackendConfig


JsonRequester = Any
CommandRunner = Any
CommandResolver = Any


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class AdapterSupport:
    supported: bool
    disabled_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "disabled_reason": self.disabled_reason,
        }


@dataclass(frozen=True)
class RuntimeCapabilities:
    provider: str
    runtime_kind: str
    health: AdapterSupport
    discover_models: AdapterSupport
    list_loaded_models: AdapterSupport
    load_model: AdapterSupport
    unload_model: AdapterSupport
    logs: AdapterSupport
    endpoint_url: str | None = None
    detect_runtime: AdapterSupport = field(
        default_factory=lambda: AdapterSupport(True),
    )
    start_server: AdapterSupport = field(
        default_factory=lambda: AdapterSupport(
            False,
            "Runtime adapter does not expose a start action.",
        ),
    )
    stop_server: AdapterSupport = field(
        default_factory=lambda: AdapterSupport(
            False,
            "Runtime adapter does not expose a stop action.",
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "runtime_kind": self.runtime_kind,
            "endpoint_url": self.endpoint_url,
            "detect_runtime": self.detect_runtime.to_dict(),
            "health": self.health.to_dict(),
            "discover_models": self.discover_models.to_dict(),
            "list_loaded_models": self.list_loaded_models.to_dict(),
            "start_server": self.start_server.to_dict(),
            "stop_server": self.stop_server.to_dict(),
            "load_model": self.load_model.to_dict(),
            "unload_model": self.unload_model.to_dict(),
            "logs": self.logs.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeDetection:
    provider: str
    runtime_kind: str
    endpoint_url: str
    installed: bool | None
    available: bool | None
    detail: str
    command: tuple[str, ...] = ()
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_id": self.provider,
            "provider": self.provider,
            "runtime_kind": self.runtime_kind,
            "detected": self.installed is True or self.available is True,
            "endpoint_url": self.endpoint_url,
            "endpoint": self.endpoint_url,
            "installed": self.installed,
            "available": self.available,
            "version": self.version,
            "detail": self.detail,
            "command": list(self.command),
        }


@dataclass(frozen=True)
class RuntimeHealth:
    status: str
    reachable: bool
    ok: bool
    detail: str
    status_code: int | None = None
    checked_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reachable": self.reachable,
            "ok": self.ok,
            "detail": self.detail,
            "status_code": self.status_code,
            "checked_url": self.checked_url,
        }


@dataclass(frozen=True)
class RuntimeModel:
    model_id: str
    loaded: bool | None = None
    source: str = "runtime"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self.loaded,
            "source": self.source,
        }


@dataclass(frozen=True)
class RuntimeActionResult:
    ok: bool
    status: str
    message: str
    disabled_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "disabled_reason": self.disabled_reason,
        }


@dataclass(frozen=True)
class RuntimeLogInfo:
    supported: bool
    paths: tuple[str, ...] = ()
    tail_preview: tuple[str, ...] = ()
    error: str | None = None
    disabled_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "paths": list(self.paths),
            "tail_preview": list(self.tail_preview),
            "error": self.error,
            "disabled_reason": self.disabled_reason,
        }


class RuntimeAdapter(Protocol):
    backend: ProxyBackendConfig

    def endpoint_url(self) -> str:
        """Return the endpoint URL this adapter targets."""

    def capabilities(self) -> RuntimeCapabilities:
        """Return supported runtime actions and disabled reasons."""

    def detect(self) -> RuntimeDetection:
        """Return installation/configuration availability without raising."""

    def health(self, *, timeout_seconds: float = 0.25) -> RuntimeHealth:
        """Return best-effort health without raising to UI callers."""

    def discover_models(
        self,
        *,
        timeout_seconds: float = 0.25,
    ) -> tuple[RuntimeModel, ...]:
        """Return model ids visible to the runtime."""

    def list_loaded_models(
        self,
        *,
        timeout_seconds: float = 0.25,
    ) -> tuple[RuntimeModel, ...]:
        """Return loaded models when the runtime can distinguish them."""

    def load_model(self, model_id: str) -> RuntimeActionResult:
        """Load a model if the runtime supports explicit load control."""

    def unload_model(self, model_id: str) -> RuntimeActionResult:
        """Unload a model if the runtime supports explicit unload control."""

    def start_server(self) -> RuntimeActionResult:
        """Start a server if the adapter can do so safely."""

    def stop_server(self) -> RuntimeActionResult:
        """Stop a server if the adapter can do so safely."""

    def logs(self) -> RuntimeLogInfo:
        """Return safe runtime log metadata."""


class GenericOpenAICompatibleAdapter:
    """Adapter for local OpenAI-compatible `/v1/models` runtimes."""

    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
        command_runner: CommandRunner | None = None,
        command_resolver: CommandResolver | None = None,
        provider: str | None = None,
        runtime_kind: str | None = None,
    ) -> None:
        self.backend = backend
        self._requester = requester or _request_json
        self._command_runner = command_runner or _run_command
        self._command_resolver = command_resolver or _resolve_command
        self._provider = provider or provider_for_backend(backend)
        self._runtime_kind = runtime_kind or runtime_kind_for_backend(backend)

    def _available_command(self, *candidates: str) -> tuple[str, ...]:
        for candidate in candidates:
            if self._command_resolver(candidate):
                return (candidate,)
        return ()

    def _run_native_command(
        self,
        command: tuple[str, ...],
        *,
        timeout_seconds: float = 2.0,
    ) -> CommandResult:
        if not command:
            return CommandResult(127, stderr="runtime command unavailable")
        return self._command_runner(command, timeout_seconds)

    def endpoint_url(self) -> str:
        return self.backend.base_url.rstrip("/")

    def capabilities(self) -> RuntimeCapabilities:
        local = _is_local_base_url(self.backend.base_url)
        health_reason = None if local else "Hosted runtime health is not checked by settings."
        lifecycle_reason = (
            "External OpenAI-compatible runtime lifecycle is managed outside ModelRouter."
            if local
            else "Hosted runtime lifecycle is managed by the provider."
        )
        return RuntimeCapabilities(
            provider=self._provider,
            runtime_kind=self._runtime_kind,
            endpoint_url=self.endpoint_url(),
            detect_runtime=AdapterSupport(True),
            health=AdapterSupport(local, health_reason),
            discover_models=AdapterSupport(
                local,
                None if local else "Hosted model discovery is disabled by default.",
            ),
            list_loaded_models=AdapterSupport(
                False,
                "OpenAI-compatible /models does not distinguish loaded models.",
            ),
            load_model=AdapterSupport(
                False,
                "OpenAI-compatible runtimes do not expose a standard load action.",
            ),
            unload_model=AdapterSupport(
                False,
                "OpenAI-compatible runtimes do not expose a standard unload action.",
            ),
            start_server=AdapterSupport(False, lifecycle_reason),
            stop_server=AdapterSupport(False, lifecycle_reason),
            logs=AdapterSupport(
                self.backend.runtime.enabled and bool(self.backend.runtime.log_path),
                None
                if self.backend.runtime.enabled and self.backend.runtime.log_path
                else "No managed runtime log path is configured.",
            ),
        )

    def detect(self) -> RuntimeDetection:
        local = _is_local_base_url(self.backend.base_url)
        if local:
            return RuntimeDetection(
                provider=self._provider,
                runtime_kind=self._runtime_kind,
                endpoint_url=self.endpoint_url(),
                installed=None,
                available=True,
                detail="Local endpoint is configured; health check determines reachability.",
            )
        return RuntimeDetection(
            provider=self._provider,
            runtime_kind=self._runtime_kind,
            endpoint_url=self.endpoint_url(),
            installed=None,
            available=None,
            detail="Hosted endpoint configured; availability is not probed by default.",
        )

    def health(self, *, timeout_seconds: float = 0.25) -> RuntimeHealth:
        capability = self.capabilities().health
        models_url = _models_url(self.backend)
        if not capability.supported:
            return RuntimeHealth(
                status="unsupported",
                reachable=False,
                ok=False,
                detail=capability.disabled_reason or "Health unsupported.",
                checked_url=models_url,
            )
        try:
            status_code, payload = self._requester(
                models_url,
                _auth_headers(self.backend),
                timeout_seconds,
            )
        except HTTPError as exc:
            return RuntimeHealth(
                status="degraded",
                reachable=True,
                ok=False,
                detail=f"HTTP {exc.code}",
                status_code=exc.code,
                checked_url=models_url,
            )
        except (OSError, URLError, TimeoutError) as exc:
            return RuntimeHealth(
                status="unreachable",
                reachable=False,
                ok=False,
                detail=exc.__class__.__name__,
                checked_url=models_url,
            )
        except Exception as exc:
            return RuntimeHealth(
                status="error",
                reachable=False,
                ok=False,
                detail=exc.__class__.__name__,
                checked_url=models_url,
            )
        models = _models_from_payload(payload)
        if self.backend.model in {model.model_id for model in models}:
            return RuntimeHealth(
                status="ready",
                reachable=True,
                ok=True,
                detail="configured model listed by runtime",
                status_code=status_code,
                checked_url=models_url,
            )
        return RuntimeHealth(
            status="degraded",
            reachable=True,
            ok=False,
            detail=f"configured model {self.backend.model!r} not listed",
            status_code=status_code,
            checked_url=models_url,
        )

    def discover_models(
        self,
        *,
        timeout_seconds: float = 0.25,
    ) -> tuple[RuntimeModel, ...]:
        capability = self.capabilities().discover_models
        if not capability.supported:
            return ()
        try:
            _status, payload = self._requester(
                _models_url(self.backend),
                _auth_headers(self.backend),
                timeout_seconds,
            )
        except Exception:
            return ()
        return _models_from_payload(payload)

    def list_loaded_models(
        self,
        *,
        timeout_seconds: float = 0.25,
    ) -> tuple[RuntimeModel, ...]:
        del timeout_seconds
        return ()

    def load_model(self, model_id: str) -> RuntimeActionResult:
        del model_id
        reason = self.capabilities().load_model.disabled_reason
        return RuntimeActionResult(
            ok=False,
            status="unsupported",
            message=reason or "Load model unsupported.",
            disabled_reason=reason,
        )

    def unload_model(self, model_id: str) -> RuntimeActionResult:
        del model_id
        reason = self.capabilities().unload_model.disabled_reason
        return RuntimeActionResult(
            ok=False,
            status="unsupported",
            message=reason or "Unload model unsupported.",
            disabled_reason=reason,
        )

    def start_server(self) -> RuntimeActionResult:
        reason = self.capabilities().start_server.disabled_reason
        return RuntimeActionResult(
            ok=False,
            status="unsupported",
            message=reason or "Start server unsupported.",
            disabled_reason=reason,
        )

    def stop_server(self) -> RuntimeActionResult:
        reason = self.capabilities().stop_server.disabled_reason
        return RuntimeActionResult(
            ok=False,
            status="unsupported",
            message=reason or "Stop server unsupported.",
            disabled_reason=reason,
        )

    def logs(self) -> RuntimeLogInfo:
        reason = self.capabilities().logs.disabled_reason
        if reason:
            return RuntimeLogInfo(supported=False, disabled_reason=reason)
        path = str(Path(self.backend.runtime.log_path).expanduser())
        return RuntimeLogInfo(supported=True, paths=(path,))


class LMStudioAdapter(GenericOpenAICompatibleAdapter):
    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
        command_runner: CommandRunner | None = None,
        command_resolver: CommandResolver | None = None,
    ) -> None:
        super().__init__(
            backend,
            requester=requester,
            command_runner=command_runner,
            command_resolver=command_resolver,
            provider="lmstudio",
            runtime_kind="lmstudio",
        )

    def capabilities(self) -> RuntimeCapabilities:
        capabilities = super().capabilities()
        lifecycle_reason = (
            "LM Studio native lifecycle commands are not wired until a stable "
            "local CLI/API contract is confirmed."
        )
        return RuntimeCapabilities(
            provider=capabilities.provider,
            runtime_kind=capabilities.runtime_kind,
            endpoint_url=capabilities.endpoint_url,
            detect_runtime=capabilities.detect_runtime,
            health=capabilities.health,
            discover_models=capabilities.discover_models,
            list_loaded_models=AdapterSupport(
                False,
                "LM Studio loaded-model state is not exposed through the stable OpenAI-compatible API.",
            ),
            load_model=AdapterSupport(False, lifecycle_reason),
            unload_model=AdapterSupport(False, lifecycle_reason),
            start_server=AdapterSupport(False, lifecycle_reason),
            stop_server=AdapterSupport(False, lifecycle_reason),
            logs=capabilities.logs,
        )

    def detect(self) -> RuntimeDetection:
        command = self._available_command("lms", "lmstudio")
        installed = bool(command)
        return RuntimeDetection(
            provider="lmstudio",
            runtime_kind="lmstudio",
            endpoint_url=self.endpoint_url(),
            installed=installed,
            available=True,
            detail="LM Studio endpoint configured; health check determines whether the server is running.",
            command=command,
        )


class OllamaAdapter(GenericOpenAICompatibleAdapter):
    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
        command_runner: CommandRunner | None = None,
        command_resolver: CommandResolver | None = None,
    ) -> None:
        super().__init__(
            backend,
            requester=requester,
            command_runner=command_runner,
            command_resolver=command_resolver,
            provider="ollama",
            runtime_kind="ollama",
        )

    def capabilities(self) -> RuntimeCapabilities:
        capabilities = super().capabilities()
        command = self._available_command("ollama")
        cli_reason = "Ollama CLI not found; install Ollama or use /v1/models discovery."
        lifecycle_reason = (
            "External Ollama server lifecycle is managed by the Ollama app, "
            "OS service, or a future confirmation-gated supervisor."
        )
        load_reason = (
            "Ollama load/run and pull/download are separate actions; ModelRouter "
            "does not run or pull models silently."
        )
        return RuntimeCapabilities(
            provider=capabilities.provider,
            runtime_kind=capabilities.runtime_kind,
            endpoint_url=capabilities.endpoint_url,
            detect_runtime=capabilities.detect_runtime,
            health=capabilities.health,
            discover_models=AdapterSupport(
                capabilities.discover_models.supported or bool(command),
                None
                if capabilities.discover_models.supported or command
                else cli_reason,
            ),
            list_loaded_models=AdapterSupport(bool(command), None if command else cli_reason),
            load_model=AdapterSupport(False, load_reason),
            unload_model=AdapterSupport(
                bool(command),
                None
                if command
                else "Ollama unload requires the ollama CLI for `ollama stop <model>`.",
            ),
            start_server=AdapterSupport(False, lifecycle_reason),
            stop_server=AdapterSupport(False, lifecycle_reason),
            logs=capabilities.logs,
        )

    def detect(self) -> RuntimeDetection:
        command = self._available_command("ollama")
        installed = bool(command)
        return RuntimeDetection(
            provider="ollama",
            runtime_kind="ollama",
            endpoint_url=self.endpoint_url(),
            installed=installed,
            available=True,
            detail="Ollama endpoint configured; health check determines whether ollama serve is running.",
            command=command,
        )

    def discover_models(
        self,
        *,
        timeout_seconds: float = 0.25,
    ) -> tuple[RuntimeModel, ...]:
        command = self._available_command("ollama")
        if command:
            result = self._run_native_command(
                (*command, "list"),
                timeout_seconds=timeout_seconds,
            )
            if result.returncode == 0:
                models = _models_from_ollama_table(result.stdout, loaded=False)
                if models:
                    return models
        return super().discover_models(timeout_seconds=timeout_seconds)

    def list_loaded_models(
        self,
        *,
        timeout_seconds: float = 0.25,
    ) -> tuple[RuntimeModel, ...]:
        command = self._available_command("ollama")
        if not command:
            return ()
        result = self._run_native_command(
            (*command, "ps"),
            timeout_seconds=timeout_seconds,
        )
        if result.returncode != 0:
            return ()
        return _models_from_ollama_table(result.stdout, loaded=True)

    def unload_model(self, model_id: str) -> RuntimeActionResult:
        command = self._available_command("ollama")
        if not command:
            reason = self.capabilities().unload_model.disabled_reason
            return RuntimeActionResult(
                ok=False,
                status="unsupported",
                message=reason or "Ollama unload unsupported.",
                disabled_reason=reason,
            )
        model = model_id.strip()
        if not model:
            return RuntimeActionResult(
                ok=False,
                status="invalid_request",
                message="Ollama unload requires a non-empty model id.",
            )
        result = self._run_native_command((*command, "stop", model), timeout_seconds=10.0)
        if result.returncode == 0:
            return RuntimeActionResult(
                ok=True,
                status="unloaded",
                message=f"Ollama stop requested for {model}.",
            )
        detail = (result.stderr or result.stdout or "ollama stop failed").strip()
        return RuntimeActionResult(
            ok=False,
            status="error",
            message=detail,
        )


class LocalAIAdapter(GenericOpenAICompatibleAdapter):
    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
    ) -> None:
        super().__init__(
            backend,
            requester=requester,
            provider="localai",
            runtime_kind="localai",
        )

    def capabilities(self) -> RuntimeCapabilities:
        capabilities = super().capabilities()
        lifecycle_reason = (
            "LocalAI lifecycle and model loading are managed by the LocalAI server "
            "or its deployment supervisor."
        )
        return RuntimeCapabilities(
            provider=capabilities.provider,
            runtime_kind=capabilities.runtime_kind,
            endpoint_url=capabilities.endpoint_url,
            detect_runtime=capabilities.detect_runtime,
            health=capabilities.health,
            discover_models=capabilities.discover_models,
            list_loaded_models=AdapterSupport(
                False,
                "LocalAI OpenAI-compatible /models does not distinguish loaded models.",
            ),
            load_model=AdapterSupport(False, lifecycle_reason),
            unload_model=AdapterSupport(False, lifecycle_reason),
            start_server=AdapterSupport(False, lifecycle_reason),
            stop_server=AdapterSupport(False, lifecycle_reason),
            logs=capabilities.logs,
        )

    def detect(self) -> RuntimeDetection:
        return RuntimeDetection(
            provider="localai",
            runtime_kind="localai",
            endpoint_url=self.endpoint_url(),
            installed=None,
            available=True,
            detail=(
                "LocalAI-compatible endpoint configured; health check determines "
                "whether the server is running."
            ),
        )


class VLLMAdapter(GenericOpenAICompatibleAdapter):
    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
    ) -> None:
        super().__init__(
            backend,
            requester=requester,
            provider="vllm",
            runtime_kind="vllm",
        )

    def capabilities(self) -> RuntimeCapabilities:
        capabilities = super().capabilities()
        lifecycle_reason = (
            "vLLM lifecycle and model loading are managed by the vLLM server "
            "process or its deployment supervisor."
        )
        return RuntimeCapabilities(
            provider=capabilities.provider,
            runtime_kind=capabilities.runtime_kind,
            endpoint_url=capabilities.endpoint_url,
            detect_runtime=capabilities.detect_runtime,
            health=capabilities.health,
            discover_models=capabilities.discover_models,
            list_loaded_models=AdapterSupport(
                False,
                "vLLM OpenAI-compatible /models does not distinguish loaded models.",
            ),
            load_model=AdapterSupport(False, lifecycle_reason),
            unload_model=AdapterSupport(False, lifecycle_reason),
            start_server=AdapterSupport(False, lifecycle_reason),
            stop_server=AdapterSupport(False, lifecycle_reason),
            logs=capabilities.logs,
        )

    def detect(self) -> RuntimeDetection:
        return RuntimeDetection(
            provider="vllm",
            runtime_kind="vllm",
            endpoint_url=self.endpoint_url(),
            installed=None,
            available=True,
            detail=(
                "vLLM OpenAI-compatible endpoint configured; health check determines "
                "whether the server is running."
            ),
        )


class ManagedRuntimeAdapter(GenericOpenAICompatibleAdapter):
    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
    ) -> None:
        super().__init__(
            backend,
            requester=requester,
            provider=provider_for_backend(backend),
            runtime_kind=backend.runtime.kind,
        )

    def capabilities(self) -> RuntimeCapabilities:
        capabilities = super().capabilities()
        lifecycle_reason = (
            "Managed runtime lifecycle is executed by model-router-proxy for configured backends."
            if self.backend.runtime.command
            else "Managed runtime has no command configured."
        )
        return RuntimeCapabilities(
            provider=capabilities.provider,
            runtime_kind=capabilities.runtime_kind,
            endpoint_url=capabilities.endpoint_url,
            detect_runtime=capabilities.detect_runtime,
            health=capabilities.health,
            discover_models=capabilities.discover_models,
            list_loaded_models=AdapterSupport(
                False,
                "Managed runtime process state is owned by model-router-proxy.",
            ),
            load_model=AdapterSupport(
                False,
                "Managed runtimes load by starting their configured process.",
            ),
            unload_model=AdapterSupport(
                False,
                "Managed runtimes unload after idle timeout or proxy shutdown.",
            ),
            start_server=AdapterSupport(False, lifecycle_reason),
            stop_server=AdapterSupport(False, lifecycle_reason),
            logs=capabilities.logs,
        )

    def detect(self) -> RuntimeDetection:
        command = self.backend.runtime.command
        command_name = command[0] if command else ""
        installed = _command_available(command_name) if command_name else False
        return RuntimeDetection(
            provider=provider_for_backend(self.backend),
            runtime_kind=self.backend.runtime.kind,
            endpoint_url=self.endpoint_url(),
            installed=installed,
            available=installed if command_name else False,
            detail=(
                "Managed runtime command is available; model-router-proxy controls process start/stop."
                if installed
                else (
                    f"Managed runtime command missing: {command_name}"
                    if command_name
                    else "Managed runtime command is not configured."
                )
            ),
            command=tuple(command),
        )


def adapter_for_backend(
    backend: ProxyBackendConfig,
    *,
    requester: JsonRequester | None = None,
) -> RuntimeAdapter:
    if backend.runtime.enabled:
        return ManagedRuntimeAdapter(backend, requester=requester)
    provider = provider_for_backend(backend)
    if provider == "lmstudio":
        return LMStudioAdapter(backend, requester=requester)
    if provider == "ollama":
        return OllamaAdapter(backend, requester=requester)
    if provider == "localai":
        return LocalAIAdapter(backend, requester=requester)
    if provider == "vllm":
        return VLLMAdapter(backend, requester=requester)
    return GenericOpenAICompatibleAdapter(backend, requester=requester)


def runtime_state_for_backend(
    backend: ProxyBackendConfig,
    *,
    timeout_seconds: float = 0.25,
    requester: JsonRequester | None = None,
    checked_at: str | None = None,
) -> dict[str, Any]:
    last_checked_at = checked_at or _utc_timestamp()
    try:
        adapter = adapter_for_backend(backend, requester=requester)
    except Exception as exc:
        return _adapter_error_state(backend, exc, checked_at=last_checked_at)
    try:
        capabilities = adapter.capabilities()
    except Exception as exc:
        return _adapter_error_state(backend, exc, checked_at=last_checked_at)
    endpoint_url = _safe_endpoint_url(adapter, backend)
    try:
        detection = adapter.detect()
    except Exception as exc:
        detection = RuntimeDetection(
            provider=capabilities.provider,
            runtime_kind=capabilities.runtime_kind,
            endpoint_url=endpoint_url,
            installed=None,
            available=None,
            detail=exc.__class__.__name__,
        )
    try:
        health = adapter.health(timeout_seconds=timeout_seconds)
    except Exception as exc:
        health = RuntimeHealth(
            status="error",
            reachable=False,
            ok=False,
            detail=exc.__class__.__name__,
        )
    try:
        models = adapter.discover_models(timeout_seconds=timeout_seconds)
    except Exception:
        models = ()
    try:
        loaded = adapter.list_loaded_models(timeout_seconds=timeout_seconds)
    except Exception:
        loaded = ()
    try:
        logs = adapter.logs()
    except Exception as exc:
        logs = RuntimeLogInfo(supported=False, error=exc.__class__.__name__)
    detected = _runtime_detected(backend, detection, health)
    missing_dependency = _missing_dependency(backend, detection, capabilities)
    return {
        "adapter": adapter.__class__.__name__,
        "runtime_id": capabilities.provider,
        "provider": capabilities.provider,
        "runtime_kind": capabilities.runtime_kind,
        "runtime_mode": runtime_mode_for_backend(backend),
        "endpoint_url": endpoint_url,
        "endpoint": endpoint_url,
        "version": detection.version,
        "detected": detected,
        "last_checked_at": last_checked_at,
        "health_status": health.status,
        "missing_dependency": missing_dependency,
        "install_hint": _install_hint(
            backend,
            provider=capabilities.provider,
            detection=detection,
            health=health,
            missing_dependency=missing_dependency,
        ),
        "detection": detection.to_dict(),
        "health": health.to_dict(),
        "models": [model.to_dict() for model in models],
        "loaded_models": [model.to_dict() for model in loaded],
        "capabilities": capabilities.to_dict(),
        "logs": logs.to_dict(),
    }


def provider_for_backend(backend: ProxyBackendConfig) -> str:
    if backend.runtime.enabled:
        return {
            "llama-server": "llamacpp",
            "mlx-lm": "mlx_lm",
            "generic": "managed_openai_compatible",
        }.get(backend.runtime.kind, backend.runtime.kind)
    parsed = urlparse(backend.base_url)
    host = parsed.hostname or ""
    port = parsed.port
    fingerprint = " ".join(
        part.lower()
        for part in (backend.name, backend.model, host)
        if isinstance(part, str)
    )
    if _local_host(host) and port == 1234:
        return "lmstudio"
    if _local_host(host) and port == 11434:
        return "ollama"
    if "localai" in fingerprint:
        return "localai"
    if "vllm" in fingerprint:
        return "vllm"
    if _is_local_base_url(backend.base_url):
        return "openai_compatible_local"
    return "openai_compatible_hosted"


def runtime_kind_for_backend(backend: ProxyBackendConfig) -> str:
    if backend.runtime.enabled:
        return backend.runtime.kind
    provider = provider_for_backend(backend)
    return provider


def runtime_mode_for_backend(backend: ProxyBackendConfig) -> str:
    if backend.runtime.enabled:
        return "external_cli"
    return "external_managed"


def _adapter_error_state(
    backend: ProxyBackendConfig,
    exc: Exception,
    *,
    checked_at: str,
) -> dict[str, Any]:
    provider = provider_for_backend(backend)
    runtime_kind = runtime_kind_for_backend(backend)
    endpoint_url = backend.base_url.rstrip("/")
    health = RuntimeHealth(
        status="error",
        reachable=False,
        ok=False,
        detail=exc.__class__.__name__,
    )
    detection = RuntimeDetection(
        provider=provider,
        runtime_kind=runtime_kind,
        endpoint_url=endpoint_url,
        installed=None,
        available=None,
        detail=exc.__class__.__name__,
    )
    return {
        "adapter": "error",
        "runtime_id": provider,
        "provider": provider,
        "runtime_kind": runtime_kind,
        "runtime_mode": runtime_mode_for_backend(backend),
        "endpoint_url": endpoint_url,
        "endpoint": endpoint_url,
        "version": None,
        "detected": False,
        "last_checked_at": checked_at,
        "health_status": health.status,
        "missing_dependency": None,
        "install_hint": _install_hint(
            backend,
            provider=provider,
            detection=detection,
            health=health,
            missing_dependency=None,
        ),
        "detection": detection.to_dict(),
        "health": health.to_dict(),
        "models": [],
        "loaded_models": [],
        "capabilities": RuntimeCapabilities(
            provider=provider,
            runtime_kind=runtime_kind,
            health=AdapterSupport(False, "Adapter failed."),
            discover_models=AdapterSupport(False, "Adapter failed."),
            list_loaded_models=AdapterSupport(False, "Adapter failed."),
            load_model=AdapterSupport(False, "Adapter failed."),
            unload_model=AdapterSupport(False, "Adapter failed."),
            start_server=AdapterSupport(False, "Adapter failed."),
            stop_server=AdapterSupport(False, "Adapter failed."),
            logs=AdapterSupport(False, "Adapter failed."),
        ).to_dict(),
        "logs": RuntimeLogInfo(supported=False, error=exc.__class__.__name__).to_dict(),
    }


def _safe_endpoint_url(adapter: RuntimeAdapter, backend: ProxyBackendConfig) -> str:
    try:
        endpoint = adapter.endpoint_url()
    except Exception:
        endpoint = backend.base_url
    return endpoint.rstrip("/")


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _runtime_detected(
    backend: ProxyBackendConfig,
    detection: RuntimeDetection,
    health: RuntimeHealth,
) -> bool:
    if provider_for_backend(backend) == "openai_compatible_hosted":
        return True
    return (
        detection.installed is True
        or detection.available is True
        or health.reachable
        or bool(backend.base_url)
    )


def _missing_dependency(
    backend: ProxyBackendConfig,
    detection: RuntimeDetection,
    capabilities: RuntimeCapabilities,
) -> str | None:
    if backend.runtime.enabled and detection.installed is False and detection.command:
        return detection.command[0]
    if capabilities.provider == "ollama" and (
        not capabilities.list_loaded_models.supported
        or not capabilities.unload_model.supported
    ):
        return "ollama CLI"
    if capabilities.provider == "lmstudio" and detection.installed is False:
        return "lms CLI (optional)"
    return None


def _install_hint(
    backend: ProxyBackendConfig,
    *,
    provider: str,
    detection: RuntimeDetection,
    health: RuntimeHealth,
    missing_dependency: str | None,
) -> str | None:
    if provider == "openai_compatible_hosted":
        return "Hosted backend is configured; ModelRouter does not probe provider availability by default."
    if provider == "lmstudio":
        if not health.reachable:
            return (
                "Start the LM Studio local server at "
                f"{backend.base_url.rstrip('/')} and use exact model ids it lists."
            )
        if missing_dependency:
            return "Install the LM Studio CLI only if native lifecycle commands are needed."
        return None
    if provider == "ollama":
        if not health.reachable:
            return "Start Ollama with `ollama serve` or update the backend base_url."
        if missing_dependency:
            return "Install the Ollama CLI for loaded-model and explicit unload actions."
        return None
    if provider == "llamacpp":
        if missing_dependency:
            return "Install llama.cpp so `llama-server` is on PATH, or update runtime.command."
        if not health.reachable:
            return "Start the configured llama.cpp server or let the proxy manage it when enabled."
        return None
    if provider == "mlx_lm":
        if missing_dependency:
            return "Install MLX-LM so `mlx_lm.server` is on PATH, or update runtime.command."
        if not health.reachable:
            return "Start the configured MLX-LM server or let the proxy manage it when enabled."
        return None
    if provider in {"localai", "vllm", "openai_compatible_local"}:
        if not health.reachable:
            return "Start the configured local OpenAI-compatible server or update backend base_url."
        return None
    if detection.installed is False and detection.command:
        return f"Install `{detection.command[0]}` or update runtime.command."
    if not health.reachable and _is_local_base_url(backend.base_url):
        return "Start the configured local runtime or update backend base_url."
    return None


def _request_json(
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, Any]:
    request = Request(url, headers=dict(headers), method="GET")
    with urlopen(request, timeout=timeout_seconds) as response:
        import json

        raw = response.read()
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        return int(getattr(response, "status", 200)), payload


def _models_url(backend: ProxyBackendConfig) -> str:
    return urljoin(backend.base_url.rstrip("/") + "/", "models")


def _auth_headers(backend: ProxyBackendConfig) -> dict[str, str]:
    api_key = backend.resolved_api_key
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _models_from_payload(payload: Any) -> tuple[RuntimeModel, ...]:
    if not isinstance(payload, Mapping):
        return ()
    data = payload.get("data")
    if not isinstance(data, list):
        return ()
    models: list[RuntimeModel] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            models.append(RuntimeModel(model_id=model_id.strip(), loaded=None))
    return tuple(models)


def _models_from_ollama_table(output: str, *, loaded: bool) -> tuple[RuntimeModel, ...]:
    models: list[RuntimeModel] = []
    seen: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped.split(maxsplit=1)[0]
        if first.upper() in {"NAME", "MODEL"}:
            continue
        if first in seen:
            continue
        seen.add(first)
        models.append(
            RuntimeModel(
                model_id=first,
                loaded=loaded,
                source="ollama_cli",
            )
        )
    return tuple(models)


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return _local_host(parsed.hostname or "")


def _local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _command_available(command: str) -> bool:
    return _resolve_command(command) is not None


def _resolve_command(command: str) -> str | None:
    if not command:
        return None
    expanded = Path(command).expanduser()
    if expanded.is_absolute() or "/" in command:
        return str(expanded) if expanded.exists() else None
    return shutil.which(command)


def _run_command(command: tuple[str, ...], timeout_seconds: float) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return CommandResult(127, stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "command timed out",
        )
    except OSError as exc:
        return CommandResult(1, stderr=f"{exc.__class__.__name__}: {exc}")
    return CommandResult(
        completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
