"""Runtime adapter foundation for ModelRouter admin surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from hermes.plugins.model_router.proxy_config import ProxyBackendConfig


JsonRequester = Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "runtime_kind": self.runtime_kind,
            "health": self.health.to_dict(),
            "discover_models": self.discover_models.to_dict(),
            "list_loaded_models": self.list_loaded_models.to_dict(),
            "load_model": self.load_model.to_dict(),
            "unload_model": self.unload_model.to_dict(),
            "logs": self.logs.to_dict(),
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

    def capabilities(self) -> RuntimeCapabilities:
        """Return supported runtime actions and disabled reasons."""

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

    def logs(self) -> RuntimeLogInfo:
        """Return safe runtime log metadata."""


class GenericOpenAICompatibleAdapter:
    """Adapter for local OpenAI-compatible `/v1/models` runtimes."""

    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
        provider: str | None = None,
        runtime_kind: str | None = None,
    ) -> None:
        self.backend = backend
        self._requester = requester or _request_json
        self._provider = provider or provider_for_backend(backend)
        self._runtime_kind = runtime_kind or runtime_kind_for_backend(backend)

    def capabilities(self) -> RuntimeCapabilities:
        local = _is_local_base_url(self.backend.base_url)
        health_reason = None if local else "Hosted runtime health is not checked by settings."
        return RuntimeCapabilities(
            provider=self._provider,
            runtime_kind=self._runtime_kind,
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
            logs=AdapterSupport(
                self.backend.runtime.enabled and bool(self.backend.runtime.log_path),
                None
                if self.backend.runtime.enabled and self.backend.runtime.log_path
                else "No managed runtime log path is configured.",
            ),
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
    ) -> None:
        super().__init__(
            backend,
            requester=requester,
            provider="lmstudio",
            runtime_kind="lmstudio",
        )


class OllamaAdapter(GenericOpenAICompatibleAdapter):
    def __init__(
        self,
        backend: ProxyBackendConfig,
        *,
        requester: JsonRequester | None = None,
    ) -> None:
        super().__init__(
            backend,
            requester=requester,
            provider="ollama",
            runtime_kind="ollama",
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
        return RuntimeCapabilities(
            provider=capabilities.provider,
            runtime_kind=capabilities.runtime_kind,
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
            logs=capabilities.logs,
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
    return GenericOpenAICompatibleAdapter(backend, requester=requester)


def runtime_state_for_backend(
    backend: ProxyBackendConfig,
    *,
    timeout_seconds: float = 0.25,
    requester: JsonRequester | None = None,
) -> dict[str, Any]:
    try:
        adapter = adapter_for_backend(backend, requester=requester)
    except Exception as exc:
        return _adapter_error_state(backend, exc)
    try:
        capabilities = adapter.capabilities()
    except Exception as exc:
        return _adapter_error_state(backend, exc)
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
    return {
        "adapter": adapter.__class__.__name__,
        "provider": capabilities.provider,
        "runtime_kind": capabilities.runtime_kind,
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
    if _local_host(host) and port == 1234:
        return "lmstudio"
    if _local_host(host) and port == 11434:
        return "ollama"
    if _is_local_base_url(backend.base_url):
        return "openai_compatible_local"
    return "openai_compatible_hosted"


def runtime_kind_for_backend(backend: ProxyBackendConfig) -> str:
    if backend.runtime.enabled:
        return backend.runtime.kind
    provider = provider_for_backend(backend)
    return provider


def _adapter_error_state(
    backend: ProxyBackendConfig,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "adapter": "error",
        "provider": provider_for_backend(backend),
        "runtime_kind": runtime_kind_for_backend(backend),
        "health": RuntimeHealth(
            status="error",
            reachable=False,
            ok=False,
            detail=exc.__class__.__name__,
        ).to_dict(),
        "models": [],
        "loaded_models": [],
        "capabilities": RuntimeCapabilities(
            provider=provider_for_backend(backend),
            runtime_kind=runtime_kind_for_backend(backend),
            health=AdapterSupport(False, "Adapter failed."),
            discover_models=AdapterSupport(False, "Adapter failed."),
            list_loaded_models=AdapterSupport(False, "Adapter failed."),
            load_model=AdapterSupport(False, "Adapter failed."),
            unload_model=AdapterSupport(False, "Adapter failed."),
            logs=AdapterSupport(False, "Adapter failed."),
        ).to_dict(),
        "logs": RuntimeLogInfo(supported=False, error=exc.__class__.__name__).to_dict(),
    }


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


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return _local_host(parsed.hostname or "")


def _local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
