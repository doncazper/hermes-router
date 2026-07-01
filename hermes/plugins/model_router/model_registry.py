"""Local-first known-model registry for ModelRouter control surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from hermes.plugins.model_router.models import RouterConfig
from hermes.plugins.model_router.proxy_config import RoutingProxyConfig
from hermes.plugins.model_router.runtime_adapters import (
    RuntimeModel,
    provider_for_backend,
    runtime_kind_for_backend,
)


@dataclass(frozen=True)
class KnownModel:
    """JSON-safe record for a model ModelRouter knows about."""

    provider: str
    runtime: str
    model_id: str
    name: str | None = None
    source: str = "unknown"
    local_path: str | None = None
    format: str | None = None
    context_length: int | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    license: str | None = None
    install_state: str = "unknown"
    health_state: str = "unknown"
    load_state: str = "unknown"
    tags: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    routing_eligible: bool = True
    backend: str | None = None
    assigned_routes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "runtime": self.runtime,
            "model_id": self.model_id,
            "name": self.name,
            "source": self.source,
            "local_path": self.local_path,
            "format": self.format,
            "context_length": self.context_length,
            "quantization": self.quantization,
            "size_bytes": self.size_bytes,
            "license": self.license,
            "install_state": self.install_state,
            "health_state": self.health_state,
            "load_state": self.load_state,
            "tags": list(self.tags),
            "capabilities": list(self.capabilities),
            "routing_eligible": self.routing_eligible,
            "backend": self.backend,
            "assigned_routes": list(self.assigned_routes),
            "metadata": _json_safe_mapping(self.metadata),
        }


@dataclass(frozen=True)
class ModelRegistry:
    """Collection of known models from config, discovery, and adapters."""

    models: tuple[KnownModel, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [model.to_dict() for model in self.models],
            "count": len(self.models),
            "sources": sorted({model.source for model in self.models}),
        }

    def by_model_id(self, model_id: str) -> tuple[KnownModel, ...]:
        return tuple(model for model in self.models if model.model_id == model_id)


def build_model_registry(
    *,
    router_config: RouterConfig | None = None,
    proxy_config: RoutingProxyConfig | None = None,
    discovery: Any = None,
    runtime_models: Mapping[str, Sequence[Any]] | None = None,
    user_models: Sequence[Mapping[str, Any]] = (),
) -> ModelRegistry:
    """Build a deterministic local registry without network calls.

    ``runtime_models`` is an explicit input from caller-owned adapter code. This
    helper never discovers runtime models by itself, so routing and proxy hot
    paths do not gain new network or filesystem dependencies.
    """

    rows: list[KnownModel] = []
    if router_config is not None:
        rows.extend(_models_from_router_config(router_config))
    if proxy_config is not None:
        rows.extend(_models_from_proxy_config(proxy_config))
    if discovery is not None:
        rows.extend(_models_from_discovery(discovery))
    if runtime_models:
        rows.extend(_models_from_runtime(proxy_config, runtime_models))
    rows.extend(_models_from_user_models(user_models))
    return ModelRegistry(models=_merge_models(rows))


def _models_from_router_config(config: RouterConfig) -> tuple[KnownModel, ...]:
    assigned_by_engine: dict[str, list[str]] = {}
    for route_id, engine_name in config.routing_targets.items():
        assigned_by_engine.setdefault(engine_name, []).append(route_id)

    rows: list[KnownModel] = []
    for engine in config.engines.values():
        model_id = _clean_string(engine.model)
        if not model_id:
            continue
        capabilities = [*engine.modalities]
        if engine.supports_tools:
            capabilities.append("tools")
        rows.append(
            KnownModel(
                provider=_clean_string(engine.provider) or "unknown",
                runtime=_clean_string(engine.adapter) or "unknown",
                model_id=model_id,
                name=model_id,
                source="router_config",
                format=_format_from_values(engine.adapter, model_id),
                context_length=engine.max_context,
                install_state="configured",
                health_state="unknown",
                load_state="unknown",
                tags=_unique_strings((*engine.strengths, engine.name)),
                capabilities=_unique_strings(capabilities),
                routing_eligible=engine.enabled,
                assigned_routes=tuple(sorted(assigned_by_engine.get(engine.name, ()))),
                metadata={
                    "engine": engine.name,
                    "cost_tier": engine.cost_tier,
                    "latency_tier": engine.latency_tier,
                    "capability_tier": engine.capability_tier,
                    "trust_tier": engine.trust_tier,
                },
            )
        )
    return tuple(rows)


def _models_from_proxy_config(config: RoutingProxyConfig) -> tuple[KnownModel, ...]:
    routes_by_backend: dict[str, list[str]] = {}
    for route_id, backend_name in config.engine_backends.items():
        routes_by_backend.setdefault(backend_name, []).append(route_id)

    rows: list[KnownModel] = []
    for backend in config.backends.values():
        model_id = _clean_string(backend.model)
        if not model_id:
            continue
        provider = provider_for_backend(backend)
        runtime = runtime_kind_for_backend(backend)
        rejection = config.backend_policy_rejection_reason(backend.name)
        rows.append(
            KnownModel(
                provider=provider,
                runtime=runtime,
                model_id=model_id,
                name=model_id,
                source="proxy_config",
                format=_format_from_values(runtime, model_id, backend.base_url),
                install_state=_install_state_for_backend(provider),
                health_state="unknown",
                load_state="configured",
                tags=_unique_strings((backend.name, provider, runtime)),
                capabilities=_backend_capabilities(backend),
                routing_eligible=rejection is None,
                backend=backend.name,
                assigned_routes=tuple(sorted(routes_by_backend.get(backend.name, ()))),
                metadata={
                    "base_url": backend.base_url,
                    "runtime_enabled": backend.runtime.enabled,
                    "policy_status": rejection or "allowed",
                },
            )
        )
    return tuple(rows)


def _models_from_discovery(discovery: Any) -> tuple[KnownModel, ...]:
    rows: list[KnownModel] = []
    for model in getattr(discovery, "models", ()) or ():
        model_id = _clean_string(getattr(model, "repo_id", None))
        if not model_id:
            continue
        path = _clean_string(getattr(model, "path", None))
        source = _clean_string(getattr(model, "source", None)) or "local_discovery"
        name = _clean_string(getattr(model, "name", None)) or model_id
        roles = _unique_strings(getattr(model, "roles", ()) or ())
        provider = _provider_from_discovered(source, path, model_id)
        runtime = _runtime_from_discovered(source, path, model_id)
        rows.append(
            KnownModel(
                provider=provider,
                runtime=runtime,
                model_id=model_id,
                name=name,
                source=source,
                local_path=path or None,
                format=_format_from_values(path, source, model_id),
                quantization=_quantization_from_values(path, model_id),
                size_bytes=_path_size(path),
                install_state="installed",
                health_state="unknown",
                load_state="unknown",
                tags=roles,
                capabilities=_capabilities_from_roles(roles),
                routing_eligible=True,
                assigned_routes=roles,
            )
        )
    return tuple(rows)


def _models_from_runtime(
    config: RoutingProxyConfig | None,
    runtime_models: Mapping[str, Sequence[Any]],
) -> tuple[KnownModel, ...]:
    rows: list[KnownModel] = []
    for backend_name, models in sorted(runtime_models.items()):
        backend = config.backends.get(backend_name) if config is not None else None
        provider = provider_for_backend(backend) if backend is not None else "unknown"
        runtime = runtime_kind_for_backend(backend) if backend is not None else "unknown"
        for item in models:
            runtime_model = _coerce_runtime_model(item)
            if runtime_model is None:
                continue
            load_state = (
                "loaded"
                if runtime_model.loaded is True
                else ("unloaded" if runtime_model.loaded is False else "unknown")
            )
            rows.append(
                KnownModel(
                    provider=provider,
                    runtime=runtime,
                    model_id=runtime_model.model_id,
                    name=runtime_model.model_id,
                    source=runtime_model.source or "runtime_discovery",
                    format=_format_from_values(runtime, runtime_model.model_id),
                    install_state="available",
                    health_state="unknown",
                    load_state=load_state,
                    tags=_unique_strings((backend_name, provider, runtime)),
                    capabilities=("chat",),
                    routing_eligible=True,
                    backend=backend_name,
                    assigned_routes=_routes_for_backend(config, backend_name),
                )
            )
    return tuple(rows)


def _models_from_user_models(
    user_models: Sequence[Mapping[str, Any]],
) -> tuple[KnownModel, ...]:
    rows: list[KnownModel] = []
    for item in user_models:
        model_id = _clean_string(item.get("model_id") or item.get("model"))
        if not model_id:
            continue
        provider = _clean_string(item.get("provider")) or "user_declared"
        runtime = _clean_string(item.get("runtime")) or _runtime_for_provider(provider)
        local_path = _clean_string(item.get("local_path") or item.get("path"))
        install_state = _clean_string(item.get("install_state")) or (
            "installed" if local_path else "remote"
        )
        rows.append(
            KnownModel(
                provider=provider,
                runtime=runtime,
                model_id=model_id,
                name=_clean_string(item.get("name")) or model_id,
                source=_clean_string(item.get("source")) or "user_declared",
                local_path=local_path or None,
                format=_clean_string(item.get("format"))
                or _format_from_values(local_path, runtime, model_id),
                context_length=_optional_int(item.get("context_length")),
                quantization=_clean_string(item.get("quantization"))
                or _quantization_from_values(local_path, model_id),
                size_bytes=_optional_int(item.get("size_bytes")),
                license=_clean_string(item.get("license")) or None,
                install_state=install_state,
                health_state=_clean_string(item.get("health_state")) or "unknown",
                load_state=_clean_string(item.get("load_state")) or "unknown",
                tags=_unique_strings(item.get("tags", ())),
                capabilities=_unique_strings(item.get("capabilities", ())),
                routing_eligible=bool(item.get("routing_eligible", True)),
                backend=_clean_string(item.get("backend")) or None,
                assigned_routes=_unique_strings(item.get("assigned_routes", ())),
                metadata=_metadata_from_user_model(item),
            )
        )
    return tuple(rows)


def _merge_models(rows: Sequence[KnownModel]) -> tuple[KnownModel, ...]:
    merged: dict[tuple[str, str, str, str, str], KnownModel] = {}
    for row in rows:
        key = _merge_key(row)
        existing = merged.get(key)
        merged[key] = row if existing is None else _merge_model(existing, row)
    return tuple(sorted(merged.values(), key=_sort_key))


def _merge_model(left: KnownModel, right: KnownModel) -> KnownModel:
    return KnownModel(
        provider=_prefer(left.provider, right.provider, default="unknown") or "unknown",
        runtime=_prefer(left.runtime, right.runtime, default="unknown") or "unknown",
        model_id=left.model_id,
        name=_prefer(left.name, right.name),
        source="+".join(_unique_strings((left.source, right.source))),
        local_path=_prefer(left.local_path, right.local_path),
        format=_prefer(left.format, right.format),
        context_length=left.context_length or right.context_length,
        quantization=_prefer(left.quantization, right.quantization),
        size_bytes=left.size_bytes or right.size_bytes,
        license=_prefer(left.license, right.license),
        install_state=_state_preference(left.install_state, right.install_state),
        health_state=_state_preference(left.health_state, right.health_state),
        load_state=_state_preference(left.load_state, right.load_state),
        tags=_unique_strings((*left.tags, *right.tags)),
        capabilities=_unique_strings((*left.capabilities, *right.capabilities)),
        routing_eligible=left.routing_eligible or right.routing_eligible,
        backend=_prefer(left.backend, right.backend),
        assigned_routes=_unique_strings((*left.assigned_routes, *right.assigned_routes)),
        metadata={**_json_safe_mapping(left.metadata), **_json_safe_mapping(right.metadata)},
    )


def _merge_key(model: KnownModel) -> tuple[str, str, str, str, str]:
    if model.local_path:
        return ("path", model.local_path, model.model_id, "", "")
    if model.backend:
        return ("backend", model.provider, model.runtime, model.backend, model.model_id)
    return ("model", model.provider, model.runtime, model.model_id, "")


def _sort_key(model: KnownModel) -> tuple[str, str, str, str]:
    return (model.provider, model.runtime, model.backend or "", model.model_id)


def _routes_for_backend(
    config: RoutingProxyConfig | None,
    backend_name: str,
) -> tuple[str, ...]:
    if config is None:
        return ()
    return tuple(
        sorted(
            route_id
            for route_id, candidate_backend in config.engine_backends.items()
            if candidate_backend == backend_name
        )
    )


def _coerce_runtime_model(item: Any) -> RuntimeModel | None:
    if isinstance(item, RuntimeModel):
        return item if item.model_id else None
    if isinstance(item, str):
        model_id = item.strip()
        return RuntimeModel(model_id=model_id) if model_id else None
    if isinstance(item, Mapping):
        model_id = _clean_string(item.get("model_id") or item.get("id"))
        if not model_id:
            return None
        loaded = item.get("loaded")
        return RuntimeModel(
            model_id=model_id,
            loaded=loaded if isinstance(loaded, bool) else None,
            source=_clean_string(item.get("source")) or "runtime_discovery",
        )
    return None


def _provider_from_discovered(source: str, path: str, model_id: str) -> str:
    text = f"{source} {path} {model_id}".lower()
    if "lmstudio" in text or "lm studio" in text or ".lmstudio" in text:
        return "lmstudio"
    if "ollama" in text:
        return "ollama"
    if "huggingface" in text:
        return "huggingface"
    if "mlx" in text:
        return "mlx_lm"
    if "localai" in text:
        return "localai"
    return "local"


def _runtime_from_discovered(source: str, path: str, model_id: str) -> str:
    text = f"{source} {path} {model_id}".lower()
    if "lmstudio" in text or "lm studio" in text or ".lmstudio" in text:
        return "lmstudio"
    if "ollama" in text:
        return "ollama"
    if "mlx" in text:
        return "mlx-lm"
    if ".gguf" in text or "gguf" in text:
        return "llama.cpp"
    if "localai" in text:
        return "localai"
    return "local"


def _runtime_for_provider(provider: str) -> str:
    if provider in {"openai", "anthropic", "hosted", "openai_compatible_hosted"}:
        return "api"
    return provider or "unknown"


def _install_state_for_backend(provider: str) -> str:
    if provider.endswith("_hosted") or provider in {"openai", "anthropic"}:
        return "remote"
    return "configured"


def _backend_capabilities(backend: Any) -> tuple[str, ...]:
    capabilities = ["chat"]
    if not getattr(backend, "strip_tools", False):
        capabilities.append("tools_passthrough")
    return tuple(capabilities)


def _capabilities_from_roles(roles: Sequence[str]) -> tuple[str, ...]:
    capabilities = ["chat"]
    if any("vision" in role for role in roles):
        capabilities.append("vision")
    if any("image" in role for role in roles):
        capabilities.append("image_generation")
    if any("code" in role for role in roles):
        capabilities.append("code")
    return tuple(capabilities)


def _format_from_values(*values: str | None) -> str | None:
    text = " ".join(value for value in values if value).lower()
    if ".gguf" in text or "gguf" in text:
        return "GGUF"
    if ".safetensors" in text or "safetensors" in text:
        return "safetensors"
    if ".onnx" in text:
        return "ONNX"
    if ".bin" in text:
        return "bin"
    if "mlx" in text:
        return "MLX"
    if "api" in text or "openai" in text or "anthropic" in text:
        return "API"
    return None


def _quantization_from_values(*values: str | None) -> str | None:
    text = " ".join(value for value in values if value)
    for pattern in (r"q[2-8]_[a-z0-9_]+", r"q[2-8]\b", r"[2348]bit"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _path_size(path: str) -> int | None:
    if not path:
        return None
    expanded = Path(path).expanduser()
    try:
        if expanded.is_file():
            return expanded.stat().st_size
    except OSError:
        return None
    return None


def _metadata_from_user_model(item: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {
        "provider",
        "runtime",
        "model_id",
        "model",
        "name",
        "source",
        "local_path",
        "path",
        "format",
        "context_length",
        "quantization",
        "size_bytes",
        "license",
        "install_state",
        "health_state",
        "load_state",
        "tags",
        "capabilities",
        "routing_eligible",
        "backend",
        "assigned_routes",
    }
    return _json_safe_mapping({key: value for key, value in item.items() if key not in reserved})


def _clean_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        iterable: Sequence[Any] = (values,)
    elif isinstance(values, Sequence):
        iterable = values
    else:
        iterable = (values,)
    cleaned = [_clean_string(value) for value in iterable]
    return tuple(dict.fromkeys(value for value in cleaned if value))


def _prefer(left: str | None, right: str | None, *, default: str | None = None) -> str | None:
    return left or right or default


def _state_preference(left: str, right: str) -> str:
    order = {
        "installed": 6,
        "available": 5,
        "loaded": 5,
        "configured": 4,
        "remote": 3,
        "unloaded": 2,
        "unknown": 1,
    }
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in mapping.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
