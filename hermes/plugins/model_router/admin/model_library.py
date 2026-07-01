"""Shared model-library state for settings UI, future TUI, and admin API."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from hermes.plugins.model_router.model_advisor import (
    CatalogModel,
    load_model_catalog,
    score_catalog_model,
)
from hermes.plugins.model_router.model_registry import build_model_registry
from hermes.plugins.model_router.proxy_config import RoutingProxyConfig


def build_model_library_state(
    *,
    paths: Mapping[str, Path],
    config: RoutingProxyConfig | None,
    discovery: Any,
    recommendation: Any,
    download_plan: Any,
    benchmark_results: Any = None,
    eval_results: Any = None,
    runtime_models: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a real model-library state block from scan/catalog/config data."""

    local_scores = _local_scores_by_repo(recommendation)
    return {
        "registry": build_model_registry(
            proxy_config=config,
            discovery=discovery,
            runtime_models=runtime_models,
            eval_results=_eval_rows(eval_results),
        ).to_dict(),
        "installed": _installed_models(config, discovery, local_scores),
        "discover": _discover_state(
            recommendation=recommendation,
            benchmark_results=benchmark_results,
        ),
        "recommended": _recommended_results(recommendation),
        "downloads": _download_states(paths, download_plan),
        "assignments": _assignment_rows(config),
    }


def _installed_models(
    config: RoutingProxyConfig | None,
    discovery: Any,
    local_scores: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in getattr(discovery, "models", ()):
        repo_id = str(getattr(model, "repo_id", "") or "").strip()
        if not repo_id:
            continue
        path = str(getattr(model, "path", "") or "")
        source = _source_label(str(getattr(model, "source", "") or ""), path, repo_id)
        rows.append(
            {
                "model_id": repo_id,
                "display_name": str(getattr(model, "name", "") or repo_id),
                "path": path or None,
                "source": source,
                "runtime_compatibility": _runtime_compatibility(source, path, repo_id),
                "roles": list(getattr(model, "roles", ()) or ()),
                "size_bytes": None,
                "quantization": _quantization(repo_id, path),
                "context_length": None,
                "license": None,
                "local": True,
                "loaded": _loaded_state(config, repo_id),
                "assigned_routes": _assigned_routes(config, repo_id),
                "score": local_scores.get(repo_id),
                "warnings": _score_warnings(local_scores.get(repo_id)),
            }
        )
    return rows


def _discover_state(
    *,
    recommendation: Any,
    benchmark_results: Any,
) -> dict[str, Any]:
    hardware = getattr(recommendation, "hardware_profile", None)
    results: list[dict[str, Any]] = []
    error: str | None = None
    try:
        catalog = load_model_catalog()
        for model in catalog.models:
            score = (
                score_catalog_model(
                    model,
                    hardware=hardware,
                    benchmark_results=benchmark_results,
                ).to_dict()
                if hardware is not None
                else None
            )
            results.append(_catalog_result(model, score=score))
    except Exception as exc:  # pragma: no cover - defensive local file guard.
        error = str(exc)
    return {
        "query": "",
        "source": "curated_catalog",
        "filters": {
            "route": None,
            "runtime_kind": None,
            "quantization": None,
            "max_size_gb": None,
            "license": None,
            "local_only": False,
        },
        "results": results,
        "error": error,
        "external_search_enabled": False,
        "external_search_maturity": "planned",
    }


def _recommended_results(recommendation: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suggestion in getattr(recommendation, "download_suggestions", ()):
        rows.append(
            {
                "model_id": suggestion.repo_id,
                "display_name": suggestion.repo_id,
                "provider": suggestion.provider,
                "route_fit": [suggestion.route],
                "runtime_kind": _runtime_kind_from_adapter(
                    suggestion.adapter,
                    suggestion.repo_id,
                ),
                "estimated_size_gb": None,
                "min_memory_gb": None,
                "recommended_memory_gb": None,
                "license": None,
                "download_supported": True,
                "score_label": _score_value(suggestion, "label"),
                "score_reasons": _score_list(suggestion, "reasons"),
                "warnings": _score_list(suggestion, "warnings"),
                "score": suggestion.score.to_dict() if suggestion.score else None,
                "reason": suggestion.reason,
                "command": list(suggestion.command),
            }
        )
    return rows


def _download_states(
    paths: Mapping[str, Path],
    download_plan: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, suggestion in enumerate(getattr(download_plan, "suggestions", ())):
        local_dir = _local_dir_from_command(suggestion.command) or str(
            paths["models"] / suggestion.route / _repo_slug(suggestion.repo_id)
        )
        rows.append(
            {
                "download_id": f"{suggestion.route}:{suggestion.repo_id}",
                "model_id": suggestion.repo_id,
                "route": suggestion.route,
                "status": "planned",
                "command": list(suggestion.command),
                "local_dir": local_dir,
                "progress_percent": None,
                "bytes_downloaded": None,
                "total_bytes": None,
                "error": None,
                "retryable": True,
                "provider": suggestion.provider,
                "adapter": suggestion.adapter,
                "score": suggestion.score.to_dict() if suggestion.score else None,
                "position": index + 1,
            }
        )
    return rows


def _assignment_rows(config: RoutingProxyConfig | None) -> list[dict[str, Any]]:
    if config is None:
        return []
    rows: list[dict[str, Any]] = []
    for route_id, backend_name in sorted(config.engine_backends.items()):
        backend = config.backends.get(backend_name)
        rows.append(
            {
                "route_id": route_id,
                "route_class": route_id.replace("_", " ").title(),
                "target_engine": route_id,
                "backend": backend_name,
                "model": backend.model if backend else None,
                "provider": _provider_from_backend(backend),
                "runtime_kind": _runtime_kind_from_backend(backend),
                "latency_tier": "unknown",
                "cost_tier": "unknown",
                "privacy": "unknown",
                "tools": "unknown",
                "fallback_chain": list(config.fallback_backends.get(backend_name, ())),
                "enabled": backend is not None,
                "available": None,
                "last_selected_at": None,
                "restart_recommended": True,
                "save_action": "model.assign_route",
            }
        )
    return rows


def _catalog_result(model: CatalogModel, *, score: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "model_id": model.repo_id,
        "display_name": model.repo_id,
        "provider": model.provider,
        "route_fit": [model.route],
        "runtime_kind": model.runtime_kind,
        "estimated_size_gb": None,
        "min_memory_gb": model.min_memory_gb,
        "recommended_memory_gb": model.recommended_memory_gb,
        "license": None,
        "download_supported": model.provider == "huggingface",
        "score_label": score.get("label") if score else None,
        "score_reasons": list(score.get("reasons", ())) if score else [],
        "warnings": list(score.get("warnings", ())) if score else [],
        "score": score,
        "reason": model.reason,
        "include": list(model.include),
        "adapter": model.adapter,
        "quantization": model.quantization,
    }


def _local_scores_by_repo(recommendation: Any) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for item in getattr(recommendation, "local_model_recommendations", ()):
        repo_id = str(getattr(item, "repo_id", "") or "")
        score = getattr(item, "score", None)
        if repo_id and score is not None:
            scores[repo_id] = score.to_dict()
    return scores


def _eval_rows(eval_results: Any) -> tuple[Mapping[str, Any], ...] | None:
    if eval_results is None:
        return None
    if isinstance(eval_results, Mapping):
        rows = eval_results.get("results")
        if isinstance(rows, list):
            return tuple(row for row in rows if isinstance(row, Mapping))
        return (eval_results,)
    if isinstance(eval_results, (list, tuple)):
        return tuple(row for row in eval_results if isinstance(row, Mapping))
    return None


def _assigned_routes(config: RoutingProxyConfig | None, model_id: str) -> list[str]:
    if config is None:
        return []
    routes: list[str] = []
    for route_id, backend_name in sorted(config.engine_backends.items()):
        backend = config.backends.get(backend_name)
        if backend and backend.model == model_id:
            routes.append(route_id)
    return routes


def _loaded_state(config: RoutingProxyConfig | None, model_id: str) -> bool | None:
    if config is None:
        return None
    for backend in config.backends.values():
        if backend.model == model_id and backend.runtime.enabled:
            return None
    return None


def _source_label(source: str, path: str, repo_id: str) -> str:
    lowered = f"{source} {path} {repo_id}".lower()
    if "ollama" in lowered:
        return "ollama"
    if "lmstudio" in lowered or "lm studio" in lowered:
        return "lmstudio"
    if "mlx" in lowered:
        return "mlx_lm"
    if ".gguf" in lowered:
        return "llamacpp"
    if "huggingface" in lowered or "cache/huggingface" in lowered:
        return "huggingface_cache"
    if source in {"local_directory", "modelrouter"}:
        return "modelrouter"
    return "unknown"


def _runtime_compatibility(source: str, path: str, repo_id: str) -> list[str]:
    text = f"{source} {path} {repo_id}".lower()
    values: list[str] = []
    if source == "mlx_lm" or "mlx" in text:
        values.append("mlx-lm")
    if source == "llamacpp" or ".gguf" in text or "gguf" in text:
        values.append("llama.cpp")
    if source == "ollama":
        values.append("ollama")
    if source == "lmstudio":
        values.append("lmstudio")
    if not values:
        values.append("openai-compatible")
    return values


def _runtime_kind_from_adapter(adapter: str, repo_id: str) -> str | None:
    lowered = f"{adapter} {repo_id}".lower()
    if "mlx" in lowered:
        return "mlx-lm"
    if "gguf" in lowered:
        return "llama.cpp"
    if "local" in lowered:
        return "openai-compatible"
    return None


def _runtime_kind_from_backend(backend: Any) -> str | None:
    if backend is None:
        return None
    runtime = getattr(backend, "runtime", None)
    if runtime is not None and getattr(runtime, "enabled", False):
        kind = getattr(runtime, "kind", None)
        if kind:
            return str(kind)
    return _provider_from_backend(backend)


def _provider_from_backend(backend: Any) -> str | None:
    if backend is None:
        return None
    base_url = str(getattr(backend, "base_url", "") or "").lower()
    model = str(getattr(backend, "model", "") or "").lower()
    if "11434" in base_url:
        return "ollama"
    if "1234" in base_url:
        return "lmstudio"
    if "mlx" in model:
        return "mlx_lm"
    if "gguf" in model:
        return "llamacpp"
    return "openai_compatible"


def _quantization(*values: str) -> str | None:
    text = " ".join(values)
    patterns = (
        r"q[2-8]_[a-z0-9_]+",
        r"q[2-8]\b",
        r"[2348]bit",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _score_value(item: Any, key: str) -> Any:
    score = getattr(item, "score", None)
    if score is None:
        return None
    return score.to_dict().get(key)


def _score_list(item: Any, key: str) -> list[str]:
    value = _score_value(item, key)
    return [str(item) for item in value] if isinstance(value, list) else []


def _score_warnings(score: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(score, Mapping):
        return []
    warnings = score.get("warnings")
    return [str(item) for item in warnings] if isinstance(warnings, list) else []


def _local_dir_from_command(command: tuple[str, ...]) -> str | None:
    parts = list(command)
    try:
        index = parts.index("--local-dir")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")
