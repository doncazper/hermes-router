"""Shared admin action dispatcher for settings UI, future TUI, and API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from hermes.plugins.model_router.admin.config_edit import save_proxy_config_patch
from hermes.plugins.model_router.admin.supervisor import ProxyProcessSupervisor
from hermes.plugins.model_router.catalog_update import (
    apply_catalog_update,
    catalog_diff,
    catalog_status,
)
from hermes.plugins.model_router.model_benchmark import (
    BenchmarkResult,
    BenchmarkTarget,
    execute_benchmark_plan,
    load_benchmark_results,
    plan_backend_benchmarks,
)
from hermes.plugins.model_router.pricing_catalog import (
    apply_pricing_catalog,
    pricing_diff,
    pricing_status,
)
from hermes.plugins.model_router.product import doctor_proxy_config as _doctor_proxy_config
from hermes.plugins.model_router.proxy_config import ProxyConfigError, load_proxy_config
from hermes.plugins.model_router.routing_log import RoutingLogWriter, build_feedback
from hermes.plugins.model_router.setup_assistant import (
    DownloadPlan,
    execute_download_plan,
    plan_model_downloads,
    recommend_setup,
    scan_local_environment as _scan_local_environment,
)


class AdminActionError(RuntimeError):
    """Raised when an admin action cannot run safely."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = dict(details or {})


ACTION_ALIASES = {
    "doctor": "doctor.run",
    "scan": "model.scan",
    "save_config": "config.save_proxy_patch",
    "save-config": "config.save_proxy_patch",
    "proxy_start": "proxy.start",
    "proxy_stop": "proxy.stop",
    "proxy_restart": "proxy.restart",
    "download_plan": "model.download.plan",
    "download_run": "model.download.run",
    "assign_route": "model.assign_route",
    "assign-route": "model.assign_route",
    "benchmark_plan": "benchmark.plan",
    "benchmark_run": "benchmark.run",
    "catalog_diff": "catalog.diff",
    "catalog_apply": "catalog.apply",
    "pricing_diff": "pricing.diff",
    "pricing_apply": "pricing.apply",
    "feedback": "telemetry.feedback.write",
}

MUTATING_ACTIONS = {
    "benchmark.run",
    "catalog.apply",
    "config.save_proxy_patch",
    "model.download.run",
    "model.assign_route",
    "proxy.restart",
    "proxy.start",
    "proxy.stop",
    "pricing.apply",
    "telemetry.feedback.write",
}

CONFIRMATION_ERRORS = {
    "benchmark.run": "Benchmark execution requires confirm=true.",
    "catalog.apply": "Catalog apply requires confirm=true.",
    "config.save_proxy_patch": "Config save requires confirm=true.",
    "model.download.run": "Download execution requires confirm=true.",
    "model.assign_route": "Route assignment requires confirm=true.",
    "proxy.restart": "Proxy restart requires confirm=true.",
    "proxy.start": "Proxy start requires confirm=true.",
    "proxy.stop": "Proxy stop requires confirm=true.",
    "pricing.apply": "Pricing catalog apply requires confirm=true.",
    "telemetry.feedback.write": "Feedback submission requires confirm=true.",
}

_ACTION_DESCRIPTORS: tuple[dict[str, Any], ...] = (
    {
        "id": "doctor.run",
        "label": "Run doctor",
        "mutates": False,
        "requires_confirm": False,
        "description": "Validate local proxy/router config and show remediation.",
    },
    {
        "id": "model.scan",
        "label": "Scan models",
        "mutates": False,
        "requires_confirm": False,
        "description": "Scan local runtimes/models and refresh recommendations.",
    },
    {
        "id": "config.save_proxy_patch",
        "label": "Save config",
        "mutates": True,
        "requires_confirm": True,
        "description": "Apply an explicit patch to routing_proxy.yaml.",
    },
    {
        "id": "proxy.start",
        "label": "Start proxy",
        "mutates": True,
        "requires_confirm": True,
        "description": "Start model-router-proxy as a settings-owned child process.",
    },
    {
        "id": "proxy.stop",
        "label": "Stop proxy",
        "mutates": True,
        "requires_confirm": True,
        "description": "Stop the settings-owned proxy child process.",
    },
    {
        "id": "proxy.restart",
        "label": "Restart proxy",
        "mutates": True,
        "requires_confirm": True,
        "description": "Restart the settings-owned proxy child process.",
    },
    {
        "id": "catalog.diff",
        "label": "Catalog diff",
        "mutates": False,
        "requires_confirm": False,
        "description": "Compare packaged model catalog defaults with local config.",
    },
    {
        "id": "catalog.apply",
        "label": "Apply catalog",
        "mutates": True,
        "requires_confirm": True,
        "description": "Apply packaged catalog defaults to local config.",
    },
    {
        "id": "pricing.status",
        "label": "Pricing status",
        "mutates": False,
        "requires_confirm": False,
        "description": "Inspect local pricing catalog metadata without network checks.",
    },
    {
        "id": "pricing.diff",
        "label": "Pricing diff",
        "mutates": False,
        "requires_confirm": False,
        "description": "Preview packaged pricing metadata against the local override.",
    },
    {
        "id": "pricing.apply",
        "label": "Apply pricing",
        "mutates": True,
        "requires_confirm": True,
        "description": "Write packaged pricing metadata to the local override.",
    },
    {
        "id": "telemetry.feedback.write",
        "label": "Save feedback",
        "mutates": True,
        "requires_confirm": True,
        "description": "Append a wrong-route label to the feedback JSONL file.",
    },
    {
        "id": "model.download.plan",
        "label": "Plan download",
        "mutates": False,
        "requires_confirm": False,
        "description": "Plan model downloads without running them.",
    },
    {
        "id": "model.download.run",
        "label": "Run download",
        "mutates": True,
        "requires_confirm": True,
        "description": "Run an explicitly confirmed model download command.",
    },
    {
        "id": "model.assign_route",
        "label": "Assign route model",
        "mutates": True,
        "requires_confirm": True,
        "description": "Assign a model to a configured route/backend.",
    },
    {
        "id": "benchmark.plan",
        "label": "Plan benchmark",
        "mutates": False,
        "requires_confirm": False,
        "description": "Plan local backend smoke benchmarks.",
    },
    {
        "id": "benchmark.run",
        "label": "Run benchmark",
        "mutates": True,
        "requires_confirm": True,
        "description": "Run confirmed local benchmarks using a synthetic prompt.",
    },
)


def action_descriptors() -> list[dict[str, Any]]:
    """Return shared descriptors for admin actions."""

    return deepcopy(list(_ACTION_DESCRIPTORS))


def run_admin_action(
    action_id: str,
    paths: Mapping[str, Path],
    payload: Mapping[str, Any] | None = None,
    *,
    supervisor: ProxyProcessSupervisor | None = None,
    download_runner: Callable[[tuple[str, ...]], int] | None = None,
    benchmark_runner: Callable[[BenchmarkTarget, float], BenchmarkResult] | None = None,
) -> dict[str, Any]:
    """Run one shared admin action and return a structured action result."""

    normalized = ACTION_ALIASES.get(action_id, action_id)
    action_payload = dict(payload or {})
    _require_confirmation(normalized, action_payload)

    if normalized == "model.scan":
        body = _scan_action(paths)
    elif normalized == "config.save_proxy_patch":
        body = _save_config_action(paths, action_payload)
    elif normalized == "doctor.run":
        body = _doctor_action(paths)
    elif normalized == "proxy.start":
        body = _proxy_start_action(supervisor)
    elif normalized == "proxy.stop":
        body = _proxy_stop_action(supervisor)
    elif normalized == "proxy.restart":
        body = _proxy_restart_action(supervisor)
    elif normalized == "model.download.plan":
        body = _download_plan_action(paths, action_payload)
    elif normalized == "model.download.run":
        body = _download_run_action(paths, action_payload, download_runner)
    elif normalized == "model.assign_route":
        body = _assign_route_action(paths, action_payload)
    elif normalized == "benchmark.plan":
        body = _benchmark_plan_action(paths)
    elif normalized == "benchmark.run":
        body = _benchmark_run_action(paths, benchmark_runner)
    elif normalized == "catalog.diff":
        body = _catalog_diff_action(paths)
    elif normalized == "catalog.apply":
        body = _catalog_apply_action(paths)
    elif normalized == "pricing.status":
        body = _pricing_status_action(paths)
    elif normalized == "pricing.diff":
        body = _pricing_diff_action(paths)
    elif normalized == "pricing.apply":
        body = _pricing_apply_action(paths)
    elif normalized == "telemetry.feedback.write":
        body = _feedback_action(paths, action_payload)
    else:
        raise AdminActionError(f"Unknown admin action: {action_id}")

    return {"ok": body.get("ok", True) is not False, "action_id": normalized, "payload": body}


def _require_confirmation(action_id: str, payload: Mapping[str, Any]) -> None:
    if action_id in MUTATING_ACTIONS and not _payload_bool(
        payload,
        "confirm",
        default=False,
    ):
        raise AdminActionError(
            CONFIRMATION_ERRORS.get(action_id, "Action requires confirm=true."),
            status_code=400,
        )


def _scan_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    discovery = _scan_local_environment_compat()
    benchmark_results = load_benchmark_results(paths["benchmarks"])
    recommendation = recommend_setup(
        discovery,
        download_alternatives=2,
        benchmark_results=benchmark_results,
    )
    plan = _download_plan(
        paths,
        discovery=discovery,
        benchmark_results=benchmark_results,
    )
    return {
        "discovery": discovery.to_dict(),
        "recommendation": recommendation.to_dict(),
        "download_plan": plan.to_dict(),
        "benchmarks": benchmark_summary_compat(paths["benchmarks"]),
    }


def _save_config_action(
    paths: Mapping[str, Path],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return {"ok": True, **save_proxy_config_patch(paths["proxy_config"], payload)}
    except (OSError, ProxyConfigError, ValueError, yaml.YAMLError) as exc:
        raise AdminActionError(str(exc), status_code=400) from exc


def _doctor_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    return _doctor_proxy_config_compat(paths["proxy_config"]).to_dict()


def _proxy_start_action(supervisor: ProxyProcessSupervisor | None) -> dict[str, Any]:
    if supervisor is None:
        raise AdminActionError("Proxy supervisor is unavailable.", status_code=500)
    try:
        return {"ok": True, "proxy": supervisor.start().to_dict()}
    except OSError as exc:
        raise AdminActionError(
            str(exc),
            status_code=500,
            details={"proxy": supervisor.status().to_dict()},
        ) from exc


def _proxy_stop_action(supervisor: ProxyProcessSupervisor | None) -> dict[str, Any]:
    if supervisor is None:
        raise AdminActionError("Proxy supervisor is unavailable.", status_code=500)
    return {"ok": True, "proxy": supervisor.stop().to_dict()}


def _proxy_restart_action(supervisor: ProxyProcessSupervisor | None) -> dict[str, Any]:
    if supervisor is None:
        raise AdminActionError("Proxy supervisor is unavailable.", status_code=500)
    try:
        return {"ok": True, "proxy": supervisor.restart().to_dict()}
    except OSError as exc:
        raise AdminActionError(
            str(exc),
            status_code=500,
            details={"proxy": supervisor.status().to_dict()},
        ) from exc


def _download_plan_action(
    paths: Mapping[str, Path],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        plan = _download_plan_from_payload(paths, payload)
    except ValueError as exc:
        raise AdminActionError(str(exc), status_code=400) from exc
    return {"ok": True, "plan": plan.to_dict()}


def _download_run_action(
    paths: Mapping[str, Path],
    payload: Mapping[str, Any],
    download_runner: Callable[[tuple[str, ...]], int] | None,
) -> dict[str, Any]:
    try:
        plan = _download_plan_from_payload(paths, payload)
    except ValueError as exc:
        raise AdminActionError(str(exc), status_code=400) from exc
    result = execute_download_plan(
        plan,
        execute=True,
        confirmed=True,
        runner=download_runner,
    )
    return {"ok": result.ok, "result": result.to_dict()}


def _assign_route_action(
    paths: Mapping[str, Path],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    route_id = str(payload.get("route_id", "")).strip()
    backend_name = str(payload.get("backend", "")).strip()
    model = str(payload.get("model", "")).strip()
    if not model:
        raise AdminActionError("model is required.", status_code=400)
    try:
        config = load_proxy_config(paths["proxy_config"])
    except (OSError, ProxyConfigError) as exc:
        raise AdminActionError(str(exc), status_code=400) from exc
    if not backend_name and route_id:
        backend_name = config.engine_backends.get(route_id, "")
    if not backend_name:
        raise AdminActionError("backend or route_id is required.", status_code=400)
    if backend_name not in config.backends:
        raise AdminActionError(
            f"backend {backend_name!r} is not configured.",
            status_code=400,
        )
    result = save_proxy_config_patch(
        paths["proxy_config"],
        {
            "backends": {backend_name: {"model": model}},
        },
    )
    return {
        "ok": True,
        "restart_recommended": True,
        "assignment": {
            "route_id": route_id,
            "backend": backend_name,
            "model": model,
        },
        **result,
    }


def _benchmark_plan_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    try:
        targets = plan_backend_benchmarks(paths["proxy_config"])
    except Exception as exc:
        raise AdminActionError(str(exc), status_code=400) from exc
    return {
        "ok": True,
        "targets": [target.to_dict() for target in targets],
        "output_path": str(paths["benchmarks"]),
    }


def _benchmark_run_action(
    paths: Mapping[str, Path],
    benchmark_runner: Callable[[BenchmarkTarget, float], BenchmarkResult] | None,
) -> dict[str, Any]:
    try:
        targets = plan_backend_benchmarks(paths["proxy_config"])
    except Exception as exc:
        raise AdminActionError(str(exc), status_code=400) from exc
    result = execute_benchmark_plan(
        targets,
        output_path=paths["benchmarks"],
        execute=True,
        confirmed=True,
        runner=benchmark_runner,
    )
    return {"ok": result.ok, "result": result.to_dict()}


def _catalog_diff_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    diff = catalog_diff(paths["model_router_config"])
    return {"ok": True, "diff": diff.to_dict()}


def _catalog_apply_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = apply_catalog_update(paths["model_router_config"], confirmed=True)
    return {
        "ok": result.ok,
        "result": result.to_dict(),
        "catalog": catalog_status(paths["model_router_config"]).to_dict(),
    }


def _pricing_status_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "ok": True,
        "status": pricing_status(paths["pricing"]).to_dict(),
    }


def _pricing_diff_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "ok": True,
        "diff": pricing_diff(paths["pricing"]).to_dict(),
    }


def _pricing_apply_action(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = apply_pricing_catalog(paths["pricing"], confirmed=True)
    return {
        "ok": result.ok,
        "result": result.to_dict(),
        "status": pricing_status(paths["pricing"]).to_dict(),
    }


def _feedback_action(
    paths: Mapping[str, Path],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    request_id = str(payload.get("request_id", "")).strip()
    expected_engine = str(payload.get("expected_engine", "")).strip()
    outcome_label = str(payload.get("outcome_label", "")).strip() or None
    notes = str(payload.get("notes", "")).strip() or None
    if not request_id or not expected_engine:
        raise AdminActionError(
            "request_id and expected_engine are required.",
            status_code=400,
        )
    try:
        feedback = build_feedback(
            request_id=request_id,
            expected_engine=expected_engine,
            outcome_label=outcome_label,
            notes=notes,
        )
    except ValueError as exc:
        raise AdminActionError(str(exc), status_code=400) from exc
    writer = RoutingLogWriter(paths["feedback"])
    if not writer.write(feedback):
        raise AdminActionError("Failed to write feedback.", status_code=500)
    return {"ok": True, "feedback_path": str(paths["feedback"])}


def _download_plan(
    paths: Mapping[str, Path],
    *,
    discovery: Any = None,
    benchmark_results: Any = None,
) -> DownloadPlan:
    return plan_model_downloads(
        discovery=discovery,
        alternatives=2,
        local_root=paths["models"],
        benchmark_results=benchmark_results,
    )


def _download_plan_from_payload(
    paths: Mapping[str, Path],
    payload: Mapping[str, Any],
) -> DownloadPlan:
    route = str(payload.get("route", "")).strip() or None
    repo_id = str(payload.get("repo_id", "")).strip() or None
    adapter = str(payload.get("adapter", "")).strip() or None
    routes = (route,) if route else None
    return plan_model_downloads(
        routes=routes,
        repo_id=repo_id,
        adapter=adapter,
        alternatives=1,
        local_root=paths["models"],
        benchmark_results=load_benchmark_results(paths["benchmarks"]),
    )


def _payload_bool(payload: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _scan_local_environment_compat() -> Any:
    from hermes.plugins.model_router import settings_ui as legacy_settings_ui

    return getattr(legacy_settings_ui, "scan_local_environment", _scan_local_environment)()


def _doctor_proxy_config_compat(config_path: Path) -> Any:
    from hermes.plugins.model_router import settings_ui as legacy_settings_ui

    doctor = getattr(legacy_settings_ui, "doctor_proxy_config", _doctor_proxy_config)
    return doctor(config_path)


def benchmark_summary_compat(path: Path) -> dict[str, Any]:
    from hermes.plugins.model_router import settings_ui as legacy_settings_ui
    from hermes.plugins.model_router.model_benchmark import benchmark_summary

    summary = getattr(legacy_settings_ui, "benchmark_summary", benchmark_summary)
    return summary(path)
