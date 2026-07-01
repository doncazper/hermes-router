"""Local admin settings UI for ModelRouter.

The settings surface is intentionally an admin/config UI: it never accepts
prompts, never renders chat transcripts, and keeps proxy/runtime operations
explicitly user-triggered.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from html import escape
import json
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

import yaml

from hermes.plugins.model_router.admin.actions import (
    AdminActionError,
    action_descriptors,
    run_admin_action,
)
from hermes.plugins.model_router.admin.config_edit import (
    save_proxy_config_patch as _shared_save_proxy_config_patch,
)
from hermes.plugins.model_router.admin.model_library import build_model_library_state
from hermes.plugins.model_router.admin.state import build_admin_state, settings_paths
from hermes.plugins.model_router.admin.supervisor import (
    ProxyProcessStatus,
    ProxyProcessSupervisor,
)
from hermes.plugins.model_router.catalog_update import catalog_status
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.eval_runner import (
    eval_comparison_summaries_from_rows,
    load_eval_results,
)
from hermes.plugins.model_router.installer import build_installer_state
from hermes.plugins.model_router.product import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PROXY_PORT,
    PRESETS,
    doctor_proxy_config,  # noqa: F401 - compatibility patch point for tests/callers.
    initialize_product_config,
)
from hermes.plugins.model_router.pricing_catalog import pricing_status
from hermes.plugins.model_router.model_benchmark import (
    BenchmarkResult,
    BenchmarkTarget,
    benchmark_summary,
    load_benchmark_results,
)
from hermes.plugins.model_router.maturity import feature_maturity_state
from hermes.plugins.model_router.proxy_config import (
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)
from hermes.plugins.model_router.profiles import ROUTING_PROFILE_VALUES
from hermes.plugins.model_router.routing_log import (
    OUTCOME_LABELS,
    PROMPT_CAPTURE_MODES,
    read_jsonl,
    redact_text,
)
from hermes.plugins.model_router.runtime_adapters import runtime_state_for_backend
from hermes.plugins.model_router.setup_assistant import (
    DownloadPlan,
    plan_model_downloads,
    recommend_setup,
    scan_local_environment,
)
from hermes.plugins.model_router.telemetry import (
    event_usage_summary,
    feedback_summary,
    pricing_override_skeleton_from_gaps,
    replay_events,
    review_queue,
)


DEFAULT_SETTINGS_HOST = "127.0.0.1"
DEFAULT_SETTINGS_PORT = 8099
ROUTE_LABELS = {
    "fast": "fast",
    "balanced": "balanced",
    "reasoning": "reasoning",
    "code": "code",
}
DASHBOARD_ENGINE_ORDER = (
    "fast_local",
    "balanced_local",
    "reasoning_local",
    "code_agent",
    "web_research",
    "multimodal_vision",
    "image_generation",
    "human_confirm",
)
ROUTE_CLASS_BY_ENGINE = {
    "fast_local": "Simple",
    "balanced_local": "Balanced",
    "reasoning_local": "Reasoning",
    "code_agent": "Coding",
    "web_research": "Research",
    "multimodal_vision": "Vision",
    "image_generation": "Image generation",
    "human_confirm": "Risky actions",
}
ROUTE_DESCRIPTION_BY_ENGINE = {
    "fast_local": "Fast/local route",
    "balanced_local": "Default everyday route",
    "reasoning_local": "Reasoning route",
    "code_agent": "Code and repository route",
    "web_research": "Research/RAG route",
    "multimodal_vision": "Vision/OCR route",
    "image_generation": "Image generation route",
    "human_confirm": "Human confirmation gate",
}
TOOL_HINT_BY_ENGINE = {
    "code_agent": "Yes",
    "web_research": "Yes",
    "multimodal_vision": "Yes",
    "image_generation": "Limited",
    "human_confirm": "N/A",
}


class SettingsDependencyError(RuntimeError):
    """Raised when optional settings UI dependencies are not installed."""


def create_settings_app(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    proxy_supervisor: ProxyProcessSupervisor | None = None,
    download_runner: Callable[[tuple[str, ...]], int] | None = None,
    benchmark_runner: Callable[[BenchmarkTarget, float], BenchmarkResult] | None = None,
):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised through CLI path.
        raise SettingsDependencyError(
            "settings UI dependencies are missing; install with "
            'python -m pip install "hermes-router[proxy]"'
        ) from exc
    globals()["Request"] = Request

    paths = settings_paths(config_dir)
    supervisor = proxy_supervisor or ProxyProcessSupervisor(
        config_path=paths["proxy_config"],
        log_path=paths["settings_proxy_log"],
    )
    app = FastAPI(
        title="ModelRouter settings",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(render_dashboard_page(build_settings_state(paths, supervisor)))

    @app.get("/compact", response_class=HTMLResponse)
    async def compact() -> HTMLResponse:
        return HTMLResponse(render_compact_page(build_settings_state(paths, supervisor)))

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(build_settings_state(paths, supervisor))

    def action_error_response(exc: AdminActionError) -> JSONResponse:
        payload = {"ok": False, "error": str(exc)}
        payload.update(exc.details)
        return JSONResponse(payload, status_code=exc.status_code)

    def action_payload(result: Mapping[str, Any]) -> dict[str, Any]:
        payload = result.get("payload", {})
        return payload if isinstance(payload, dict) else {"ok": False}

    async def run_endpoint_action(
        action_id: str,
        request: Request | None = None,
    ) -> JSONResponse:
        payload = await _request_payload(request) if request is not None else {}
        try:
            result = run_admin_action(
                action_id,
                paths,
                payload,
                supervisor=supervisor,
                download_runner=download_runner,
                benchmark_runner=benchmark_runner,
            )
        except AdminActionError as exc:
            return action_error_response(exc)
        return JSONResponse(action_payload(result))

    @app.post("/api/action")
    async def api_action(request: Request) -> JSONResponse:
        payload = await _request_payload(request)
        action_id = str(payload.get("action_id", "")).strip()
        action_body = payload.get("payload")
        if isinstance(action_body, dict):
            body = dict(action_body)
        else:
            body = {
                key: value
                for key, value in payload.items()
                if key not in {"action_id", "payload"}
            }
        if "confirm" in payload and "confirm" not in body:
            body["confirm"] = payload["confirm"]
        try:
            result = run_admin_action(
                action_id,
                paths,
                body,
                supervisor=supervisor,
                download_runner=download_runner,
                benchmark_runner=benchmark_runner,
            )
        except AdminActionError as exc:
            return action_error_response(exc)
        return JSONResponse(result)

    @app.post("/api/scan")
    async def api_scan() -> JSONResponse:
        return await run_endpoint_action("model.scan")

    @app.post("/api/save-config")
    async def api_save_config(request: Request) -> JSONResponse:
        return await run_endpoint_action("config.save_proxy_patch", request)

    @app.post("/api/doctor")
    async def api_doctor() -> JSONResponse:
        return await run_endpoint_action("doctor.run")

    @app.post("/api/proxy/start")
    async def api_proxy_start(request: Request) -> JSONResponse:
        return await run_endpoint_action("proxy.start", request)

    @app.post("/api/proxy/stop")
    async def api_proxy_stop(request: Request) -> JSONResponse:
        return await run_endpoint_action("proxy.stop", request)

    @app.post("/api/proxy/restart")
    async def api_proxy_restart(request: Request) -> JSONResponse:
        return await run_endpoint_action("proxy.restart", request)

    @app.post("/api/download/plan")
    async def api_download_plan(request: Request) -> JSONResponse:
        return await run_endpoint_action("model.download.plan", request)

    @app.post("/api/download/run")
    async def api_download_run(request: Request) -> JSONResponse:
        return await run_endpoint_action("model.download.run", request)

    @app.post("/api/model/assign-route")
    async def api_model_assign_route(request: Request) -> JSONResponse:
        return await run_endpoint_action("model.assign_route", request)

    @app.post("/api/benchmark/plan")
    async def api_benchmark_plan() -> JSONResponse:
        return await run_endpoint_action("benchmark.plan")

    @app.post("/api/benchmark/run")
    async def api_benchmark_run(request: Request) -> JSONResponse:
        return await run_endpoint_action("benchmark.run", request)

    @app.post("/api/catalog/diff")
    async def api_catalog_diff() -> JSONResponse:
        return await run_endpoint_action("catalog.diff")

    @app.post("/api/catalog/apply")
    async def api_catalog_apply(request: Request) -> JSONResponse:
        return await run_endpoint_action("catalog.apply", request)

    @app.post("/api/pricing/status")
    async def api_pricing_status() -> JSONResponse:
        return await run_endpoint_action("pricing.status")

    @app.post("/api/pricing/diff")
    async def api_pricing_diff() -> JSONResponse:
        return await run_endpoint_action("pricing.diff")

    @app.post("/api/pricing/apply")
    async def api_pricing_apply(request: Request) -> JSONResponse:
        return await run_endpoint_action("pricing.apply", request)

    @app.post("/api/feedback")
    async def api_feedback(request: Request) -> JSONResponse:
        return await run_endpoint_action("telemetry.feedback.write", request)

    return app


def run_settings_server(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    host: str = DEFAULT_SETTINGS_HOST,
    port: int = DEFAULT_SETTINGS_PORT,
    open_browser: bool = True,
    log_level: str = "info",
) -> int:
    _ensure_local_host(host)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised through CLI path.
        raise SettingsDependencyError(
            "settings UI dependencies are missing; install with "
            'python -m pip install "hermes-router[proxy]"'
        ) from exc

    app = create_settings_app(config_dir=config_dir)
    url = f"http://{host}:{port}"
    print(f"ModelRouter settings: {url}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0


def _ensure_local_host(host: str) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SettingsDependencyError(
            "settings UI is local-only; use 127.0.0.1, localhost, or ::1"
        )


def build_settings_state(
    paths: Mapping[str, Path],
    supervisor: ProxyProcessSupervisor | None = None,
) -> dict[str, Any]:
    return build_admin_state(paths, supervisor)


def save_proxy_config_patch(
    config_path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _shared_save_proxy_config_patch(config_path, payload)


def _build_settings_state_impl(
    paths: Mapping[str, Path],
    supervisor: ProxyProcessSupervisor | None = None,
) -> dict[str, Any]:
    proxy_config_path = paths["proxy_config"]
    config: RoutingProxyConfig | None = None
    config_error: str | None = None
    try:
        config = load_proxy_config(proxy_config_path)
    except ProxyConfigError as exc:
        config_error = str(exc)

    discovery = scan_local_environment()
    benchmark_results = load_benchmark_results(paths["benchmarks"])
    eval_results_path = paths.get("eval_results")
    eval_results = load_eval_results(eval_results_path) if eval_results_path else ()
    eval_comparisons = eval_comparison_summaries_from_rows(eval_results)
    recommendation = recommend_setup(
        discovery,
        download_alternatives=2,
        benchmark_results=benchmark_results,
    )
    download_plan = _download_plan(
        paths,
        discovery=discovery,
        benchmark_results=benchmark_results,
    )
    proxy_process = (
        supervisor.status().to_dict()
        if supervisor is not None
        else ProxyProcessStatus("unknown").to_dict()
    )
    backend_states = _redacted_backend_states(config)
    runtime_models = _runtime_models_from_backend_states(backend_states)
    recent_events = _recent_routing_events(paths["events"])
    latest_event = recent_events[0] if recent_events else {}
    state: dict[str, Any] = {
        "product": "ModelRouter",
        "paths": {name: str(path) for name, path in paths.items()},
        "presets": list(PRESETS),
        "prompt_capture_modes": list(PROMPT_CAPTURE_MODES),
        "config_exists": proxy_config_path.exists(),
        "config_valid": config is not None,
        "config_error": config_error,
        "proxy": _redacted_proxy_state(config),
        "provider_policy": _provider_policy_state(config),
        "backend_policy": _backend_policy_state(config),
        "verifier": _verifier_state(config),
        "catalog": catalog_status(paths["model_router_config"]).to_dict(),
        "pricing_catalog": pricing_status(paths["pricing"]).to_dict(),
        "backends": backend_states,
        "engine_backends": dict(sorted(config.engine_backends.items())) if config else {},
        "observability": _observability_state(config),
        "discovery": discovery.to_dict(),
        "recommendation": recommendation.to_dict(),
        "download_plan": download_plan.to_dict(),
        "model_library": build_model_library_state(
            paths=paths,
            config=config,
            discovery=discovery,
            recommendation=recommendation,
            download_plan=download_plan,
            benchmark_results=benchmark_results,
            eval_results=eval_results,
            runtime_models=runtime_models,
        ),
        "installer": build_installer_state(paths, discovery=discovery),
        "benchmarks": benchmark_summary(paths["benchmarks"]),
        "workflow_benchmarks": _workflow_benchmark_state(paths),
        "evals": {
            "result_path": str(eval_results_path) if eval_results_path else "",
            "comparisons": list(eval_comparisons),
            "privacy": _eval_privacy_state(),
            "read_only": True,
        },
        "maturity": feature_maturity_state(),
        "telemetry": _telemetry_state(paths, config),
        "proxy_process": proxy_process,
        "route_map": _route_map_state(config, latest_event),
        "provider_runtime": _provider_runtime_state(config, latest_event),
        "route_receipt": _route_receipt_state(config, latest_event),
        "recent_events": recent_events,
        "review": _review_state(paths),
        "model_options": _model_options_state(discovery, download_plan),
        "actions": action_descriptors(),
        "not_chat_ui": True,
    }
    return state


def _save_proxy_config_patch_impl(
    config_path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
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


def render_settings_page(state: Mapping[str, Any]) -> str:
    config_error = state.get("config_error")
    backend_rows = "\n".join(_backend_row(backend) for backend in state["backends"])
    model_options = "\n".join(
        f"<option value=\"{escape(model['repo_id'])}\">{escape(model['repo_id'])}</option>"
        for model in state["discovery"]["models"][:80]
    )
    download_rows = "\n".join(
        _download_row(item) for item in state["download_plan"]["suggestions"]
    )
    recommendation_rows = "\n".join(
        _recommendation_row(item)
        for item in state["recommendation"].get("local_model_recommendations", [])[:8]
    )
    if not recommendation_rows:
        recommendation_rows = '<tr><td colspan="4" class="muted">No local models scored yet.</td></tr>'
    telemetry = state["telemetry"]
    benchmarks = state["benchmarks"]
    proxy = state["proxy"]
    observability = state["observability"]
    provider_policy = state["provider_policy"]
    backend_policy = state["backend_policy"]
    proxy_process = state["proxy_process"]
    status_class = "ok" if state["config_valid"] else "bad"
    config_status = "valid" if state["config_valid"] else "invalid"
    proxy_state = escape(str(proxy_process["state"]))
    proxy_endpoint = escape(str(proxy.get("endpoint") or "not configured"))
    proxy_log_path = escape(str(proxy_process.get("log_path") or ""))
    proxy_host = escape(str(proxy.get("host") or "127.0.0.1"))
    proxy_port = escape(str(proxy.get("port") or DEFAULT_PROXY_PORT))
    routing_mode_options = _options(
        ["decision", "manual"],
        selected=proxy.get("routing_mode") or "decision",
    )
    default_backend_options = _backend_options(
        state.get("backends", []),
        selected=proxy.get("default_backend"),
    )
    default_model = escape(str(proxy.get("default_model") or ""))
    respect_client_model_options = _bool_options(proxy.get("respect_client_model"))
    unknown_model_behavior_options = _options(
        ["fallback_to_default", "reject_404"],
        selected=proxy.get("unknown_model_behavior") or "fallback_to_default",
    )
    safety_gate_mode_options = _options(
        ["decision_only", "always_static", "off"],
        selected=proxy.get("safety_gate_mode") or "decision_only",
    )
    observability_enabled = _bool_options(observability.get("enabled"))
    prompt_capture_options = _options(
        state["prompt_capture_modes"],
        selected=observability.get("prompt_capture"),
    )
    observability_log_path = escape(str(observability.get("log_path") or ""))
    telemetry_events = escape(str(telemetry.get("events", 0)))
    telemetry_feedback = escape(str(telemetry.get("feedback_labels", 0)))
    telemetry_unlabeled = escape(str(telemetry.get("unlabeled_replayable", 0)))
    telemetry_mismatches = escape(str(telemetry.get("expected_mismatch_count", 0)))
    telemetry_fallbacks = escape(str(telemetry.get("fallback_count", 0)))
    telemetry_usage_events = escape(str(telemetry.get("usage_events", 0)))
    telemetry_usage_tokens = escape(_format_usage_summary(telemetry))
    telemetry_estimated_cost = escape(_format_cost_summary(telemetry))
    telemetry_outcomes = escape(_compact_counts(telemetry.get("outcome_label_counts", {})))
    telemetry_pricing_matches = escape(_compact_counts(telemetry.get("pricing_match_counts", {})))
    telemetry_catalog_coverage = escape(
        _format_catalog_coverage(telemetry.get("catalog_coverage"))
    )
    telemetry_catalog_gaps = escape(
        _format_catalog_gap_list(telemetry.get("catalog_coverage_gaps"))
    )
    engine_counts = escape(_compact_counts(telemetry.get("selected_engine_counts", {})))
    backend_counts = escape(_compact_counts(telemetry.get("backend_counts", {})))
    status_counts = escape(_compact_counts(telemetry.get("status_counts", {})))
    usage_backend_counts = escape(
        _compact_usage_groups(telemetry.get("usage_by_backend", {}))
    )
    recent_request_ids = escape(
        ", ".join(telemetry.get("recent_request_ids", [])[:5]) or "none"
    )
    benchmark_results = escape(str(benchmarks.get("results", 0)))
    benchmark_completed = escape(str(benchmarks.get("completed", 0)))
    benchmark_failed = escape(str(benchmarks.get("failed", 0)))
    benchmark_best = escape(_benchmark_best_summary(benchmarks))
    provider_policy_summary = escape(_settings_provider_policy_summary(provider_policy))
    backend_allowlist = escape(", ".join(backend_policy.get("backend_allowlist") or []))
    backend_denylist = escape(", ".join(backend_policy.get("backend_denylist") or []))
    doctor_disabled = "" if state["config_exists"] else "disabled"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ModelRouter Settings</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #f0f3f6;
      --text: #17202a;
      --muted: #5b6876;
      --line: #d8dee6;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #9a3412;
      --bad: #b42318;
      --good: #047857;
      --shadow: 0 1px 2px rgba(17, 24, 39, .06);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.45;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .topbar {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{ font-size: 20px; margin: 0; letter-spacing: 0; }}
    h2 {{ font-size: 15px; margin: 0 0 12px; letter-spacing: 0; }}
    h3 {{ font-size: 13px; margin: 0 0 8px; color: var(--muted); letter-spacing: 0; }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(330px, .75fr);
      gap: 16px;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 16px;
    }}
    .stack {{ display: flex; flex-direction: column; gap: 16px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .field {{ display: flex; flex-direction: column; gap: 5px; }}
    label {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    input, select, textarea {{
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 7px 9px;
      font: inherit;
    }}
    textarea {{ min-height: 70px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    button {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface-2);
      color: var(--text);
      padding: 7px 11px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    button.primary:hover {{ background: var(--accent-dark); }}
    button.danger {{ color: var(--bad); }}
    button:disabled {{ opacity: .5; cursor: not-allowed; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; }}
    .badge {{ display: inline-flex; align-items: center; gap: 5px; border-radius: 999px; padding: 3px 8px; background: var(--surface-2); font-size: 12px; color: var(--muted); }}
    .badge.ok {{ color: var(--good); background: #ecfdf5; }}
    .badge.bad {{ color: var(--bad); background: #fef3f2; }}
    .badge.warn {{ color: var(--warn); background: #fff7ed; }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    .message {{ min-height: 20px; color: var(--muted); }}
    .notice {{ border-left: 3px solid var(--accent); padding: 8px 10px; background: #ecfeff; border-radius: 6px; }}
    .error {{ border-left-color: var(--bad); background: #fef3f2; color: var(--bad); }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; padding: 12px; }}
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .grid {{ grid-template-columns: 1fr; }}
      th:nth-child(4), td:nth-child(4) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>ModelRouter Settings</h1>
        <div class="muted">Local admin UI for proxy configuration, telemetry, and runtime operations. No chat surface.</div>
      </div>
      <div class="toolbar">
        <span class="badge {status_class}">config {config_status}</span>
        <span class="badge">proxy {proxy_state}</span>
      </div>
    </div>
  </header>
  <main>
    <div>
      {_config_notice(config_error)}
      <section>
        <h2>Proxy Process</h2>
        <div class="toolbar">
          <button class="primary" onclick="postAction('/api/proxy/start', {{confirm: true}})">Start</button>
          <button onclick="postAction('/api/proxy/restart', {{confirm: true}})">Restart</button>
          <button class="danger" onclick="postAction('/api/proxy/stop', {{confirm: true}})">Stop</button>
          <button {doctor_disabled} onclick="postAction('/api/doctor')">Run doctor</button>
        </div>
        <p class="muted">Endpoint: <span class="mono">{proxy_endpoint}</span></p>
        <p class="muted">Proxy log: <span class="mono">{proxy_log_path}</span></p>
        <div id="action-message" class="message"></div>
      </section>

      <section>
        <h2>Config</h2>
        <div class="grid">
          <div class="field">
            <label for="preset">Preset</label>
            <select id="preset">
              {_options(state["presets"], selected="lmstudio")}
            </select>
            <button onclick="applyPreset()">Apply preset template</button>
          </div>
          <div class="field">
            <label for="model-options">Scanned or recommended models</label>
            <select id="model-options">
              <option value="">Choose a scanned model</option>
              {model_options}
            </select>
          </div>
          <div class="field">
            <label for="proxy-host">Proxy host</label>
            <input id="proxy-host" value="{proxy_host}">
          </div>
          <div class="field">
            <label for="proxy-port">Proxy port</label>
            <input id="proxy-port" type="number" min="1" value="{proxy_port}">
          </div>
          <div class="field">
            <label for="routing-mode">Routing mode</label>
            <select id="routing-mode">{routing_mode_options}</select>
          </div>
          <div class="field">
            <label for="default-backend">Manual default backend</label>
            <select id="default-backend">{default_backend_options}</select>
          </div>
          <div class="field">
            <label for="default-model">Manual default model</label>
            <input id="default-model" value="{default_model}">
          </div>
          <div class="field">
            <label for="respect-client-model">Respect client model</label>
            <select id="respect-client-model">{respect_client_model_options}</select>
          </div>
          <div class="field">
            <label for="unknown-model-behavior">Unknown model behavior</label>
            <select id="unknown-model-behavior">{unknown_model_behavior_options}</select>
          </div>
          <div class="field">
            <label for="safety-gate-mode">Safety gate mode</label>
            <select id="safety-gate-mode">{safety_gate_mode_options}</select>
          </div>
          <div class="field">
            <label for="observability-enabled">Observability</label>
            <select id="observability-enabled">
              {observability_enabled}
            </select>
          </div>
          <div class="field">
            <label for="prompt-capture">Prompt capture mode</label>
            <select id="prompt-capture">
              {prompt_capture_options}
            </select>
          </div>
        </div>
        <div class="field" style="margin-top:12px">
          <label for="observability-log">Telemetry log path</label>
          <input id="observability-log" value="{observability_log_path}">
        </div>
        <p class="muted">Literal API keys are not displayed. Use environment variables where possible.</p>
        <button class="primary" onclick="saveConfig()">Save config</button>
      </section>

      <section>
        <h2>Policy Controls</h2>
        <h3>Backend policy</h3>
        <div class="grid">
          <div class="field">
            <label for="backend-allowlist">Backend allowlist</label>
            <input id="backend-allowlist" value="{backend_allowlist}">
          </div>
          <div class="field">
            <label for="backend-denylist">Backend denylist</label>
            <input id="backend-denylist" value="{backend_denylist}">
          </div>
        </div>
        <p class="muted">Provider policy: <span class="mono">{provider_policy_summary}</span></p>
      </section>

      <section>
        <h2>Per-route Backends</h2>
        <table id="backend-table">
          <thead>
            <tr><th>Route</th><th>Model</th><th>Base URL</th><th>Runtime</th><th>Status</th></tr>
          </thead>
          <tbody>{backend_rows}</tbody>
        </table>
      </section>

      <section>
        <h2>Model Recommendations</h2>
        <table>
          <thead><tr><th>Route</th><th>Model</th><th>Label</th><th>Score</th></tr></thead>
          <tbody>{recommendation_rows}</tbody>
        </table>
      </section>
    </div>

    <div>
      <section>
        <h2>Telemetry</h2>
        <div class="grid">
          <div><h3>events</h3><strong>{telemetry_events}</strong></div>
          <div><h3>feedback labels</h3><strong>{telemetry_feedback}</strong></div>
          <div><h3>unlabeled replayable</h3><strong>{telemetry_unlabeled}</strong></div>
          <div><h3>mismatches</h3><strong>{telemetry_mismatches}</strong></div>
          <div><h3>fallbacks</h3><strong>{telemetry_fallbacks}</strong></div>
          <div><h3>usage events</h3><strong>{telemetry_usage_events}</strong></div>
          <div><h3>tokens</h3><span class="mono">{telemetry_usage_tokens}</span></div>
          <div><h3>estimated cost</h3><span class="mono">{telemetry_estimated_cost}</span></div>
          <div><h3>outcomes</h3><span class="mono">{telemetry_outcomes}</span></div>
          <div><h3>pricing</h3><span class="mono">{telemetry_pricing_matches}</span></div>
          <div><h3>catalog coverage</h3><span class="mono">{telemetry_catalog_coverage}</span></div>
          <div><h3>coverage gaps</h3><span class="mono">{telemetry_catalog_gaps}</span></div>
          <div><h3>engines</h3><span class="mono">{engine_counts}</span></div>
          <div><h3>backends</h3><span class="mono">{backend_counts}</span></div>
          <div><h3>usage by backend</h3><span class="mono">{usage_backend_counts}</span></div>
          <div><h3>statuses</h3><span class="mono">{status_counts}</span></div>
        </div>
        <p class="muted">Recent request ids: <span class="mono">{recent_request_ids}</span></p>
      </section>

      <section>
        <h2>Label Wrong Route</h2>
        <div class="field">
          <label for="feedback-request-id">Request ID</label>
          <input id="feedback-request-id" placeholder="X-ModelRouter-Request-ID">
        </div>
        <div class="field">
          <label for="feedback-engine">Expected engine</label>
          <input id="feedback-engine" placeholder="code_agent">
        </div>
        <div class="field">
          <label for="feedback-outcome">Outcome label</label>
          <select id="feedback-outcome">{_outcome_label_options()}</select>
        </div>
        <div class="field">
          <label for="feedback-notes">Notes</label>
          <textarea id="feedback-notes" placeholder="Optional private note"></textarea>
        </div>
        <button onclick="sendFeedback()">Save feedback</button>
      </section>

      <section>
        <h2>Downloads</h2>
        <p class="muted">Plans are safe to inspect. Running a download requires an explicit confirmed click.</p>
        <table>
          <thead><tr><th>Route</th><th>Repo</th><th>Action</th></tr></thead>
          <tbody>{download_rows}</tbody>
        </table>
      </section>

      <section>
        <h2>Benchmarks</h2>
        <div class="grid">
          <div><h3>results</h3><strong>{benchmark_results}</strong></div>
          <div><h3>completed</h3><strong>{benchmark_completed}</strong></div>
          <div><h3>failed</h3><strong>{benchmark_failed}</strong></div>
          <div><h3>best</h3><span class="mono">{benchmark_best}</span></div>
        </div>
        <p class="muted">Uses a fixed synthetic smoke prompt and stores metrics only.</p>
        <div class="toolbar">
          <button onclick="postAction('/api/benchmark/plan')">Plan benchmark</button>
          <button onclick="runBenchmark()">Run benchmark</button>
        </div>
      </section>

      <section>
        <h2>Doctor Output</h2>
        <pre id="doctor-output" class="mono">{escape("Run doctor to refresh local remediation.")}</pre>
      </section>
    </div>
  </main>
  <script>
    async function postAction(path, payload = {{}}) {{
      const response = await fetch(path, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
      }});
      const data = await response.json();
      document.getElementById('action-message').textContent = data.ok === false ? data.error : 'Action complete';
      if (path === '/api/doctor') {{
        document.getElementById('doctor-output').textContent = JSON.stringify(data, null, 2);
      }}
      return data;
    }}
    function backendPayload() {{
      const backends = {{}};
      document.querySelectorAll('[data-backend]').forEach(row => {{
        const name = row.dataset.backend;
        backends[name] = {{
          model: row.querySelector('[data-field="model"]').value,
          base_url: row.querySelector('[data-field="base_url"]').value,
          runtime: {{
            enabled: row.querySelector('[data-field="runtime_enabled"]').value === 'true',
            kind: row.querySelector('[data-field="runtime_kind"]').value,
            command: row.querySelector('[data-field="runtime_command"]').value,
            readiness_url: row.querySelector('[data-field="readiness_url"]').value,
            idle_timeout_seconds: row.querySelector('[data-field="idle_timeout_seconds"]').value,
            log_path: row.querySelector('[data-field="log_path"]').value
          }}
        }};
      }});
      return backends;
    }}
    async function saveConfig() {{
      const payload = {{
        proxy: {{
          host: document.getElementById('proxy-host').value,
          port: document.getElementById('proxy-port').value,
          routing_mode: document.getElementById('routing-mode').value,
          default_backend: document.getElementById('default-backend').value,
          default_model: document.getElementById('default-model').value,
          respect_client_model: document.getElementById('respect-client-model').value === 'true',
          unknown_model_behavior: document.getElementById('unknown-model-behavior').value,
          safety_gate_mode: document.getElementById('safety-gate-mode').value
        }},
        observability: {{
          enabled: document.getElementById('observability-enabled').value === 'true',
          prompt_capture: document.getElementById('prompt-capture').value,
          log_path: document.getElementById('observability-log').value
        }},
        backend_policy: {{
          backend_allowlist: document.getElementById('backend-allowlist').value,
          backend_denylist: document.getElementById('backend-denylist').value
        }},
        backends: backendPayload()
      }};
      payload.confirm = true;
      await postAction('/api/save-config', payload);
    }}
    async function applyPreset() {{
      const preset = document.getElementById('preset').value;
      if (!window.confirm('Replace current config with the ' + preset + ' preset?')) return;
      await postAction('/api/save-config', {{confirm: true, apply_preset: true, preset: preset}});
    }}
    async function sendFeedback() {{
      await postAction('/api/feedback', {{
        confirm: true,
        request_id: document.getElementById('feedback-request-id').value,
        expected_engine: document.getElementById('feedback-engine').value,
        outcome_label: document.getElementById('feedback-outcome').value,
        notes: document.getElementById('feedback-notes').value
      }});
    }}
    async function runDownload(route, repoId) {{
      if (!window.confirm('Download ' + repoId + ' for ' + route + '?')) return;
      await postAction('/api/download/run', {{confirm: true, route: route, repo_id: repoId}});
    }}
    async function runBenchmark() {{
      if (!window.confirm('Run local backend benchmark requests with a fixed synthetic prompt?')) return;
      await postAction('/api/benchmark/run', {{confirm: true}});
    }}
  </script>
</body>
</html>"""


def render_dashboard_page(state: Mapping[str, Any]) -> str:
    """Render the local-control dashboard from real config and telemetry state."""

    config_error = state.get("config_error")
    proxy = state["proxy"]
    observability = state["observability"]
    proxy_process = state.get("proxy_process", {})
    endpoint = escape(str(proxy.get("endpoint") or "http://127.0.0.1:8082/v1"))
    profile_value = str(proxy.get("routing_profile") or "balanced")
    profile_label = _profile_label(profile_value)
    telemetry_state = "On" if observability.get("enabled") else "Off"
    telemetry_dot = "green-dot" if observability.get("enabled") else "yellow-dot"
    proxy_state = str(proxy_process.get("state") or "unknown")
    proxy_dot = (
        "green-dot"
        if proxy_state == "running"
        else ("yellow-dot" if proxy_state in {"stopped", "unknown"} else "red-dot")
    )
    proxy_label = proxy_state.replace("_", " ").title()
    health_checks = _dashboard_health_checks(state)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ModelRouter Settings</title>
  <style>{_dashboard_css()}</style>
</head>
<body>
  <div class="app-shell" id="dashboard">
    <aside class="sidebar" aria-label="Primary navigation">
      <div class="traffic-lights" aria-hidden="true">
        <span class="red"></span><span class="yellow"></span><span class="green"></span>
      </div>
      <nav class="side-nav">
        {_sidebar_item("llama.cpp", "active", "runtimes")}
        {_sidebar_item("Ollama", "", "providers")}
        {_sidebar_item("Models", "", "models")}
        {_sidebar_item("Runtimes", "", "runtimes")}
        {_sidebar_item("Risky", "", "safety")}
        {_sidebar_item("Research", "", "routing-map")}
        {_sidebar_item("CloseAI", "", "providers")}
        {_sidebar_item("Codings", "", "routing-map")}
      </nav>
      <p class="sidebar-note">No chat surface. Local infrastructure only.</p>
    </aside>

    <main class="workspace">
      <header class="topbar">
        <div class="brand">
          <span class="brand-mark">{_icon("shield")}</span>
          <div>
            <h1>ModelRouter</h1>
            <p>Local endpoint: <a href="{endpoint}">{endpoint}</a>
              <button class="copy-button" type="button" aria-label="Copy endpoint">
                {_icon("copy")}
              </button>
            </p>
          </div>
        </div>
        <div class="top-status" aria-label="Runtime status">
          <span>Status: <strong><i class="dot {proxy_dot}"></i> {escape(proxy_label)}</strong></span>
          <span>Mode: <strong class="accent" id="top-mode">{profile_label}</strong></span>
          <span>Telemetry: <strong><i class="dot {telemetry_dot}"></i> {telemetry_state}</strong></span>
          <button class="icon-button" type="button" onclick="postAction('/api/doctor')">
            {_icon("pulse")}<span>Live</span>
          </button>
          <a class="icon-button" href="/compact" aria-label="Open compact windowed mode">
            {_icon("open")}<span>Compact</span>
          </a>
          <button class="icon-only" type="button" aria-label="Toggle appearance">
            {_icon("moon")}
          </button>
          <button class="icon-only" type="button" onclick="jumpTo('settings')" aria-label="Settings">
            {_icon("gear")}
          </button>
        </div>
      </header>

      {_config_notice(config_error)}

      <section class="health-strip" aria-label="System health">
        <div class="check-grid">
          {health_checks}
        </div>
      </section>

      <div class="dashboard-grid">
        <div class="primary-column">
          <section class="panel flow-panel" aria-labelledby="flow-title">
            <h2 id="flow-title" class="sr-only">Route Flow</h2>
            {_route_flow(state)}
            <div class="profile-row">
              <div class="segmented" role="tablist" aria-label="Routing profile">
                {_profile_button("Fast", profile_value == "fast")}
                {_profile_button("Balanced", profile_value == "balanced")}
                {_profile_button("Quality", profile_value == "quality")}
                {_profile_button("Private", profile_value == "private")}
                {_profile_button("Safe", profile_value == "safe")}
              </div>
              <p class="profile-help">
                <strong>Fast</strong> = lowest latency, local-first;
                <strong>Balanced</strong> = default everyday routing;
                <strong>Quality</strong> = stronger local or hosted models allowed;
                <strong>Private</strong> = local-only, no hosted APIs;
                <strong>Safe</strong> = stricter human-confirmation gates.
              </p>
              <button class="button-blue" type="button" onclick="saveProfile()">Save mode</button>
            </div>
          </section>

          <section class="panel" id="routing-map" aria-labelledby="routing-title">
            <div class="panel-title">
              <h2 id="routing-title">Routing Map</h2>
              <button class="icon-only" type="button" aria-label="Refresh routing map">
                {_icon("refresh")}
              </button>
            </div>
            {_routing_map_table(state)}
          </section>

          <section class="panel" id="runtimes" aria-labelledby="runtimes-title">
            <div class="panel-title">
              <h2 id="runtimes-title">Providers / Runtimes</h2>
              <span class="muted">{escape(_runtime_panel_summary(state))}</span>
            </div>
            {_providers_runtime_section(state)}
          </section>

          {_model_library_panel(state)}

          {_settings_follow_through_panel(state)}

          <section class="panel" id="telemetry" aria-labelledby="telemetry-title">
            <div class="panel-title">
              <h2 id="telemetry-title">Recent Requests / Telemetry</h2>
              <button class="text-button" type="button" onclick="jumpTo('review')">Review queue {_icon("arrow-right")}</button>
            </div>
            {_recent_requests_table(state)}
          </section>
        </div>

        <aside class="inspector-column" aria-label="Inspectors">
          {_runtime_status_panel(state)}
          {_route_receipt_panel(state)}
          {_safety_panel()}
          {_maturity_panel(state)}
          {_benchmark_status_panel(state)}
          {_review_panel(state)}
          {_catalog_panel(state)}
          {_pricing_catalog_panel(state)}
        </aside>
      </div>
    </main>
  </div>

  <script>{_dashboard_js()}</script>
</body>
</html>"""


def render_compact_page(state: Mapping[str, Any]) -> str:
    """Render the standalone compact control panel/windowed mode."""

    proxy = state["proxy"]
    observability = state["observability"]
    proxy_process = state.get("proxy_process", {})
    endpoint = escape(str(proxy.get("endpoint") or "http://127.0.0.1:8082/v1"))
    profile_label = _profile_label(str(proxy.get("routing_profile") or "balanced"))
    telemetry_state = "On" if observability.get("enabled") else "Off"
    proxy_state = str(proxy_process.get("state") or "unknown").replace("_", " ").title()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ModelRouter Compact</title>
  <style>{_dashboard_css()}</style>
</head>
<body class="compact-body">
  {_compact_control_panel(state, endpoint, proxy_state, telemetry_state, profile_label)}
  <script>{_dashboard_js()}</script>
</body>
</html>"""


def _dashboard_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f4f6f9;
  --chrome: #f8fafc;
  --surface: #ffffff;
  --surface-soft: #f6f8fb;
  --surface-blue: #eef5ff;
  --text: #111827;
  --muted: #617086;
  --subtle: #8895a7;
  --line: #dce3ec;
  --line-soft: #edf1f6;
  --accent: #1f6feb;
  --accent-strong: #0b57d0;
  --green: #24a148;
  --green-bg: #eaf7ee;
  --yellow: #f5a400;
  --yellow-bg: #fff6dc;
  --red: #e5484d;
  --red-bg: #fff0f0;
  --shadow: 0 8px 28px rgba(15, 23, 42, .07);
  --tiny-shadow: 0 1px 2px rgba(15, 23, 42, .08);
  --radius: 8px;
  --radius-sm: 6px;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  min-width: 1180px;
  background: var(--bg);
  color: var(--text);
  font-size: 12px;
  line-height: 1.32;
  letter-spacing: 0;
}
button, input, select, textarea { font: inherit; letter-spacing: 0; }
button {
  border: 1px solid var(--line);
  background: var(--surface);
  color: #23324a;
  border-radius: var(--radius-sm);
  min-height: 30px;
  padding: 6px 10px;
  font-weight: 650;
  cursor: pointer;
}
button:hover { border-color: #c4cedb; background: #f9fbfd; }
button.icon-button, a.icon-button {
  border: 1px solid var(--line);
  background: var(--surface);
  border-radius: var(--radius-sm);
  min-height: 30px;
  padding: 6px 10px;
  font-weight: 650;
}
button.icon-button:hover, a.icon-button:hover {
  border-color: #c4cedb;
  background: #f9fbfd;
}
input, select, textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--text);
  min-height: 28px;
  padding: 5px 7px;
}
textarea {
  min-height: 44px;
  resize: vertical;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10.5px;
}
a { color: var(--accent); text-decoration: none; }
h1, h2, h3, p { margin: 0; }
h1 { font-size: 22px; line-height: 1.08; font-weight: 760; }
h2 { font-size: 15px; line-height: 1.18; font-weight: 730; }
h3 { font-size: 12px; color: var(--muted); font-weight: 680; }
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 170px minmax(1010px, 1fr);
}
.sidebar {
  background: linear-gradient(90deg, #f8fafc 0%, #f5f7fa 100%);
  border-right: 1px solid var(--line);
  padding: 18px 10px;
}
.traffic-lights { display: flex; gap: 8px; margin: 2px 0 30px 10px; }
.traffic-lights span {
  width: 13px;
  height: 13px;
  border-radius: 999px;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .08);
}
.traffic-lights .red { background: #ff5f57; }
.traffic-lights .yellow { background: #febc2e; }
.traffic-lights .green { background: #28c840; }
.side-nav { display: grid; gap: 4px; }
.side-link {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 9px;
  color: #30415b;
  text-align: left;
  border: 0;
  background: transparent;
  border-radius: 7px;
  min-height: 37px;
  padding: 8px 9px;
}
.side-link.active, .side-link:hover {
  background: #eaf2ff;
  color: var(--accent-strong);
}
.sidebar-note {
  color: var(--muted);
  font-size: 12px;
  margin: 24px 10px 0;
}
.workspace { min-width: 0; padding-bottom: 32px; }
.topbar {
  min-height: 62px;
  background: rgba(255, 255, 255, .86);
  backdrop-filter: blur(18px);
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 9px 18px 8px 18px;
  position: sticky;
  top: 0;
  z-index: 10;
}
.brand { display: flex; align-items: center; gap: 13px; }
.brand p { margin-top: 5px; color: #34435a; }
.brand-mark {
  width: 31px;
  height: 31px;
  display: grid;
  place-items: center;
  color: var(--accent);
}
.copy-button {
  min-height: 18px;
  padding: 0;
  border: 0;
  color: #5a6a83;
  background: transparent;
  vertical-align: middle;
}
.top-status {
  display: flex;
  align-items: center;
  gap: 16px;
  color: #26344b;
  white-space: nowrap;
}
.top-status > span:not(:last-of-type) {
  border-right: 1px solid var(--line);
  padding-right: 16px;
}
.accent { color: var(--accent-strong); }
.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 999px;
  margin-right: 6px;
}
.green-dot { background: var(--green); }
.yellow-dot { background: var(--yellow); }
.red-dot { background: var(--red); }
.icon { width: 18px; height: 18px; stroke: currentColor; stroke-width: 1.8; }
.icon-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #25344c;
}
.icon-only {
  width: 32px;
  height: 32px;
  padding: 0;
  display: inline-grid;
  place-items: center;
}
button.icon-only, a.icon-only {
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: #25344c;
}
button.icon-only:hover, a.icon-only:hover {
  border-color: #c4cedb;
  background: #f9fbfd;
}
.health-strip { padding: 10px 14px 0 14px; }
.health-title { font-weight: 720; }
.health-title.ok { color: #1f6b35; }
.health-title.danger { color: var(--red); }
.health-meta { color: var(--muted); }
.check-grid {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.check-item {
  display: flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--line-soft);
  border-radius: 6px;
  min-height: 24px;
  padding: 4px 8px;
  color: #2d3b52;
  background: #fff;
  box-shadow: var(--tiny-shadow);
  font-size: 11px;
  font-weight: 690;
}
.dashboard-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(330px, 360px);
  gap: 12px;
  padding: 12px 12px 0 12px;
}
.primary-column, .inspector-column { min-width: 0; }
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--tiny-shadow);
  margin-bottom: 12px;
  overflow: hidden;
}
.panel-title {
  min-height: 38px;
  padding: 9px 14px;
  border-bottom: 1px solid var(--line-soft);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.muted { color: var(--muted); }
.flow-panel { padding: 8px 12px; }
.flow {
  display: grid;
  grid-template-columns: minmax(115px, 1fr) 34px minmax(160px, 1.35fr) 34px
    minmax(150px, 1.25fr) 34px minmax(160px, 1.45fr) 34px minmax(120px, 1fr);
  align-items: center;
  gap: 4px;
}
.flow-node {
  min-height: 50px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: #fff;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  box-shadow: 0 4px 12px rgba(15, 23, 42, .04);
}
.flow-icon {
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: 7px;
  color: #1f2937;
  background: #f7f9fc;
  border: 1px solid var(--line-soft);
}
.flow-node.router .flow-icon,
.flow-node.selected .flow-icon {
  color: var(--accent);
  background: #eef5ff;
}
.flow-kicker { font-size: 10.5px; color: var(--muted); margin-top: 2px; }
.flow-arrow { color: #9aa5b5; display: grid; place-items: center; }
.profile-row {
  display: grid;
  grid-template-columns: minmax(380px, 44%) 1fr auto;
  gap: 14px;
  align-items: center;
  margin-top: 8px;
}
.segmented {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #fff;
  overflow: hidden;
  height: 32px;
}
.segment {
  border: 0;
  border-right: 1px solid var(--line-soft);
  border-radius: 0;
  background: transparent;
  color: #1f2937;
  min-height: 30px;
}
.segment:last-child { border-right: 0; }
.segment.active {
  color: #fff;
  background: linear-gradient(180deg, #367ff2 0%, #1f6feb 100%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .22);
}
.profile-help { color: #2f3d53; font-size: 12px; }
.data-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
.data-table th, .data-table td {
  border-bottom: 1px solid var(--line-soft);
  padding: 5px 8px;
  text-align: left;
  vertical-align: middle;
  line-height: 1.12;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: clip;
}
.data-table td:first-child,
.data-table td:nth-child(3),
.data-table td:nth-child(4) {
  white-space: normal;
}
.data-table th {
  color: #5f6e83;
  font-size: 10px;
  font-weight: 720;
  background: #fbfcfe;
}
.data-table tr:last-child td { border-bottom: 0; }
.route-cell, .provider-cell {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 0;
}
.row-icon {
  width: 18px;
  height: 18px;
  display: inline-grid;
  place-items: center;
  color: #41516a;
}
.selected-row td {
  background: #eaf3ff;
  border-top: 1px solid #d5e8ff;
  border-bottom-color: #d5e8ff;
}
.selected-row td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
.code {
  color: #24415f;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10.5px;
  overflow-wrap: anywhere;
}
.linkish { color: var(--accent-strong); font-weight: 680; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 18px;
  padding: 2px 6px;
  border-radius: 5px;
  font-size: 10px;
  font-weight: 690;
  white-space: nowrap;
}
.pill.green { background: var(--green-bg); color: #176b34; }
.pill.blue { background: #e8f1ff; color: #1557b0; }
.pill.yellow { background: var(--yellow-bg); color: #895d00; }
.pill.red { background: var(--red-bg); color: #c9272f; }
.pill.gray { background: #eef2f7; color: #536174; }
.runtime-grid {
  display: grid;
  grid-template-columns: 290px 1fr;
  min-height: 210px;
}
.provider-list {
  border-right: 1px solid var(--line);
  background: #fbfcfe;
  padding: 7px;
}
.provider-row {
  display: grid;
  grid-template-columns: 20px 82px 78px 1fr;
  gap: 7px;
  align-items: center;
  border-radius: 6px;
  min-height: 27px;
  padding: 4px 7px;
  color: #26344b;
}
.provider-row.active { background: #eaf3ff; color: var(--accent-strong); }
.provider-row strong { white-space: nowrap; }
.provider-status {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  color: var(--muted);
  white-space: nowrap;
}
.provider-row .muted { font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.runtime-detail {
  padding: 14px 18px 12px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 24px;
  row-gap: 9px;
}
.detail-field label {
  display: block;
  color: #59687d;
  font-size: 11px;
  font-weight: 720;
  margin-bottom: 4px;
}
.detail-field span { color: #172033; }
.detail-wide { grid-column: 1 / -1; }
.runtime-actions {
  grid-column: 1 / -1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
}
.action-group { display: flex; gap: 10px; }
.button-blue { color: var(--accent-strong); }
.button-red { color: #d92d37; }
.danger-text { color: var(--red); }
.catalog-output {
  max-height: 180px;
  overflow: auto;
  margin: 12px 18px 18px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface-soft);
  color: #34435a;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  white-space: pre-wrap;
}
.settings-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  padding: 14px;
}
.settings-grid label,
.backend-editor label,
.review-form label {
  display: grid;
  gap: 4px;
  color: #59687d;
  font-size: 10.5px;
  font-weight: 720;
}
.backend-editor {
  padding: 0 14px 14px;
}
.backend-editor .data-table input,
.backend-editor .data-table select,
.backend-editor .data-table textarea {
  min-width: 0;
}
.settings-actions {
  padding: 0 14px 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  align-items: center;
}
.compact-actions {
  padding: 0;
}
.model-ops-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  padding: 0 14px 10px;
}
.model-op-row {
  min-width: 0;
  border: 1px solid var(--line-soft);
  border-radius: 7px;
  background: #fbfcfe;
  padding: 8px 9px;
  display: grid;
  gap: 5px;
}
.model-op-heading {
  min-width: 0;
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
}
.model-op-title {
  color: #59687d;
  font-size: 10px;
  font-weight: 760;
  text-transform: uppercase;
}
.model-op-meta {
  color: #617086;
  font-size: 10px;
  white-space: nowrap;
}
.model-op-main {
  color: #172033;
  font-size: 11px;
  line-height: 1.2;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.model-op-detail {
  color: #5d6b80;
  font-size: 10.5px;
  line-height: 1.25;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.model-op-actions {
  display: flex;
  gap: 6px;
  align-items: center;
}
.model-op-actions button {
  min-height: 24px;
  padding: 3px 7px;
  font-size: 10px;
}
.models-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 7px;
  padding: 0 14px 14px;
}
.model-card {
  border: 1px solid var(--line-soft);
  border-radius: 7px;
  background: #fbfcfe;
  overflow: hidden;
}
.model-card summary {
  min-height: 38px;
  padding: 8px 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  cursor: pointer;
  color: #273750;
  list-style: none;
}
.model-card summary::-webkit-details-marker { display: none; }
.model-card summary::after {
  content: "Expand";
  color: var(--accent-strong);
  font-size: 10px;
  font-weight: 720;
}
.model-card[open] summary {
  border-bottom: 1px solid var(--line-soft);
}
.model-card[open] summary::after { content: "Collapse"; }
.model-detail-title {
  font-size: 12px;
  font-weight: 740;
}
.model-detail-body {
  max-height: 260px;
  overflow: auto;
}
.model-card-wide {
  grid-column: 1 / -1;
}
.model-table th,
.model-table td {
  white-space: normal;
}
.review-list {
  padding: 12px 14px;
  display: grid;
  gap: 9px;
}
.review-item {
  border: 1px solid var(--line-soft);
  border-radius: 7px;
  padding: 8px;
  background: #fbfcfe;
}
.review-form {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}
.review-form label:last-of-type { grid-column: 1 / -1; }
.inspector-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--tiny-shadow);
  margin-bottom: 14px;
  overflow: hidden;
}
.inspector-card header {
  min-height: 50px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line-soft);
}
.receipt-body { padding: 14px 16px 16px; }
.runtime-status-body {
  padding: 12px 14px 14px;
  display: grid;
  gap: 10px;
}
.runtime-status-active {
  border: 1px solid var(--line-soft);
  border-radius: 7px;
  background: #fbfcfe;
  padding: 8px 9px;
  display: grid;
  gap: 6px;
}
.runtime-status-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}
.runtime-status-heading strong,
.runtime-row-main strong {
  color: #172033;
}
.runtime-status-meta,
.runtime-row-meta {
  color: #5d6b80;
  font-size: 10.5px;
  line-height: 1.25;
}
.runtime-next-action {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: #fff;
  color: var(--accent-strong);
  font-size: 10px;
  font-weight: 720;
  min-height: 22px;
  padding: 2px 6px;
  white-space: nowrap;
}
.runtime-status-details {
  border-top: 1px solid var(--line-soft);
  padding-top: 8px;
}
.runtime-status-details summary {
  cursor: pointer;
  color: var(--accent-strong);
  font-weight: 720;
  list-style: none;
}
.runtime-status-details summary::-webkit-details-marker { display: none; }
.runtime-status-details summary::after {
  content: "Expand";
  float: right;
  color: var(--muted);
  font-size: 10px;
}
.runtime-status-details[open] summary::after { content: "Collapse"; }
.runtime-status-list {
  display: grid;
  gap: 7px;
  margin-top: 8px;
}
.runtime-status-row {
  border: 1px solid var(--line-soft);
  border-radius: 6px;
  background: #fff;
  padding: 7px;
  display: grid;
  gap: 4px;
}
.runtime-row-main {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}
.receipt-summary {
  margin: 0 0 12px;
  color: #1d2b3f;
  font-size: 13px;
  line-height: 1.4;
}
.receipt-grid {
  display: grid;
  grid-template-columns: 122px 1fr;
  gap: 10px 12px;
  align-items: center;
}
.receipt-grid dt { color: #4e5d72; font-weight: 690; }
.receipt-grid dd { margin: 0; color: #111827; }
.receipt-divider { height: 1px; background: var(--line); margin: 15px 0; }
.rationale { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 9px; }
.receipt-button {
  width: 100%;
  justify-content: space-between;
  display: flex;
  align-items: center;
  margin-top: 14px;
}
.safety-list { padding: 14px 16px 12px; display: grid; gap: 10px; }
.toggle-row { display: flex; align-items: center; gap: 9px; }
.switch {
  position: relative;
  width: 32px;
  height: 18px;
  flex: 0 0 auto;
}
.switch input { position: absolute; opacity: 0; }
.slider {
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background: #cbd5e1;
}
.slider::after {
  content: "";
  position: absolute;
  width: 14px;
  height: 14px;
  top: 2px;
  left: 2px;
  border-radius: 999px;
  background: #fff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .24);
  transition: transform .16s ease;
}
.switch input:checked + .slider { background: #2f73e0; }
.switch input:checked + .slider::after { transform: translateX(14px); }
.protected-note {
  margin: 10px 16px 14px;
  border: 1px solid var(--line);
  background: #f8fafc;
  border-radius: 6px;
  color: #59687d;
  min-height: 34px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
}
.text-button {
  border: 0;
  background: transparent;
  color: var(--accent-strong);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0;
}
.compact-body {
  min-width: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 18px;
  background: #eef2f7;
}
.compact-window {
  width: min(360px, calc(100vw - 24px));
  background: rgba(255, 255, 255, .96);
  border: 1px solid #cfd7e3;
  border-radius: 8px;
  box-shadow: 0 24px 70px rgba(15, 23, 42, .23);
  overflow: hidden;
}
.compact-header {
  height: 34px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 8px 0 10px;
  border-bottom: 1px solid var(--line-soft);
}
.compact-title { display: flex; align-items: center; gap: 8px; font-weight: 760; }
.compact-title .traffic-lights { margin: 0; }
.compact-body-content { padding: 8px 9px 9px; }
.compact-chips, .compact-bottom {
  display: flex;
  gap: 6px;
}
.compact-chips { flex-wrap: nowrap; }
.compact-chip {
  min-height: 20px;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 3px 6px;
  background: #fbfcfe;
  font-size: 8.5px;
  white-space: nowrap;
}
.compact-flow {
  display: grid;
  grid-template-columns: 46px 10px 62px 10px 54px 10px 54px;
  align-items: center;
  gap: 2px;
  margin: 8px 0;
  color: #526178;
}
.compact-box {
  border: 1px solid var(--line);
  border-radius: 5px;
  min-height: 24px;
  display: grid;
  place-items: center;
  background: #fff;
  font-size: 9px;
  font-weight: 700;
  color: #24344c;
}
.compact-box.selected { color: var(--accent-strong); background: #eef5ff; }
.compact-summary {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 10px;
  font-size: 9.5px;
}
.compact-summary div {
  min-width: 0;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.compact-summary span {
  color: #5d6b80;
  font-weight: 680;
  white-space: nowrap;
}
.compact-summary strong { text-align: right; }
.compact-summary span,
.compact-summary strong {
  white-space: nowrap;
}
.compact-recent {
  margin-top: 8px;
  border-top: 1px solid var(--line-soft);
  padding-top: 6px;
}
.compact-recent-row {
  display: grid;
  grid-template-columns: 34px 1fr 54px 36px;
  gap: 6px;
  align-items: center;
  min-height: 18px;
  font-size: 9px;
}
.compact-actions-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 5px;
  margin-top: 8px;
}
.compact-actions-grid a,
.compact-actions-grid button {
  min-height: 34px;
  padding: 3px;
  display: grid;
  place-items: center;
  gap: 2px;
  color: #2d3b52;
  font-size: 9px;
}
.compact-bottom {
  border-top: 1px solid var(--line-soft);
  padding: 6px 8px;
  background: #fbfcfe;
  flex-wrap: nowrap;
}
.compact-bottom .compact-chip { background: #fff; }
.status-line { display: inline-flex; align-items: center; gap: 5px; }
@media (max-width: 1180px) {
  body { min-width: 0; }
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { display: none; }
  .topbar { position: static; align-items: flex-start; flex-direction: column; }
  .top-status { flex-wrap: wrap; }
  .flow { grid-template-columns: 1fr; }
  .flow-arrow { transform: rotate(90deg); min-height: 22px; }
  .profile-row, .runtime-grid, .settings-grid, .review-form, .models-grid, .model-ops-strip { grid-template-columns: 1fr; }
  .provider-list { border-right: 0; border-bottom: 1px solid var(--line); }
}
@media (max-width: 1500px) {
  .dashboard-grid { grid-template-columns: 1fr; padding-right: 16px; }
}
"""


def _dashboard_js() -> str:
    return """
async function postAction(path, payload = {}) {
  const target = document.getElementById('last-action');
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (target) {
      target.textContent = data.ok === false ? data.error : 'Action complete';
    }
    return data;
  } catch (error) {
    if (target) {
      target.textContent = 'Action failed: ' + error;
    }
    return {ok: false, error: String(error)};
  }
}
async function saveProfile() {
  const active = document.querySelector('.segment.active');
  const profile = active ? active.dataset.profileValue : 'balanced';
  await postAction('/api/save-config', {confirm: true, proxy: {routing_profile: profile}});
}
function backendPayload() {
  const backends = {};
  document.querySelectorAll('[data-backend]').forEach(row => {
    const name = row.dataset.backend;
    backends[name] = {
      model: row.querySelector('[data-field="model"]').value,
      base_url: row.querySelector('[data-field="base_url"]').value,
      runtime: {
        enabled: row.querySelector('[data-field="runtime_enabled"]').value === 'true',
        kind: row.querySelector('[data-field="runtime_kind"]').value,
        command: row.querySelector('[data-field="runtime_command"]').value,
        readiness_url: row.querySelector('[data-field="readiness_url"]').value,
        idle_timeout_seconds: row.querySelector('[data-field="idle_timeout_seconds"]').value,
        log_path: row.querySelector('[data-field="log_path"]').value
      }
    };
  });
  return backends;
}
async function saveConfig() {
  const payload = {
    proxy: {
      host: document.getElementById('proxy-host').value,
      port: document.getElementById('proxy-port').value,
      routing_mode: document.getElementById('routing-mode').value,
      default_backend: document.getElementById('default-backend').value,
      default_model: document.getElementById('default-model').value,
      respect_client_model: document.getElementById('respect-client-model').value === 'true',
      unknown_model_behavior: document.getElementById('unknown-model-behavior').value,
      safety_gate_mode: document.getElementById('safety-gate-mode').value
    },
    observability: {
      enabled: document.getElementById('observability-enabled').value === 'true',
      prompt_capture: document.getElementById('prompt-capture').value,
      log_path: document.getElementById('observability-log').value
    },
    backend_policy: {
      backend_allowlist: document.getElementById('backend-allowlist').value,
      backend_denylist: document.getElementById('backend-denylist').value
    },
    backends: backendPayload()
  };
  payload.confirm = true;
  await postAction('/api/save-config', payload);
}
async function applyPreset() {
  const preset = document.getElementById('preset').value;
  if (!window.confirm('Replace current config with the ' + preset + ' preset?')) return;
  await postAction('/api/save-config', {confirm: true, apply_preset: true, preset: preset});
}
async function sendFeedback() {
  const outcome = document.getElementById('feedback-outcome');
  await postAction('/api/feedback', {
    confirm: true,
    request_id: document.getElementById('feedback-request-id').value,
    expected_engine: document.getElementById('feedback-engine').value,
    outcome_label: outcome ? outcome.value : '',
    notes: document.getElementById('feedback-notes').value
  });
}
async function runDownload(route, repoId) {
  if (!window.confirm('Download ' + repoId + ' for ' + route + '?')) return;
  await postAction('/api/download/run', {confirm: true, route: route, repo_id: repoId});
}
async function planDownload(route, repoId) {
  const data = await postAction('/api/download/plan', {route: route, repo_id: repoId});
  showModelAction(data);
}
async function assignRoute(routeId, backend, inputId) {
  const input = document.getElementById(inputId);
  const model = input ? input.value.trim() : '';
  if (!model) {
    const target = document.getElementById('model-library-output');
    if (target) target.textContent = 'Choose a model before saving an assignment.';
    return;
  }
  const data = await postAction('/api/model/assign-route', {
    confirm: true,
    route_id: routeId,
    backend: backend,
    model: model
  });
  showModelAction(data);
}
function showModelAction(data) {
  const target = document.getElementById('model-library-output');
  if (target) target.textContent = JSON.stringify(data, null, 2);
  return data;
}
async function runBenchmark() {
  if (!window.confirm('Run local backend benchmark requests with a fixed synthetic prompt?')) return;
  await postAction('/api/benchmark/run', {confirm: true});
}
async function scanModels() {
  const data = await postAction('/api/scan');
  const target = document.getElementById('scan-output');
  if (target) target.textContent = JSON.stringify(data.recommendation || data, null, 2);
  showModelAction(data);
}
async function planBenchmark() {
  const data = await postAction('/api/benchmark/plan');
  const target = document.getElementById('benchmark-output');
  if (target) target.textContent = JSON.stringify(data.targets || data, null, 2);
}
async function runtimeAction(actionId, backend, model = '', confirm = false) {
  const mutates = actionId === 'runtime.start_server' ||
    actionId === 'runtime.stop_server' ||
    actionId === 'runtime.load_model' ||
    actionId === 'runtime.unload_model';
  if (mutates && !window.confirm('Run ' + actionId + ' for backend ' + backend + '?')) {
    return;
  }
  const payload = {action_id: actionId, backend: backend};
  if (model) payload.model = model;
  if (confirm || mutates) payload.confirm = true;
  const output = document.getElementById('runtime-output');
  if (output) output.textContent = 'Running ' + actionId + '...';
  const data = await postAction('/api/action', payload);
  if (output) output.textContent = JSON.stringify(data.payload || data, null, 2);
}
async function copyText(text) {
  if (!text) return;
  try {
    if (navigator.clipboard) await navigator.clipboard.writeText(text);
  } catch (error) {
    // Copy support can be blocked in some local browser contexts; keep labeling usable.
  }
  const target = document.getElementById('last-action');
  const label = text.length > 80 ? 'text block' : text;
  if (target) target.textContent = 'Copied ' + label;
}
async function sendReviewFeedback(requestId) {
  const expected = document.getElementById('expected-' + requestId);
  const notes = document.getElementById('notes-' + requestId);
  const outcome = document.getElementById('outcome-' + requestId);
  if (!expected || !expected.value) return;
  await postAction('/api/feedback', {
    confirm: true,
    request_id: requestId,
    expected_engine: expected.value,
    outcome_label: outcome ? outcome.value : '',
    notes: notes ? notes.value : ''
  });
}
async function showCatalogDiff() {
  const output = document.getElementById('catalog-output');
  if (output) output.textContent = 'Loading catalog diff...';
  const data = await postAction('/api/catalog/diff');
  if (output) output.textContent = JSON.stringify(data.diff || data, null, 2);
}
async function applyCatalogUpdate() {
  if (!window.confirm('Apply packaged catalog defaults to the local config? A backup and migration log entry will be written when changes apply.')) {
    return;
  }
  const output = document.getElementById('catalog-output');
  if (output) output.textContent = 'Applying catalog update...';
  const data = await postAction('/api/catalog/apply', {confirm: true});
  if (output) output.textContent = JSON.stringify(data.result || data, null, 2);
}
async function showPricingStatus() {
  const output = document.getElementById('pricing-output');
  if (output) output.textContent = 'Loading pricing catalog status...';
  const data = await postAction('/api/pricing/status');
  if (output) output.textContent = JSON.stringify(data.status || data, null, 2);
}
async function showPricingDiff() {
  const output = document.getElementById('pricing-output');
  if (output) output.textContent = 'Loading pricing catalog diff...';
  const data = await postAction('/api/pricing/diff');
  if (output) output.textContent = JSON.stringify(data.diff || data, null, 2);
}
async function applyPricingCatalog() {
  if (!window.confirm('Write packaged pricing metadata to the local override? Verify provider prices before using estimates for spend review.')) {
    return;
  }
  const output = document.getElementById('pricing-output');
  if (output) output.textContent = 'Applying pricing catalog metadata...';
  const data = await postAction('/api/pricing/apply', {confirm: true});
  if (output) output.textContent = JSON.stringify(data.result || data, null, 2);
}
function jumpTo(id) {
  const target = document.getElementById(id) || document.getElementById('dashboard');
  target.scrollIntoView({behavior: 'smooth', block: 'start'});
}
document.querySelectorAll('.segment').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.segment').forEach((item) => {
      item.classList.remove('active');
      item.setAttribute('aria-selected', 'false');
    });
    button.classList.add('active');
    button.setAttribute('aria-selected', 'true');
    const mode = button.dataset.profile || button.textContent.trim();
    const topMode = document.getElementById('top-mode');
    const compactMode = document.getElementById('compact-mode');
    if (topMode) topMode.textContent = mode;
    if (compactMode) compactMode.textContent = mode;
  });
});
document.querySelectorAll('[data-feedback]').forEach((button) => {
  button.addEventListener('click', () => {
    const requestId = button.getAttribute('data-feedback');
    const input = document.getElementById('feedback-request-id');
    if (input) input.value = requestId;
    jumpTo('review');
    const target = document.getElementById('last-action');
    if (target) target.textContent = 'Ready to label ' + requestId + '.';
  });
});
"""


def _dashboard_health_checks(state: Mapping[str, Any]) -> str:
    proxy_process = state.get("proxy_process", {})
    discovery = state.get("discovery", {})
    commands = discovery.get("commands") if isinstance(discovery, dict) else {}
    commands = commands if isinstance(commands, dict) else {}
    backends = state.get("backends") if isinstance(state.get("backends"), list) else []
    route_ids = {
        str(row.get("route_id"))
        for row in state.get("route_map", [])
        if isinstance(row, dict)
    }
    has_llama = bool(commands.get("llama-server")) or any(
        isinstance(backend, dict)
        and backend.get("runtime", {}).get("kind") == "llama-server"
        for backend in backends
    )
    has_lmstudio = any(
        isinstance(backend, dict) and ":1234" in str(backend.get("base_url", ""))
        for backend in backends
    )
    has_ollama = bool(commands.get("ollama")) or any(
        isinstance(backend, dict) and ":11434" in str(backend.get("base_url", ""))
        for backend in backends
    )
    checks = (
        ("Proxy running", str(proxy_process.get("state")) == "running"),
        ("Config valid", bool(state.get("config_valid"))),
        ("Observability configured", bool(state.get("observability", {}).get("enabled"))),
        ("llama.cpp configured", has_llama),
        ("LM Studio configured", has_lmstudio),
        ("Ollama configured", has_ollama),
        ("Human confirm route", "human_confirm" in route_ids),
    )
    return "\n".join(
        f'<span class="check-item"><i class="dot {"green-dot" if ok else "yellow-dot"}"></i>'
        f"{escape(label)}</span>"
        for label, ok in checks
    )


def _sidebar_item(label: str, class_name: str, target: str) -> str:
    return (
        f'<button class="side-link {class_name}" type="button" '
        f'onclick="jumpTo({json.dumps(target)})">'
        f'{_nav_icon(label)}<span>{escape(label)}</span></button>'
    )


def _nav_icon(label: str) -> str:
    names = {
        "Overview": "overview",
        "Routing": "routing",
        "Runtimes": "runtime",
        "Providers": "providers",
        "Safety": "shield",
        "Telemetry": "pulse",
        "Settings": "gear",
        "llama.cpp": "code",
        "Ollama": "shield",
        "Risky": "pulse",
        "Research": "server",
        "CloseAI": "providers",
        "Codings": "code",
    }
    return _icon(names.get(label, "overview"))


def _profile_button(label: str, active: bool) -> str:
    value = label.lower().replace(" ", "_")
    return (
        f'<button class="segment {"active" if active else ""}" type="button" '
        f'role="tab" aria-selected="{str(active).lower()}" '
        f'data-profile="{escape(label)}" '
        f'data-profile-value="{escape(value)}">{escape(label)}</button>'
    )


def _profile_label(value: str) -> str:
    normalized = value if value in ROUTING_PROFILE_VALUES else "balanced"
    return normalized.replace("_", " ").title()


def _route_flow(state: Mapping[str, Any]) -> str:
    receipt = state.get("route_receipt", {})
    selected = str(receipt.get("selected") or "none yet")
    backend = str(receipt.get("backend") or "none yet")
    response = (
        "Stream / JSON"
        if receipt.get("has_event")
        else "waiting for request"
    )
    nodes = (
        ("Request", "Incoming", "request", ""),
        ("ModelRouter", "Classify & Route", "shield", "router"),
        ("Selected Engine", selected, "puzzle", "selected"),
        ("Backend Runtime", backend, "server", ""),
        ("Response", response, "response", ""),
    )
    parts: list[str] = ['<div class="flow">']
    for index, (title, subtitle, icon_name, class_name) in enumerate(nodes):
        parts.append(
            f'<div class="flow-node {class_name}">'
            f'<span class="flow-icon">{_icon(icon_name)}</span>'
            f'<div><strong>{escape(title)}</strong>'
            f'<p class="flow-kicker">{escape(subtitle)}</p></div></div>'
        )
        if index < len(nodes) - 1:
            parts.append(f'<span class="flow-arrow">{_icon("arrow-right")}</span>')
    parts.append("</div>")
    return "\n".join(parts)


def _routing_map_table(state: Mapping[str, Any]) -> str:
    rows = state.get("route_map") if isinstance(state.get("route_map"), list) else []
    body = "\n".join(_routing_map_row(row) for row in rows if isinstance(row, dict))
    if not body:
        body = '<tr><td colspan="9" class="muted">No valid routing config loaded.</td></tr>'
    return f"""<table class="data-table">
      <colgroup>
        <col style="width: 10%">
        <col style="width: 14%">
        <col style="width: 22%">
        <col style="width: 14%">
        <col style="width: 8%">
        <col style="width: 8%">
        <col style="width: 9%">
        <col style="width: 6%">
        <col style="width: 9%">
      </colgroup>
      <thead>
        <tr>
          <th>Route class</th>
          <th>Route ID</th>
          <th>Target / Description</th>
          <th>Provider / Runtime</th>
          <th>Latency</th>
          <th>Cost</th>
          <th>Privacy</th>
          <th>Tools</th>
          <th>Fallback</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>"""


def _routing_map_row(row: Mapping[str, Any]) -> str:
    route_class = str(row.get("route_class") or "")
    route_id = str(row.get("route_id") or "")
    target = str(row.get("target") or "")
    provider = str(row.get("provider") or "")
    latency = str(row.get("latency") or "")
    cost = str(row.get("cost") or "")
    privacy = str(row.get("privacy") or "")
    tools = str(row.get("tools") or "")
    fallback = str(row.get("fallback") or "—")
    policy_status = str(row.get("policy_status") or "allowed")
    class_name = "selected-row" if row.get("selected") else ""
    latency_class = (
        "red"
        if latency in {"High", "Hosted"}
        else "yellow"
        if latency in {"Medium", "On demand", "Unmeasured"}
        else "green"
    )
    privacy_class = "yellow" if privacy in {"Mixed", "Local or Hosted", "Hosted"} else "green"
    tools_class = "blue" if tools == "Yes" else "gray"
    cost_class = "yellow" if "Medium" in cost or "Hosted" in cost else "green"
    target_text = target if policy_status == "allowed" else f"{target} ({policy_status})"
    return f"""<tr class="{class_name}">
      <td><span class="route-cell"><span class="row-icon">{_icon(_route_icon_name(route_class))}</span><strong>{escape(route_class)}</strong></span></td>
      <td><span class="code {'linkish' if route_id == 'code_agent' else ''}">{escape(route_id)}</span></td>
      <td>{escape(target_text)}</td>
      <td><span class="provider-cell">{_provider_glyph(provider)}<span>{escape(provider)}</span></span></td>
      <td><span class="pill {latency_class}">{escape(latency)}</span></td>
      <td><span class="pill {cost_class}">{escape(cost)}</span></td>
      <td><span class="pill {privacy_class}">{escape(privacy)}</span></td>
      <td><span class="pill {tools_class}">{escape(tools)}</span></td>
      <td><span class="code {'linkish' if fallback == 'reasoning_local' else ''}">{escape(fallback)}</span></td>
    </tr>"""


def _route_icon_name(route_class: str) -> str:
    return {
        "Simple": "simple",
        "Balanced": "balanced",
        "Reasoning": "reasoning",
        "Coding": "code",
        "Research": "database",
        "Vision": "vision",
        "Image generation": "image",
        "Risky actions": "shield",
    }.get(route_class, "routing")


def _provider_glyph(provider: str) -> str:
    icon_name = "server"
    if "llama" in provider:
        icon_name = "code"
    elif "LM Studio" in provider:
        icon_name = "cube"
    elif "Ollama" in provider:
        icon_name = "runtime"
    elif "MLX" in provider:
        icon_name = "chip"
    elif "RAG" in provider:
        icon_name = "database"
    elif "Diffusion" in provider:
        icon_name = "image"
    elif "Safety" in provider:
        icon_name = "shield"
    return f'<span class="row-icon">{_icon(icon_name)}</span>'


def _runtime_status_panel(state: Mapping[str, Any]) -> str:
    provider_state = state.get("provider_runtime", {})
    if not isinstance(provider_state, Mapping):
        provider_state = {}
    providers = [
        item
        for item in provider_state.get("providers", [])
        if isinstance(item, Mapping)
    ]
    detail = (
        provider_state.get("detail")
        if isinstance(provider_state.get("detail"), Mapping)
        else {}
    )
    active = (
        detail
        if detail
        else (providers[0].get("runtime_adapter", {}) if providers else {})
    )
    active_backend = str(
        (detail.get("backend") if isinstance(detail, Mapping) else "")
        or provider_state.get("selected_backend")
        or "none"
    )
    active_name = str(
        (detail.get("adapter_provider") if isinstance(detail, Mapping) else "")
        or _runtime_item_adapter(active).get("provider")
        or "unconfigured"
    )
    active_health = str(
        (detail.get("health_status") if isinstance(detail, Mapping) else "")
        or _runtime_item_health(active).get("status")
        or "unknown"
    )
    active_detected = _detected_label(
        detail.get("detected")
        if isinstance(detail, Mapping)
        else active.get("detected")
    )
    active_action = _runtime_next_action(detail if isinstance(detail, Mapping) else active)
    active_capabilities = _runtime_capability_summary(
        detail.get("capabilities")
        if isinstance(detail, Mapping)
        else active.get("capabilities")
    )
    hint = str(
        (detail.get("install_hint") if isinstance(detail, Mapping) else "")
        or (detail.get("missing_dependency") if isinstance(detail, Mapping) else "")
        or "Runtime detection is advisory; routing uses configured backend policy."
    )
    rows = "\n".join(_runtime_status_row(item) for item in providers)
    if not rows:
        rows = (
            '<div class="runtime-status-row">'
            '<span class="muted">No configured runtimes.</span></div>'
        )
    return f"""<section class="inspector-card runtime-status-card" aria-labelledby="runtime-status-title">
      <header>
        <h2 id="runtime-status-title">{_icon("runtime")} Runtime Status</h2>
        <button class="text-button" type="button" onclick="jumpTo('runtimes')">View details {_icon("arrow-right")}</button>
      </header>
      <div class="runtime-status-body">
        <div class="runtime-status-active">
          <div class="runtime-status-heading">
            <strong>{escape(active_name)} / {escape(active_backend)}</strong>
            <span class="runtime-next-action">{escape(active_action)}</span>
          </div>
          <div class="runtime-status-meta">
            {escape(active_detected)} · health {escape(active_health)} · {escape(active_capabilities)}
          </div>
          <div class="runtime-status-meta">{escape(hint)}</div>
        </div>
        <details class="runtime-status-details">
          <summary>Configured runtimes ({len(providers)})</summary>
          <div class="runtime-status-list">{rows}</div>
        </details>
      </div>
    </section>"""


def _runtime_status_row(item: Mapping[str, Any]) -> str:
    adapter = _runtime_item_adapter(item)
    health = _runtime_item_health(adapter)
    name = str(item.get("name") or item.get("backend") or adapter.get("provider") or "runtime")
    backend = str(item.get("backend") or "")
    detected = _detected_label(adapter.get("detected"))
    status = str(health.get("status") or item.get("status") or "unknown")
    action_input = dict(adapter)
    action_input["backend"] = backend
    action = _runtime_next_action(action_input)
    capabilities = _runtime_capability_summary(adapter.get("capabilities"))
    hint = str(adapter.get("install_hint") or adapter.get("missing_dependency") or "")
    hint_html = (
        f'<div class="runtime-row-meta">{escape(hint)}</div>' if hint else ""
    )
    dot = escape(
        str(item.get("dot") or _adapter_dot(adapter, active=False, managed=False))
    )
    return f"""<div class="runtime-status-row">
      <div class="runtime-row-main">
        <strong><i class="dot {dot}"></i>{escape(name)}</strong>
        <span class="runtime-next-action">{escape(action)}</span>
      </div>
      <div class="runtime-row-meta">{escape(detected)} · health {escape(status)} · {escape(capabilities)}</div>
      {hint_html}
    </div>"""


def _runtime_item_adapter(item: Mapping[str, Any]) -> Mapping[str, Any]:
    adapter = item.get("runtime_adapter") if isinstance(item.get("runtime_adapter"), Mapping) else item
    return adapter if isinstance(adapter, Mapping) else {}


def _runtime_item_health(item: Mapping[str, Any]) -> Mapping[str, Any]:
    health = item.get("health") if isinstance(item.get("health"), Mapping) else {}
    return health


def _detected_label(value: Any) -> str:
    if value is True:
        return "detected"
    if value is False:
        return "not detected"
    return "unknown"


def _runtime_next_action(item: Mapping[str, Any]) -> str:
    health = _runtime_item_health(item)
    health_status = str(item.get("health_status") or health.get("status") or "").lower()
    capabilities = (
        item.get("capabilities") if isinstance(item.get("capabilities"), Mapping) else {}
    )
    start = (
        capabilities.get("start_server")
        if isinstance(capabilities.get("start_server"), Mapping)
        else {}
    )
    if item.get("install_hint") or item.get("missing_dependency"):
        return "install guide"
    if item.get("detected") is False:
        return "configure"
    if start.get("supported") is True and health_status in {
        "unreachable",
        "stopped",
        "error",
    }:
        return "start"
    if health_status in {"unreachable", "error"}:
        return "connect"
    if health_status in {"unsupported", "unknown", ""}:
        return "configure"
    return "view details"


def _runtime_capability_summary(value: Any) -> str:
    capabilities = value if isinstance(value, Mapping) else {}
    labels = (
        ("discover_models", "models"),
        ("list_loaded_models", "loaded"),
        ("load_model", "load"),
        ("unload_model", "unload"),
        ("logs", "logs"),
    )
    supported = [
        label
        for key, label in labels
        if isinstance(capabilities.get(key), Mapping)
        and capabilities[key].get("supported") is True
    ]
    if supported:
        return "capabilities: " + ", ".join(supported[:3])
    disabled = [
        label
        for key, label in labels
        if isinstance(capabilities.get(key), Mapping)
        and capabilities[key].get("disabled_reason")
    ]
    if disabled:
        return "capabilities disabled: " + ", ".join(disabled[:2])
    return "capabilities unknown"


def _providers_runtime_section(state: Mapping[str, Any]) -> str:
    provider_state = state.get("provider_runtime", {})
    providers = provider_state.get("providers") if isinstance(provider_state, dict) else []
    detail = provider_state.get("detail") if isinstance(provider_state, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    provider_rows = "\n".join(
        _provider_row(
            str(item.get("name") or item.get("backend") or ""),
            str(item.get("status") or ""),
            str(item.get("detail") or ""),
            "active" if item.get("active") else "",
            str(item.get("dot") or "yellow-dot"),
        )
        for item in providers
        if isinstance(item, dict)
    )
    if not provider_rows:
        provider_rows = '<div class="muted">No backends configured.</div>'
    builder = detail.get("builder") if isinstance(detail.get("builder"), dict) else {}
    fallback_chain = detail.get("fallback_chain") if isinstance(detail.get("fallback_chain"), list) else []
    fallback_text = ", ".join(str(item) for item in fallback_chain) if fallback_chain else "none"
    capabilities = detail.get("capabilities") if isinstance(detail.get("capabilities"), dict) else {}
    load_support = capabilities.get("load_model") if isinstance(capabilities.get("load_model"), dict) else {}
    unload_support = capabilities.get("unload_model") if isinstance(capabilities.get("unload_model"), dict) else {}
    start_support = capabilities.get("start_server") if isinstance(capabilities.get("start_server"), dict) else {}
    stop_support = capabilities.get("stop_server") if isinstance(capabilities.get("stop_server"), dict) else {}
    discovered_models = detail.get("discovered_models") if isinstance(detail.get("discovered_models"), list) else []
    loaded_models = detail.get("loaded_models") if isinstance(detail.get("loaded_models"), list) else []
    logs = detail.get("logs") if isinstance(detail.get("logs"), dict) else {}
    log_paths = logs.get("paths") if isinstance(logs.get("paths"), list) else []
    backend_name = str(detail.get("backend") or "")
    model_id = str(detail.get("model") or "")
    backend_js = _js_string(backend_name)
    detected_models_text = _runtime_model_ids_text(discovered_models)
    return f"""<div class="runtime-grid" id="providers">
      <div class="provider-list">{provider_rows}</div>
      <div class="runtime-detail" id="runtime-detail">
        <div class="detail-field detail-wide">
          <label>Runtime command</label>
          <span class="code">{escape(str(detail.get("runtime_command") or "No backend selected."))}</span>
        </div>
        <div class="detail-field">
          <label>Backend</label>
          <span>{escape(str(detail.get("backend") or "none"))}</span>
        </div>
        <div class="detail-field">
          <label>Model</label>
          <span>{escape(str(detail.get("model") or "none"))}</span>
        </div>
        <div class="detail-field">
          <label>Runtime kind</label>
          <span>{escape(str(detail.get("runtime_kind") or "unmanaged"))}</span>
        </div>
        <div class="detail-field">
          <label>Detection</label>
          <span>{escape(_runtime_detection_label(detail))}</span>
        </div>
        <div class="detail-field">
          <label>Port</label>
          <span>{escape(str(builder.get("port") or _port_from_url(str(detail.get("base_url") or "")) or "n/a"))}</span>
        </div>
        <div class="detail-field">
          <label>Readiness URL</label>
          <span>{_runtime_link(detail.get("readiness_url"))}</span>
        </div>
        <div class="detail-field">
          <label>Idle timeout</label>
          <span>{escape(str(detail.get("idle_timeout") or "not managed"))}</span>
        </div>
        <div class="detail-field">
          <label>Context size</label>
          <span>{escape(str(builder.get("context_size") or "not set"))}</span>
        </div>
        <div class="detail-field">
          <label>GPU layers</label>
          <span>{escape(str(builder.get("gpu_layers") or "not set"))}</span>
        </div>
        <div class="detail-field">
          <label>Fallback chain</label>
          <span>{escape(fallback_text)}</span>
        </div>
        <div class="detail-field">
          <label>Status</label>
          <span>{escape(str(detail.get("runtime_status") or "unknown"))}; {escape(str(detail.get("policy_status") or "allowed"))}</span>
        </div>
        <div class="detail-field">
          <label>Adapter health</label>
          <span>{escape(str(detail.get("health_status") or "unknown"))}; {escape(str(detail.get("health_detail") or "not checked"))}</span>
        </div>
        <div class="detail-field detail-wide">
          <label>Install hint</label>
          <span>{escape(str(detail.get("install_hint") or detail.get("missing_dependency") or "no action needed"))}</span>
        </div>
        <div class="detail-field">
          <label>Models visible</label>
          <span>{escape(str(len(discovered_models)))} discovered; {escape(str(len(loaded_models)))} loaded</span>
        </div>
        <div class="detail-field detail-wide">
          <label>Detected models</label>
          <span class="code">{escape(detected_models_text)}</span>
        </div>
        <div class="detail-field detail-wide">
          <label>Model guidance</label>
          <span>{escape(str(detail.get("model_guidance") or "Configured model ids are operator-owned."))}</span>
        </div>
        <div class="detail-field">
          <label>Load action</label>
          <span>{escape(_support_label(load_support))}</span>
        </div>
        <div class="detail-field">
          <label>Unload action</label>
          <span>{escape(_support_label(unload_support))}</span>
        </div>
        <div class="detail-field detail-wide">
          <label>Runtime logs</label>
          <span class="code">{escape(", ".join(str(path) for path in log_paths) or str(logs.get("disabled_reason") or "not configured"))}</span>
        </div>
        {_policy_summary_fields(state)}
        <div class="runtime-actions">
          <div class="action-group">
            <button class="button-blue" type="button" onclick="postAction('/api/proxy/start', {{confirm: true}})">{_icon("play")} Start</button>
            <button class="button-red" type="button" onclick="postAction('/api/proxy/stop', {{confirm: true}})">{_icon("stop")} Stop</button>
            <button class="button-blue" type="button" onclick="postAction('/api/proxy/restart', {{confirm: true}})">{_icon("refresh")} Restart</button>
          </div>
          <div class="action-group">
            <button type="button" onclick="runtimeAction('runtime.status', {backend_js})">{_icon("pulse")} Status</button>
            <button type="button" onclick="runtimeAction('runtime.models', {backend_js})">{_icon("server")} Models</button>
            <button type="button" onclick="runtimeAction('runtime.loaded_models', {backend_js})">{_icon("database")} Loaded</button>
          </div>
          <div class="action-group">
            {_runtime_action_button("runtime.start_server", "Start runtime", "play", backend_name, model_id, start_support, confirm=True)}
            {_runtime_action_button("runtime.stop_server", "Stop runtime", "stop", backend_name, model_id, stop_support, confirm=True)}
            {_runtime_action_button("runtime.load_model", "Load model", "download", backend_name, model_id, load_support, confirm=True)}
            {_runtime_action_button("runtime.unload_model", "Unload model", "close", backend_name, model_id, unload_support, confirm=True)}
          </div>
        </div>
        <div id="last-action" class="muted detail-wide" aria-live="polite"></div>
        <pre id="runtime-output" class="catalog-output detail-wide">External runtimes own execution. Runtime actions are explicit, confirmed, and adapter-gated.</pre>
      </div>
    </div>"""


def _model_library_panel(state: Mapping[str, Any]) -> str:
    library = state.get("model_library") if isinstance(state.get("model_library"), dict) else {}
    return f"""<section class="panel" id="models" aria-labelledby="models-title">
      <div class="panel-title">
        <h2 id="models-title">Models</h2>
        <div class="settings-actions compact-actions">
          <button type="button" onclick="scanModels()">Scan local models</button>
          <button type="button" onclick="postAction('/api/download/plan').then(showModelAction)">Plan downloads</button>
        </div>
      </div>
      {_model_ops_summary(library)}
      <div class="models-grid">
        {_installed_models_card(library)}
        {_discover_models_card(library)}
        {_recommended_models_card(library)}
        {_eval_evidence_card(library)}
        {_eval_comparison_card(state)}
        {_downloads_models_card(library)}
        {_assignments_models_card(library)}
      </div>
      <pre id="model-library-output" class="catalog-output">Model actions are plan-first. Downloads and assignments require confirmation.</pre>
    </section>"""


def _model_ops_summary(library: Mapping[str, Any]) -> str:
    installed = library.get("installed") if isinstance(library.get("installed"), list) else []
    recommended = library.get("recommended") if isinstance(library.get("recommended"), list) else []
    downloads = library.get("downloads") if isinstance(library.get("downloads"), list) else []
    primary_installed = _first_mapping(installed)
    primary_recommended = _first_mapping(recommended)
    primary_download = _first_mapping(downloads)
    return f"""<div class="model-ops-strip" aria-label="Model recommendation and download status">
      {_model_inventory_summary(primary_installed, len(installed))}
      {_model_recommendation_summary(primary_recommended, len(recommended))}
      {_model_download_summary(primary_download, len(downloads))}
    </div>"""


def _model_inventory_summary(item: Mapping[str, Any] | None, count: int) -> str:
    if item is None:
        return _model_ops_tile(
            "Installed",
            "No local models found",
            "Run a local scan or connect a runtime.",
            "0 installed",
        )
    assigned = item.get("assigned_routes") if isinstance(item.get("assigned_routes"), list) else []
    score = item.get("score") if isinstance(item.get("score"), Mapping) else {}
    score_label = str(score.get("label") or "unscored")
    return _model_ops_tile(
        "Installed",
        _short_model_id(str(item.get("model_id") or "")),
        f"{str(item.get('source') or 'local')} · assigned {', '.join(str(route) for route in assigned) or 'none'}",
        f"{count} installed · {score_label}",
    )


def _model_recommendation_summary(item: Mapping[str, Any] | None, count: int) -> str:
    if item is None:
        return _model_ops_tile(
            "Recommended",
            "No recommendation yet",
            "Scan local models or inspect install prerequisites.",
            "0 candidates",
        )
    score = item.get("score") if isinstance(item.get("score"), Mapping) else {}
    routes = item.get("route_fit") if isinstance(item.get("route_fit"), list) else []
    route_text = ", ".join(str(route) for route in routes) or "general"
    score_text = str(item.get("score_label") or score.get("label") or "unscored")
    return _model_ops_tile(
        "Recommended",
        _short_model_id(str(item.get("model_id") or "")),
        f"{route_text} · {str(item.get('provider') or 'provider unknown')}",
        f"{count} candidates · {score_text}",
    )


def _model_download_summary(item: Mapping[str, Any] | None, count: int) -> str:
    if item is None:
        return _model_ops_tile(
            "Downloads",
            "No download planned",
            "Plan first; downloads never run silently.",
            "0 planned",
        )
    route = str(item.get("route") or "")
    model_id = str(item.get("model_id") or "")
    route_js = escape(json.dumps(route))
    model_js = escape(json.dumps(model_id))
    actions = (
        f'<div class="model-op-actions">'
        f'<button type="button" onclick="planDownload({route_js}, {model_js})">Plan</button>'
        f'<button type="button" class="danger-text" onclick="runDownload({route_js}, {model_js})">Download</button>'
        f"</div>"
    )
    return _model_ops_tile(
        "Downloads",
        _short_model_id(model_id),
        f"{route} · {str(item.get('status') or 'planned')}",
        f"{count} planned · confirm required",
        actions=actions,
    )


def _model_ops_tile(
    label: str,
    primary: str,
    detail: str,
    meta: str,
    *,
    actions: str = "",
) -> str:
    return f"""<div class="model-op-row">
      <div class="model-op-heading">
        <span class="model-op-title">{escape(label)}</span>
        <span class="model-op-meta">{escape(meta)}</span>
      </div>
      <strong class="model-op-main">{escape(primary)}</strong>
      <span class="model-op-detail">{escape(detail)}</span>
      {actions}
    </div>"""


def _first_mapping(items: list[Any]) -> Mapping[str, Any] | None:
    for item in items:
        if isinstance(item, Mapping):
            return item
    return None


def _short_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    if not model_id:
        return "none"
    if len(model_id) <= 54:
        return model_id
    return f"{model_id[:26]}...{model_id[-24:]}"


def _installed_models_card(library: Mapping[str, Any]) -> str:
    installed = library.get("installed") if isinstance(library.get("installed"), list) else []
    if installed:
        rows = "\n".join(
            _installed_model_row(item)
            for item in installed
            if isinstance(item, Mapping)
        )
    else:
        rows = (
            '<tr><td colspan="5" class="muted">No local models found yet. '
            'Run Scan local models, start Ollama/LM Studio, or place models under '
            '~/.model-router/models, ~/.lmstudio/models, ~/.ollama/models, or ~/models.</td></tr>'
        )
    return f"""<details class="model-card">
      <summary><span class="model-detail-title">Installed</span><span class="muted">{len(installed)} local models</span></summary>
      <div class="model-detail-body">
        <table class="data-table model-table">
          <thead><tr><th>Model</th><th>Source</th><th>Runtime</th><th>Assigned</th><th>Score</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>"""


def _installed_model_row(item: Mapping[str, Any]) -> str:
    score = item.get("score") if isinstance(item.get("score"), Mapping) else {}
    assigned = item.get("assigned_routes") if isinstance(item.get("assigned_routes"), list) else []
    runtime = item.get("runtime_compatibility") if isinstance(item.get("runtime_compatibility"), list) else []
    warnings = item.get("warnings") if isinstance(item.get("warnings"), list) else []
    warning_text = "; ".join(str(warning) for warning in warnings[:2])
    score_text = str(score.get("label") or "unscored")
    if score.get("overall_score") is not None:
        score_text += f" {score.get('overall_score')}"
    warning_html = (
        f'<br><span class="muted">{escape(warning_text)}</span>'
        if warning_text
        else ""
    )
    return f"""<tr>
      <td><strong class="code">{escape(str(item.get("model_id") or ""))}</strong><br><span class="muted">{escape(str(item.get("path") or "local scan"))}</span></td>
      <td>{escape(str(item.get("source") or "unknown"))}</td>
      <td>{escape(", ".join(str(value) for value in runtime) or "unknown")}</td>
      <td>{escape(", ".join(str(value) for value in assigned) or "none")}</td>
      <td>{escape(score_text)}{warning_html}</td>
    </tr>"""


def _discover_models_card(library: Mapping[str, Any]) -> str:
    discover = library.get("discover") if isinstance(library.get("discover"), Mapping) else {}
    results = discover.get("results") if isinstance(discover.get("results"), list) else []
    if results:
        rows = "\n".join(
            _discover_model_row(item)
            for item in results[:10]
            if isinstance(item, Mapping)
        )
    else:
        error = discover.get("error") or "Curated catalog did not return candidates."
        rows = f'<tr><td colspan="5" class="muted">{escape(str(error))}</td></tr>'
    return f"""<details class="model-card">
      <summary><span class="model-detail-title">Discover</span><span class="muted">{len(results)} catalog candidates</span></summary>
      <div class="model-detail-body">
        <table class="data-table model-table">
          <thead><tr><th>Model</th><th>Route</th><th>Runtime</th><th>Memory</th><th>Label</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>"""


def _discover_model_row(item: Mapping[str, Any]) -> str:
    routes = item.get("route_fit") if isinstance(item.get("route_fit"), list) else []
    memory = _memory_range(
        item.get("min_memory_gb"),
        item.get("recommended_memory_gb"),
    )
    warnings = item.get("warnings") if isinstance(item.get("warnings"), list) else []
    warning_text = "; ".join(str(warning) for warning in warnings[:2])
    warning_html = (
        f'<br><span class="muted">{escape(warning_text)}</span>'
        if warning_text
        else ""
    )
    return f"""<tr>
      <td><strong class="code">{escape(str(item.get("model_id") or ""))}</strong><br><span class="muted">{escape(str(item.get("reason") or ""))}</span></td>
      <td>{escape(", ".join(str(route) for route in routes) or "general")}</td>
      <td>{escape(str(item.get("runtime_kind") or "unknown"))}</td>
      <td>{escape(memory)}</td>
      <td>{escape(str(item.get("score_label") or "unscored"))}{warning_html}</td>
    </tr>"""


def _recommended_models_card(library: Mapping[str, Any]) -> str:
    recommended = library.get("recommended") if isinstance(library.get("recommended"), list) else []
    if recommended:
        rows = "\n".join(
            _recommended_model_row(item)
            for item in recommended[:10]
            if isinstance(item, Mapping)
        )
    else:
        rows = (
            '<tr><td colspan="4" class="muted">No hardware-aware recommendations yet. '
            'Run Scan local models or check installer prerequisites.</td></tr>'
        )
    return f"""<details class="model-card">
      <summary><span class="model-detail-title">Recommended</span><span class="muted">{len(recommended)} candidates; compact by default</span></summary>
      <div class="model-detail-body">
        <table class="data-table model-table">
          <thead><tr><th>Model</th><th>Route</th><th>Score</th><th>Why</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>"""


def _recommended_model_row(item: Mapping[str, Any]) -> str:
    routes = item.get("route_fit") if isinstance(item.get("route_fit"), list) else []
    score = item.get("score") if isinstance(item.get("score"), Mapping) else {}
    reasons = item.get("score_reasons") if isinstance(item.get("score_reasons"), list) else []
    return f"""<tr>
      <td><strong class="code">{escape(str(item.get("model_id") or ""))}</strong><br><span class="muted">{escape(str(item.get("provider") or ""))}</span></td>
      <td>{escape(", ".join(str(route) for route in routes))}</td>
      <td>{escape(str(item.get("score_label") or score.get("label") or "unscored"))} {escape(str(score.get("overall_score") or ""))}</td>
      <td>{escape("; ".join(str(reason) for reason in reasons[:2]) or str(item.get("reason") or ""))}</td>
    </tr>"""


def _eval_evidence_card(library: Mapping[str, Any]) -> str:
    rows_data = _eval_evidence_rows(library)
    if rows_data:
        rows = "\n".join(_eval_evidence_row(item) for item in rows_data[:10])
    else:
        rows = (
            '<tr><td colspan="5" class="muted">'
            "No cached eval evidence. Run evals explicitly when you want local "
            "model suitability evidence.</td></tr>"
        )
    return f"""<details class="model-card">
      <summary><span class="model-detail-title">Eval evidence</span><span class="muted">{len(rows_data)} model summaries</span></summary>
      <div class="model-detail-body">
        <table class="data-table model-table">
          <thead><tr><th>Model</th><th>Backend</th><th>Status</th><th>Scores</th><th>Last run</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <p class="muted">Advisory only; cached eval evidence does not change routing automatically.</p>
      </div>
    </details>"""


def _eval_comparison_card(state: Mapping[str, Any]) -> str:
    evals = state.get("evals") if isinstance(state.get("evals"), Mapping) else {}
    comparisons = (
        evals.get("comparisons")
        if isinstance(evals.get("comparisons"), list)
        else []
    )
    row_data = _eval_comparison_rows(comparisons)
    if row_data:
        rows = "\n".join(_eval_comparison_row(item) for item in row_data[:12])
    else:
        rows = (
            '<tr><td colspan="6" class="muted">'
            "No cached eval comparison evidence; candidates are not evaluated. "
            "Run comparisons explicitly from the CLI when you want local "
            "suitability evidence.</td></tr>"
        )
    hint = (
        "model-router eval compare --candidate fast:model-a "
        "--candidate balanced:model-b --fixture strict_json_routing_control_decision"
    )
    return f"""<details class="model-card model-card-wide">
      <summary><span class="model-detail-title">Eval comparisons</span><span class="muted">{len(comparisons)} cached comparisons; read-only</span></summary>
      <div class="model-detail-body">
        <table class="data-table model-table">
          <thead><tr><th>Candidate</th><th>Fixture set</th><th>Scores</th><th>Failures</th><th>Latency / usage</th><th>Run</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <p class="muted">Advisory only; best means best on this fixture set/profile, not universal model quality. Run evals explicitly: <span class="code">{escape(hint)}</span></p>
      </div>
    </details>"""


def _eval_comparison_rows(comparisons: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            continue
        candidates = (
            comparison.get("candidates")
            if isinstance(comparison.get("candidates"), list)
            else []
        )
        if not candidates:
            rows.append(_eval_comparison_empty_candidate_row(comparison))
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            rows.append(_eval_comparison_candidate_row(comparison, candidate))
    rows.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("comparison_id") or ""),
            str(item.get("candidate") or ""),
        ),
        reverse=True,
    )
    return rows


def _eval_comparison_candidate_row(
    comparison: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    privacy = (
        comparison.get("privacy")
        if isinstance(comparison.get("privacy"), Mapping)
        else {}
    )
    return {
        "comparison_id": comparison.get("comparison_id"),
        "created_at": comparison.get("created_at"),
        "status": comparison.get("status") or "unknown",
        "stale": comparison.get("stale") is True,
        "stale_reasons": (
            comparison.get("stale_reasons")
            if isinstance(comparison.get("stale_reasons"), list)
            else []
        ),
        "fixture_count": comparison.get("fixture_count"),
        "category_count": comparison.get("category_count"),
        "categories": (
            comparison.get("categories")
            if isinstance(comparison.get("categories"), list)
            else []
        ),
        "fixture_versions": (
            comparison.get("fixture_versions")
            if isinstance(comparison.get("fixture_versions"), list)
            else []
        ),
        "fixture_pack_version": comparison.get("fixture_pack_version"),
        "candidate": candidate.get("candidate"),
        "backend": candidate.get("backend"),
        "model": candidate.get("model"),
        "score_mean_percent": candidate.get("score_mean_percent"),
        "weighted_score_mean": candidate.get("weighted_score_mean"),
        "failed": candidate.get("failed", 0),
        "timeouts": candidate.get("timeouts", 0),
        "latency_summary": (
            candidate.get("latency_summary")
            if isinstance(candidate.get("latency_summary"), Mapping)
            else {}
        ),
        "usage_summary": (
            candidate.get("usage_summary")
            if isinstance(candidate.get("usage_summary"), Mapping)
            else {}
        ),
        "top_failure_reasons": (
            candidate.get("top_failure_reasons")
            if isinstance(candidate.get("top_failure_reasons"), list)
            else []
        ),
        "privacy": privacy,
    }


def _eval_comparison_empty_candidate_row(comparison: Mapping[str, Any]) -> dict[str, Any]:
    return _eval_comparison_candidate_row(
        comparison,
        {
            "candidate": "not evaluated",
            "backend": "unknown",
            "model": "unknown",
        },
    )


def _eval_comparison_row(item: Mapping[str, Any]) -> str:
    status = str(item.get("status") or "unknown")
    if item.get("stale"):
        status = f"{status} · stale"
    stale_reasons = (
        item.get("stale_reasons")
        if isinstance(item.get("stale_reasons"), list)
        else []
    )
    stale_detail = "; ".join(str(reason) for reason in stale_reasons[:2])
    stale_html = (
        f'<br><span class="muted">{escape(stale_detail)}</span>'
        if stale_detail
        else ""
    )
    fixture_text = _eval_comparison_fixture_text(item)
    score_text = (
        f"mean {_value_or_na(item.get('score_mean_percent'))} · "
        f"weighted {_value_or_na(item.get('weighted_score_mean'))}"
    )
    failure_text = _eval_comparison_failure_text(item)
    latency_usage = _eval_comparison_latency_usage_text(item)
    run_text = str(item.get("created_at") or "never")
    privacy_text = _eval_privacy_text(item.get("privacy"))
    return f"""<tr>
      <td><strong class="code">{escape(_short_model_id(str(item.get("model") or "unknown")))}</strong><br><span class="muted">{escape(str(item.get("backend") or "unknown"))} · {escape(status)}</span>{stale_html}</td>
      <td>{escape(fixture_text)}</td>
      <td>{escape(score_text)}</td>
      <td>{escape(failure_text)}</td>
      <td>{escape(latency_usage)}</td>
      <td>{escape(run_text)}<br><span class="muted">{escape(privacy_text)}</span></td>
    </tr>"""


def _eval_comparison_fixture_text(item: Mapping[str, Any]) -> str:
    categories = item.get("categories") if isinstance(item.get("categories"), list) else []
    category_text = ", ".join(str(category) for category in categories[:2]) or "none"
    if len(categories) > 2:
        category_text += f", +{len(categories) - 2}"
    versions = item.get("fixture_versions")
    version_text = ", ".join(str(version) for version in versions) if isinstance(versions, list) else ""
    if not version_text:
        version_text = "missing"
    pack_version = _value_or_na(item.get("fixture_pack_version"))
    return (
        f"fixtures {_value_or_na(item.get('fixture_count'))}; "
        f"categories {_value_or_na(item.get('category_count'))} ({category_text}); "
        f"pack v{pack_version}; rows v{version_text}"
    )


def _eval_comparison_failure_text(item: Mapping[str, Any]) -> str:
    reasons = item.get("top_failure_reasons") if isinstance(item.get("top_failure_reasons"), list) else []
    reason_text = "none"
    if reasons:
        parts = []
        for reason in reasons[:2]:
            if not isinstance(reason, Mapping):
                continue
            parts.append(f"{reason.get('reason')}: {reason.get('count')}")
        reason_text = "; ".join(parts) or "none"
    return (
        f"failed {_value_or_na(item.get('failed'))}; "
        f"timeouts {_value_or_na(item.get('timeouts'))}; {reason_text}"
    )


def _eval_comparison_latency_usage_text(item: Mapping[str, Any]) -> str:
    latency = item.get("latency_summary") if isinstance(item.get("latency_summary"), Mapping) else {}
    usage = item.get("usage_summary") if isinstance(item.get("usage_summary"), Mapping) else {}
    mean_ms = latency.get("mean_ms")
    latency_text = f"latency mean {_format_latency(mean_ms)}"
    usage_text = _format_usage_summary(usage)
    return f"{latency_text}; usage {usage_text}"


def _eval_privacy_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "privacy unknown"
    if value.get("legacy_raw_fields_detected") is True:
        return "legacy raw fields detected; details hidden"
    prompt = str(value.get("prompt_retention") or "unknown")
    output = str(value.get("output_retention") or "unknown")
    artifacts = str(value.get("artifact_retention") or "unknown")
    return f"prompt {prompt}; output {output}; artifacts {artifacts}"


def _eval_privacy_state() -> dict[str, Any]:
    return {
        "prompt_retention": "hash_only",
        "output_retention": "hash_only",
        "artifact_retention": "disabled_by_default",
        "raw_prompts_retained": False,
        "raw_outputs_retained": False,
        "secrets_retained": False,
    }


def _value_or_na(value: Any) -> Any:
    return value if value is not None else "n/a"


def _eval_evidence_rows(library: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = library.get("registry") if isinstance(library.get("registry"), Mapping) else {}
    models = registry.get("models") if isinstance(registry.get("models"), list) else []
    rows: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, Mapping):
            continue
        metadata = model.get("metadata") if isinstance(model.get("metadata"), Mapping) else {}
        summary = (
            metadata.get("latest_eval_summary")
            if isinstance(metadata.get("latest_eval_summary"), Mapping)
            else None
        )
        if summary is None:
            continue
        rows.append(
            {
                "model_id": str(model.get("model_id") or summary.get("model") or ""),
                "backend": str(model.get("backend") or summary.get("backend") or "any"),
                "status": str(summary.get("status") or "unknown"),
                "stale": summary.get("stale") is True,
                "fixture_count": summary.get("fixture_count"),
                "score_mean_percent": summary.get("score_mean_percent"),
                "weighted_score_mean": summary.get("weighted_score_mean"),
                "last_evaluated_at": summary.get("last_evaluated_at"),
                "by_category": (
                    summary.get("by_category")
                    if isinstance(summary.get("by_category"), Mapping)
                    else {}
                ),
            }
        )
    rows.sort(key=lambda item: (item["status"] == "not_evaluated", item["model_id"]))
    return rows


def _eval_evidence_row(item: Mapping[str, Any]) -> str:
    status = str(item.get("status") or "unknown")
    if item.get("stale"):
        status = f"{status} · stale"
    fixture_count = item.get("fixture_count")
    mean_score = item.get("score_mean_percent")
    weighted = item.get("weighted_score_mean")
    score_parts = [
        f"fixtures {fixture_count if fixture_count is not None else 'n/a'}",
        f"mean {mean_score if mean_score is not None else 'n/a'}",
        f"weighted {weighted if weighted is not None else 'n/a'}",
    ]
    categories = _eval_category_text(item.get("by_category"))
    if categories:
        score_parts.append(categories)
    return f"""<tr>
      <td><strong class="code">{escape(_short_model_id(str(item.get("model_id") or "")))}</strong></td>
      <td>{escape(str(item.get("backend") or "any"))}</td>
      <td>{escape(status)}</td>
      <td>{escape(" · ".join(score_parts))}</td>
      <td>{escape(str(item.get("last_evaluated_at") or "never"))}</td>
    </tr>"""


def _eval_category_text(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return ""
    parts: list[str] = []
    for category, group in list(raw.items())[:3]:
        if not isinstance(group, Mapping):
            continue
        score = group.get("score_mean_percent")
        passed = group.get("passed", 0)
        total = group.get("total", 0)
        score_text = score if score is not None else "n/a"
        parts.append(f"{category}: {score_text} ({passed}/{total})")
    if len(raw) > 3:
        parts.append(f"+{len(raw) - 3} categories")
    return "; ".join(parts)


def _downloads_models_card(library: Mapping[str, Any]) -> str:
    downloads = library.get("downloads") if isinstance(library.get("downloads"), list) else []
    if downloads:
        rows = "\n".join(
            _download_state_row(item)
            for item in downloads[:10]
            if isinstance(item, Mapping)
        )
    else:
        rows = (
            '<tr><td colspan="4" class="muted">No downloads planned. '
            'Use Plan downloads after scanning/recommendations; downloads never run silently.</td></tr>'
        )
    return f"""<details class="model-card">
      <summary><span class="model-detail-title">Downloads</span><span class="muted">{len(downloads)} planned; expand for commands</span></summary>
      <div class="model-detail-body">
        <table class="data-table model-table">
          <thead><tr><th>Model</th><th>Route</th><th>Status</th><th>Actions</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>"""


def _download_state_row(item: Mapping[str, Any]) -> str:
    route = str(item.get("route") or "")
    model_id = str(item.get("model_id") or "")
    route_js = escape(json.dumps(route))
    model_js = escape(json.dumps(model_id))
    command = " ".join(str(part) for part in item.get("command", []) if part)
    return f"""<tr>
      <td><strong class="code">{escape(model_id)}</strong><br><span class="muted">{escape(str(item.get("local_dir") or ""))}</span></td>
      <td>{escape(route)}</td>
      <td>{escape(str(item.get("status") or "planned"))}</td>
      <td>
        <button type="button" onclick="planDownload({route_js}, {model_js})">Plan</button>
        <button type="button" class="danger-text" onclick="runDownload({route_js}, {model_js})">Download</button>
        <br><span class="muted code">{escape(command)}</span>
      </td>
    </tr>"""


def _assignments_models_card(library: Mapping[str, Any]) -> str:
    assignments = library.get("assignments") if isinstance(library.get("assignments"), list) else []
    if assignments:
        rows = "\n".join(
            _assignment_model_row(item)
            for item in assignments
            if isinstance(item, Mapping)
        )
    else:
        rows = (
            '<tr><td colspan="5" class="muted">No route assignments are available because no valid proxy config is loaded.</td></tr>'
        )
    return f"""<details class="model-card model-card-wide">
      <summary><span class="model-detail-title">Assignments</span><span class="muted">{len(assignments)} route bindings</span></summary>
      <div class="model-detail-body">
        <table class="data-table model-table">
          <thead><tr><th>Route</th><th>Backend</th><th>Current model</th><th>Assign model</th><th>Action</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>"""


def _assignment_model_row(item: Mapping[str, Any]) -> str:
    route_id = str(item.get("route_id") or "")
    backend = str(item.get("backend") or "")
    model = str(item.get("model") or "")
    dom_id = _dom_id(f"assign-{route_id}")
    return f"""<tr>
      <td><strong>{escape(route_id)}</strong><br><span class="muted">{escape(str(item.get("route_class") or ""))}</span></td>
      <td>{escape(backend)}</td>
      <td><span class="code">{escape(model)}</span></td>
      <td><input id="{dom_id}" list="model-options-list" value="{escape(model)}" aria-label="Model for {escape(route_id)}"></td>
      <td><button type="button" onclick="assignRoute({json.dumps(route_id)}, {json.dumps(backend)}, {json.dumps(dom_id)})">Save</button><br><span class="muted">restart recommended</span></td>
    </tr>"""


def _memory_range(minimum: Any, recommended: Any) -> str:
    if minimum is None and recommended is None:
        return "unknown"
    if minimum is None:
        return f"rec {recommended} GB"
    if recommended is None:
        return f"min {minimum} GB"
    return f"{minimum}-{recommended} GB"


def _dom_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value)


def _policy_summary_fields(state: Mapping[str, Any]) -> str:
    provider_policy = state.get("provider_policy", {})
    backend_policy = state.get("backend_policy", {})
    catalog = state.get("catalog", {})
    if provider_policy.get("available"):
        provider_bits = [
            f"allow: {_csv_or_any(provider_policy.get('provider_allowlist'))}",
            f"deny: {_csv_or_none(provider_policy.get('provider_denylist'))}",
            "local-only"
            if provider_policy.get("local_only")
            else (
                "hosted allowed"
                if provider_policy.get("hosted_allowed", True)
                else "hosted blocked"
            ),
        ]
        route_pools = provider_policy.get("route_pools") or {}
        if isinstance(route_pools, dict) and route_pools:
            provider_bits.append("route pools: " + ", ".join(sorted(route_pools)))
        provider_text = "; ".join(provider_bits)
    else:
        provider_text = str(provider_policy.get("error") or "router policy unavailable")
    backend_text = (
        f"allow: {_csv_or_any(backend_policy.get('backend_allowlist'))}; "
        f"deny: {_csv_or_none(backend_policy.get('backend_denylist'))}"
    )
    catalog_state = (
        "missing"
        if not catalog.get("local_exists")
        else (
            "matches packaged"
            if catalog.get("local_matches_packaged")
            else "customized"
        )
    )
    catalog_text = (
        f"model catalog v{catalog.get('packaged_model_catalog_version', '?')}; "
        f"local config {catalog_state}; remote checks off"
    )
    return f"""
        <div class="sr-only">
          Provider policy: {escape(provider_text)}
          Backend policy: {escape(backend_text)}
          Catalog: {escape(catalog_text)}
        </div>"""


def _settings_provider_policy_summary(provider_policy: Mapping[str, Any]) -> str:
    if not provider_policy.get("available"):
        return str(provider_policy.get("error") or "router policy unavailable")
    mode = (
        "local-only"
        if provider_policy.get("local_only")
        else ("hosted allowed" if provider_policy.get("hosted_allowed", True) else "hosted blocked")
    )
    route_pools = provider_policy.get("route_pools") or {}
    route_pool_names = ", ".join(sorted(route_pools)) if isinstance(route_pools, dict) else ""
    parts = [
        f"v{provider_policy.get('version', 1)}",
        f"allow={_csv_or_any(provider_policy.get('provider_allowlist'))}",
        f"deny={_csv_or_none(provider_policy.get('provider_denylist'))}",
        mode,
    ]
    if route_pool_names:
        parts.append(f"route_pools={route_pool_names}")
    return "; ".join(parts)


def _csv_or_any(value: Any) -> str:
    values = [str(item) for item in value or [] if str(item)]
    return ", ".join(values) if values else "any"


def _csv_or_none(value: Any) -> str:
    values = [str(item) for item in value or [] if str(item)]
    return ", ".join(values) if values else "none"


def _provider_row(
    name: str,
    status: str,
    detail: str,
    class_name: str,
    dot_class: str,
) -> str:
    return f"""<div class="provider-row {class_name}">
      <span>{_icon(_provider_icon_name(name))}</span>
      <strong>{escape(name)}</strong>
      <span class="provider-status"><i class="dot {dot_class}"></i>{escape(status)}</span>
      <span class="muted">{escape(detail)}</span>
    </div>"""


def _runtime_panel_summary(state: Mapping[str, Any]) -> str:
    provider_state = state.get("provider_runtime", {})
    detail = provider_state.get("detail") if isinstance(provider_state, dict) else {}
    if not isinstance(detail, dict) or not detail:
        return "No backend selected."
    backend = detail.get("backend") or "backend"
    status = detail.get("runtime_status") or "configured"
    return f"{backend} is selected from config/telemetry; runtime is {status}."


def _runtime_detection_label(detail: Mapping[str, Any]) -> str:
    detected = detail.get("detected")
    detected_label = (
        "detected"
        if detected is True
        else "not detected"
        if detected is False
        else "unknown"
    )
    mode = str(detail.get("runtime_mode") or "external")
    checked = str(detail.get("last_checked_at") or "not checked")
    return f"{detected_label}; {mode}; {checked}"


def _adapter_dot(
    adapter: Mapping[str, Any],
    *,
    active: bool,
    managed: bool,
) -> str:
    health = adapter.get("health") if isinstance(adapter.get("health"), dict) else {}
    if health.get("ok") is True:
        return "green-dot"
    if health.get("status") in {"degraded", "error"}:
        return "red-dot"
    if active or managed:
        return "yellow-dot"
    return "green-dot"


def _support_label(value: Mapping[str, Any]) -> str:
    if value.get("supported") is True:
        return "supported"
    reason = value.get("disabled_reason")
    return f"disabled: {reason}" if reason else "disabled"


def _runtime_action_button(
    action_id: str,
    label: str,
    icon: str,
    backend: str,
    model: str,
    support: Mapping[str, Any],
    *,
    confirm: bool,
) -> str:
    disabled_reason = str(support.get("disabled_reason") or "").strip()
    if support.get("supported") is not True:
        title = escape(disabled_reason or "Runtime adapter does not support this action.")
        return (
            f'<button type="button" disabled title="{title}">'
            f'{_icon(icon)} {escape(label)}</button>'
        )
    return (
        '<button type="button" '
        f"onclick=\"runtimeAction({_js_string(action_id)}, {_js_string(backend)}, "
        f"{_js_string(model)}, {str(confirm).lower()})\">"
        f"{_icon(icon)} {escape(label)}</button>"
    )


def _js_string(value: str) -> str:
    return json.dumps(str(value))


def _runtime_link(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return "not configured"
    return f'<a href="{escape(url)}">{escape(url)}</a>'


def _risk_dot(risk: str) -> str:
    lowered = risk.lower()
    if lowered == "high":
        return "red-dot"
    if lowered == "medium":
        return "yellow-dot"
    if lowered == "low":
        return "green-dot"
    return "yellow-dot"


def _provider_icon_name(name: str) -> str:
    if "llama.cpp" in name:
        return "code"
    if "LM Studio" in name or "LocalAI" in name:
        return "cube"
    if "Ollama" in name:
        return "runtime"
    if "MLX-LM" in name:
        return "chip"
    if "OpenAI" in name or "Anthropic" in name or "OpenAI-compatible" in name:
        return "providers"
    if "Codex" in name or "Claude Code" in name:
        return "terminal"
    return "server"


def _route_receipt_panel(state: Mapping[str, Any]) -> str:
    receipt = state.get("route_receipt", {})
    receipt = receipt if isinstance(receipt, dict) else {}
    reason_codes = receipt.get("reason_codes") if isinstance(receipt.get("reason_codes"), list) else []
    rationale = "\n".join(
        f'<span class="pill blue">{escape(str(code))}</span>' for code in reason_codes[:8]
    )
    if not rationale:
        rationale = '<span class="pill gray">no route receipt yet</span>'
    request_id = str(receipt.get("request_id") or "")
    copy_button = (
        f'<button class="icon-only" type="button" aria-label="Copy request id" '
        f'onclick="copyText({json.dumps(request_id)})">{_icon("copy")}</button>'
        if request_id
        else f'<button class="icon-only" type="button" aria-label="No request id">{_icon("copy")}</button>'
    )
    return f"""<section class="inspector-card" aria-labelledby="receipt-title">
      <header>
        <h2 id="receipt-title">Route Receipt</h2>
        {copy_button}
      </header>
      <div class="receipt-body">
        <p class="receipt-summary">{escape(str(receipt.get("summary") or ""))}</p>
        <dl class="receipt-grid">
          <dt>Request ID:</dt><dd><span class="code">{escape(request_id or "none yet")}</span></dd>
          <dt>Selected:</dt><dd><strong class="linkish">{escape(str(receipt.get("selected") or ""))}</strong></dd>
          <dt>Backend:</dt><dd>{escape(str(receipt.get("backend") or ""))}</dd>
          <dt>Model:</dt><dd>{escape(str(receipt.get("model") or ""))}</dd>
          <dt>Reason:</dt><dd>{escape(str(receipt.get("reason") or ""))}</dd>
          <dt>Risk:</dt><dd><span class="status-line"><i class="dot {_risk_dot(str(receipt.get("risk") or ""))}"></i> {escape(str(receipt.get("risk") or ""))}</span></dd>
          <dt>Tools:</dt><dd>{escape(str(receipt.get("tools") or ""))}</dd>
          <dt>Fallback:</dt><dd>{escape(str(receipt.get("fallback") or ""))}</dd>
          <dt>Confirmation:</dt><dd>{escape(str(receipt.get("confirmation") or ""))}</dd>
        </dl>
        <div class="receipt-divider"></div>
        <dl class="receipt-grid">
          <dt>Routing latency:</dt><dd><strong class="linkish">{escape(str(receipt.get("route_latency") or "n/a"))}</strong></dd>
          <dt>Upstream latency:</dt><dd>{escape(str(receipt.get("upstream_latency") or "n/a"))}</dd>
          <dt>Privacy:</dt><dd><strong class="linkish">{escape(str(receipt.get("privacy") or ""))}</strong></dd>
        </dl>
        <div class="receipt-divider"></div>
        <h3>Rationale</h3>
        <div class="rationale">
          {rationale}
        </div>
        <div class="receipt-divider"></div>
        <dl class="receipt-grid">
          <dt>Policy:</dt><dd>{escape(str(receipt.get("policy") or ""))}</dd>
          <dt>Fallback:</dt><dd>{escape(str(receipt.get("fallback_explanation") or ""))}</dd>
          <dt>Safety:</dt><dd>{escape(str(receipt.get("safety") or ""))}</dd>
          <dt>Wrong route:</dt><dd>{escape(str(receipt.get("wrong_route") or ""))}</dd>
        </dl>
        <button class="receipt-button" type="button" data-feedback="{escape(request_id)}">
          <span>{_icon("comment")} Label wrong route</span>
          {_icon("chevron-right")}
        </button>
      </div>
    </section>"""


def _safety_panel() -> str:
    rows = (
        "Delete or destructive actions",
        "Send or publish actions",
        "Purchases or payments",
        "Deploy, merge, commit, or push",
        "Ambiguous high-impact requests",
    )
    toggle_rows = "\n".join(
        f'<label class="toggle-row"><span class="switch"><input type="checkbox" checked>'
        f'<span class="slider"></span></span><span>{escape(row)}</span></label>'
        for row in rows
    )
    return f"""<section class="inspector-card" id="safety" aria-labelledby="safety-title">
      <header>
        <h2 id="safety-title">{_icon("shield")} Safety</h2>
        <button class="icon-only" type="button" aria-label="Safety settings">{_icon("gear")}</button>
      </header>
      <div class="safety-list">
        <h3>Require confirmation before:</h3>
        {toggle_rows}
      </div>
      <div class="protected-note">{_icon("lock")} Protected defaults: safer by design.</div>
    </section>"""


def _maturity_panel(state: Mapping[str, Any]) -> str:
    maturity = state.get("maturity") if isinstance(state.get("maturity"), dict) else {}
    features = maturity.get("features") if isinstance(maturity.get("features"), list) else []
    rows = []
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        level = str(feature.get("maturity") or "unknown")
        rows.append(
            f"""<div class="review-item">
          <strong>{escape(str(feature.get("label") or feature.get("feature_id") or "Feature"))}</strong>
          <span class="pill {escape(_maturity_pill_class(level))}">{escape(level)}</span>
          <p class="muted">{escape(str(feature.get("release_gate") or ""))}</p>
        </div>"""
        )
    if not rows:
        rows.append('<div class="review-item muted">No maturity metadata loaded.</div>')
    status = escape(str(maturity.get("status") or "unknown"))
    return f"""<section class="inspector-card" id="maturity" aria-labelledby="maturity-title">
      <header>
        <h2 id="maturity-title">{_icon("shield")} Maturity</h2>
        <span class="muted">{status}</span>
      </header>
      <div class="review-list">
        {"".join(rows)}
      </div>
    </section>"""


def _maturity_pill_class(level: str) -> str:
    if level == "stable":
        return "green"
    if level == "experimental":
        return "yellow"
    if level == "beta":
        return "blue"
    return "gray"


def _settings_follow_through_panel(state: Mapping[str, Any]) -> str:
    proxy = state.get("proxy", {})
    observability = state.get("observability", {})
    backend_policy = state.get("backend_policy", {})
    model_options = state.get("model_options") if isinstance(state.get("model_options"), list) else []
    backends = state.get("backends") if isinstance(state.get("backends"), list) else []
    datalist = "\n".join(
        f'<option value="{escape(str(item.get("value") or ""))}">{escape(str(item.get("label") or ""))}</option>'
        for item in model_options
        if isinstance(item, dict)
    )
    backend_rows = "\n".join(
        _dashboard_backend_row(backend)
        for backend in backends
        if isinstance(backend, dict)
    )
    if not backend_rows:
        backend_rows = '<tr><td colspan="6" class="muted">No backend config loaded.</td></tr>'
    return f"""<section class="panel" id="settings" aria-labelledby="settings-title">
      <div class="panel-title">
        <h2 id="settings-title">Settings UI Follow-Through</h2>
        <span class="muted">All writes are explicit: Save, Apply, Restart.</span>
      </div>
      <div class="settings-grid">
        <label>Preset
          <select id="preset">{_options(state.get("presets", []), selected="lmstudio")}</select>
        </label>
        <label>Proxy host
          <input id="proxy-host" value="{escape(str(proxy.get("host") or "127.0.0.1"))}">
        </label>
        <label>Proxy port
          <input id="proxy-port" value="{escape(str(proxy.get("port") or DEFAULT_PROXY_PORT))}">
        </label>
        <label>Routing mode
          <select id="routing-mode">{_options(["decision", "manual"], selected=proxy.get("routing_mode") or "decision")}</select>
        </label>
        <label>Manual default backend
          <select id="default-backend">{_backend_options(backends, selected=proxy.get("default_backend"))}</select>
        </label>
        <label>Manual default model
          <input id="default-model" value="{escape(str(proxy.get("default_model") or ""))}" placeholder="model id">
        </label>
        <label>Respect client model
          <select id="respect-client-model">{_bool_options(proxy.get("respect_client_model"))}</select>
        </label>
        <label>Unknown model behavior
          <select id="unknown-model-behavior">{_options(["fallback_to_default", "reject_404"], selected=proxy.get("unknown_model_behavior") or "fallback_to_default")}</select>
        </label>
        <label>Safety gate mode
          <select id="safety-gate-mode">{_options(["decision_only", "always_static", "off"], selected=proxy.get("safety_gate_mode") or "decision_only")}</select>
        </label>
        <label>Observability
          <select id="observability-enabled">{_bool_options(observability.get("enabled"))}</select>
        </label>
        <label>Prompt capture
          <select id="prompt-capture">{_options(state.get("prompt_capture_modes", []), selected=observability.get("prompt_capture"))}</select>
        </label>
        <label>Telemetry log
          <input id="observability-log" value="{escape(str(observability.get("log_path") or ""))}">
        </label>
        <label>Backend allowlist
          <input id="backend-allowlist" value="{escape(", ".join(backend_policy.get("backend_allowlist") or []))}" placeholder="any">
        </label>
        <label>Backend denylist
          <input id="backend-denylist" value="{escape(", ".join(backend_policy.get("backend_denylist") or []))}" placeholder="none">
        </label>
        <label>Scanned / recommended models
          <input id="model-options" list="model-options-list" placeholder="Select a model below">
          <datalist id="model-options-list">{datalist}</datalist>
        </label>
      </div>
      <div class="backend-editor">
        <table class="data-table">
          <thead>
            <tr>
              <th>Backend</th>
              <th>Model</th>
              <th>Base URL</th>
              <th>Runtime</th>
              <th>Readiness / idle</th>
              <th>Log</th>
            </tr>
          </thead>
          <tbody>{backend_rows}</tbody>
        </table>
      </div>
      <div class="settings-actions">
        <button type="button" onclick="applyPreset()">Apply preset template</button>
        <button class="button-blue" type="button" onclick="saveConfig()">Save config</button>
        <button type="button" onclick="saveConfig().then(() => postAction('/api/proxy/restart', {{confirm: true}}))">Apply and restart proxy</button>
        <button type="button" onclick="scanModels()">Scan models</button>
        <button type="button" onclick="postAction('/api/doctor')">Run doctor</button>
        <span id="last-action" class="muted" aria-live="polite"></span>
      </div>
      <pre id="scan-output" class="catalog-output">No scan action yet.</pre>
    </section>"""


def _dashboard_backend_row(backend: Mapping[str, Any]) -> str:
    runtime = backend.get("runtime") if isinstance(backend.get("runtime"), dict) else {}
    command = " ".join(shlex.quote(part) for part in runtime.get("command", []))
    name = escape(str(backend.get("name") or ""))
    model = escape(str(backend.get("model") or ""))
    base_url = escape(str(backend.get("base_url") or ""))
    readiness_url = escape(str(runtime.get("readiness_url") or ""))
    idle_timeout = escape(str(runtime.get("idle_timeout_seconds") or ""))
    log_path = escape(str(runtime.get("log_path") or ""))
    return f"""<tr data-backend="{name}">
      <td><strong>{name}</strong><br><span class="muted">{escape(_route_for_backend(str(backend.get("name") or "")))}</span></td>
      <td><input data-field="model" list="model-options-list" value="{model}"></td>
      <td><input data-field="base_url" value="{base_url}"></td>
      <td>
        <select data-field="runtime_enabled">{_bool_options(runtime.get("enabled"))}</select>
        <select data-field="runtime_kind">{_options(["generic", "llama-server", "mlx-lm"], selected=runtime.get("kind"))}</select>
        <textarea data-field="runtime_command">{escape(command)}</textarea>
      </td>
      <td>
        <input data-field="readiness_url" value="{readiness_url}" placeholder="readiness URL">
        <input data-field="idle_timeout_seconds" value="{idle_timeout}" placeholder="idle seconds">
      </td>
      <td><input data-field="log_path" value="{log_path}" placeholder="log path"></td>
    </tr>"""


def _benchmark_status_panel(state: Mapping[str, Any]) -> str:
    benchmarks = state.get("benchmarks", {})
    workflow = state.get("workflow_benchmarks", {})
    best = _benchmark_best_summary(benchmarks if isinstance(benchmarks, dict) else {})
    return f"""<section class="inspector-card" aria-labelledby="benchmarks-title">
      <header>
        <h2 id="benchmarks-title">{_icon("pulse")} Benchmarks</h2>
        <span class="muted">explicit only</span>
      </header>
      <dl class="receipt-grid">
        <dt>Local backend:</dt><dd>{escape(str(benchmarks.get("completed", 0) if isinstance(benchmarks, dict) else 0))} completed, {escape(str(benchmarks.get("failed", 0) if isinstance(benchmarks, dict) else 0))} failed</dd>
        <dt>Best measured:</dt><dd class="code">{escape(best)}</dd>
        <dt>Workflow:</dt><dd>{escape(str(workflow.get("status") if isinstance(workflow, dict) else "unknown"))}</dd>
        <dt>Command:</dt><dd class="code">{escape(str(workflow.get("command") if isinstance(workflow, dict) else "model-router workflow-benchmark --json --fail-on-mismatch"))}</dd>
      </dl>
      <div class="receipt-divider"></div>
      <div class="runtime-actions">
        <button type="button" onclick="planBenchmark()">Plan local benchmark</button>
        <button class="danger-text" type="button" onclick="runBenchmark()">Run local benchmark</button>
      </div>
      <pre id="benchmark-output" class="catalog-output">No benchmark action yet.</pre>
    </section>"""


def _review_panel(state: Mapping[str, Any]) -> str:
    review = state.get("review", {})
    review = review if isinstance(review, dict) else {}
    items = review.get("items") if isinstance(review.get("items"), list) else []
    latest_request = str(state.get("route_receipt", {}).get("request_id") or "")
    if items:
        rows = "\n".join(
            _review_item(item, state)
            for item in items
            if isinstance(item, dict)
        )
    else:
        rows = (
            '<div class="review-item muted">No unlabeled routing events in the review queue.</div>'
        )
    return f"""<section class="inspector-card" id="review" aria-labelledby="review-title">
      <header>
        <h2 id="review-title">{_icon("comment")} Telemetry Review</h2>
        <span class="muted">{escape(str(review.get("reviewable", 0)))} open</span>
      </header>
      <div class="review-list">
        <p class="muted">{escape(str(review.get("privacy") or "Prompts and secrets are hidden by default."))}</p>
        {rows}
        <div class="review-item">
          <strong>Manual label</strong>
          <div class="review-form">
            <label>Request ID
              <input id="feedback-request-id" value="{escape(latest_request)}" placeholder="request id">
            </label>
            <label>Expected route
              <select id="feedback-engine">{_review_engine_options(state)}</select>
            </label>
            <label>Outcome
              <select id="feedback-outcome">{_outcome_label_options()}</select>
            </label>
            <label>Notes
              <input id="feedback-notes" placeholder="optional, privacy-safe note">
            </label>
          </div>
          <div class="settings-actions">
            <button class="button-blue" type="button" onclick="sendFeedback()">Submit feedback</button>
          </div>
        </div>
      </div>
    </section>"""


def _review_item(item: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    request_id = str(item.get("request_id") or "")
    reason_codes = item.get("reason_codes") if isinstance(item.get("reason_codes"), list) else []
    reason_html = " ".join(
        f'<span class="pill blue">{escape(str(code))}</span>' for code in reason_codes[:4]
    )
    if not reason_html:
        reason_html = '<span class="pill gray">no reason codes</span>'
    return f"""<div class="review-item">
      <strong class="code">{escape(request_id)}</strong>
      <button class="text-button" type="button" onclick="copyText({json.dumps(request_id)})">Copy id</button>
      <p class="muted">{escape(str(item.get("receipt_summary") or "No receipt summary recorded."))}</p>
      <dl class="receipt-grid">
        <dt>Selected:</dt><dd>{escape(str(item.get("selected_engine") or ""))}</dd>
        <dt>Backend:</dt><dd>{escape(str(item.get("backend") or "unassigned"))}</dd>
        <dt>Status:</dt><dd>{escape(str(item.get("status") or "unknown"))}</dd>
        <dt>Tokens:</dt><dd class="code">{escape(str(item.get("usage_tokens") or "none"))}</dd>
        <dt>Cost:</dt><dd class="code">{escape(str(item.get("cost_estimate") or "none"))}</dd>
        <dt>Replayable:</dt><dd>{escape("yes" if item.get("replayable") else "no; private/no full prompt")}</dd>
      </dl>
      <div class="rationale">{reason_html}</div>
      <div class="review-form">
        <label>Expected route
          <select id="expected-{escape(request_id)}">{_review_engine_options(state)}</select>
        </label>
        <label>Outcome
          <select id="outcome-{escape(request_id)}">{_outcome_label_options()}</select>
        </label>
        <label>Notes
          <input id="notes-{escape(request_id)}" placeholder="optional, privacy-safe note">
        </label>
      </div>
      <div class="settings-actions">
        <button class="button-blue" type="button" onclick="sendReviewFeedback({json.dumps(request_id)})">Label route</button>
      </div>
    </div>"""


def _review_engine_options(state: Mapping[str, Any]) -> str:
    route_ids = [
        str(row.get("route_id"))
        for row in state.get("route_map", [])
        if isinstance(row, dict) and row.get("route_id")
    ]
    if "human_confirm" not in route_ids:
        route_ids.append("human_confirm")
    return _options(tuple(dict.fromkeys(route_ids)), selected="")


def _outcome_label_options() -> str:
    options = ['<option value="">manual outcome optional</option>']
    options.extend(
        f'<option value="{escape(label)}">{escape(label)}</option>'
        for label in OUTCOME_LABELS
    )
    return "\n".join(options)


def _catalog_panel(state: Mapping[str, Any]) -> str:
    catalog = state.get("catalog", {})
    local_state = (
        "missing"
        if not catalog.get("local_exists")
        else (
            "matches packaged"
            if catalog.get("local_matches_packaged")
            else "customized"
        )
    )
    version = escape(str(catalog.get("packaged_model_catalog_version", "?")))
    config_path = escape(str(catalog.get("local_config") or ""))
    log_path = escape(str(catalog.get("migration_log") or ""))
    overrides = catalog.get("overrides") or []
    override_text = escape(", ".join(overrides) if overrides else "none")
    return f"""<section class="inspector-card" id="catalog" aria-labelledby="catalog-title">
      <header>
        <h2 id="catalog-title">{_icon("server")} Catalog</h2>
        <span class="muted">packaged only</span>
      </header>
      <dl class="receipt-grid">
        <dt>Model catalog:</dt><dd>v{version}</dd>
        <dt>Local config:</dt><dd>{escape(local_state)}</dd>
        <dt>Overrides:</dt><dd>{override_text}</dd>
        <dt>Config path:</dt><dd class="code">{config_path}</dd>
        <dt>Migration log:</dt><dd class="code">{log_path}</dd>
      </dl>
      <div class="receipt-divider"></div>
      <div class="runtime-actions">
        <button type="button" onclick="showCatalogDiff()">{_icon("braces")} Diff</button>
        <button class="danger-text" type="button" onclick="applyCatalogUpdate()">{_icon("check")} Apply</button>
      </div>
      <pre id="catalog-output" class="catalog-output">No catalog action yet.</pre>
    </section>"""


def _pricing_catalog_panel(state: Mapping[str, Any]) -> str:
    pricing = state.get("pricing_catalog", {})
    pricing = pricing if isinstance(pricing, Mapping) else {}
    override_state = (
        "missing"
        if not pricing.get("override_exists")
        else ("valid" if pricing.get("override_valid") else "invalid")
    )
    active_version = escape(str(pricing.get("active_catalog_version") or "?"))
    active_source = escape(str(pricing.get("active_catalog_source") or "unknown"))
    override_path = escape(str(pricing.get("override_path") or ""))
    entries = escape(str(pricing.get("active_entry_count", 0)))
    validation = pricing.get("validation_errors")
    validation_text = escape(", ".join(validation) if isinstance(validation, list) and validation else "none")
    return f"""<section class="inspector-card" id="pricing" aria-labelledby="pricing-title">
      <header>
        <h2 id="pricing-title">{_icon("database")} Pricing</h2>
        <span class="muted">local catalog</span>
      </header>
      <dl class="receipt-grid">
        <dt>Active catalog:</dt><dd>v{active_version}</dd>
        <dt>Source:</dt><dd>{active_source}</dd>
        <dt>Entries:</dt><dd>{entries}</dd>
        <dt>Override:</dt><dd>{escape(override_state)}</dd>
        <dt>Override path:</dt><dd class="code">{override_path}</dd>
        <dt>Validation:</dt><dd>{validation_text}</dd>
      </dl>
      <p class="muted">Cost estimates use local metadata only. Verify provider prices before spend review.</p>
      <div class="receipt-divider"></div>
      <div class="runtime-actions">
        <button type="button" onclick="showPricingStatus()">{_icon("pulse")} Status</button>
        <button type="button" onclick="showPricingDiff()">{_icon("braces")} Diff</button>
        <button class="danger-text" type="button" onclick="applyPricingCatalog()">{_icon("check")} Apply</button>
      </div>
      <pre id="pricing-output" class="catalog-output">No pricing action yet.</pre>
    </section>"""


def _recent_requests_table(state: Mapping[str, Any]) -> str:
    rows = state.get("recent_events") if isinstance(state.get("recent_events"), list) else []
    telemetry = state.get("telemetry", {})
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    catalog_coverage = escape(
        _format_catalog_coverage(telemetry.get("catalog_coverage"))
    )
    catalog_gaps = escape(_format_catalog_gap_list(telemetry.get("catalog_coverage_gaps")))
    body = "\n".join(
        f"""<tr>
          <td>{escape(str(row.get("time") or "—"))}</td>
          <td><span class="code">{escape(str(row.get("selected_engine") or ""))}</span></td>
          <td>{escape(str(row.get("backend") or ""))}</td>
          <td><span class="status-line"><i class="dot {escape(str(row.get("dot") or "yellow-dot"))}"></i>{escape(str(row.get("status") or ""))}</span></td>
          <td>{escape(str(row.get("total_latency") or row.get("route_latency") or "n/a"))}</td>
          <td><span class="mono">{escape(str(row.get("usage_tokens") or "none"))}</span></td>
          <td><button class="text-button" type="button" onclick="copyText({json.dumps(str(row.get("request_id") or ""))})">Copy id</button></td>
          <td><button class="text-button" type="button" data-feedback="{escape(str(row.get("request_id") or ""))}">Wrong route? {_icon("comment")}</button></td>
        </tr>"""
        for row in rows[:8]
        if isinstance(row, dict)
    )
    if not body:
        body = '<tr><td colspan="8" class="muted">No routing telemetry yet. Start the proxy and send a request.</td></tr>'
    return f"""<p class="muted">Catalog coverage: <span class="code">{catalog_coverage}</span></p>
    <p class="muted">Coverage gaps: <span class="code">{catalog_gaps}</span></p>
    {_pricing_override_skeleton_block(telemetry)}
    <table class="data-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Route</th>
          <th>Backend</th>
          <th>Status</th>
          <th>Latency</th>
          <th>Tokens</th>
          <th>Request ID</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>"""


def _compact_control_panel(
    state: Mapping[str, Any],
    endpoint: str,
    proxy_state: str,
    telemetry_state: str,
    profile_label: str,
) -> str:
    receipt = state.get("route_receipt", {})
    receipt = receipt if isinstance(receipt, dict) else {}
    recent = state.get("recent_events") if isinstance(state.get("recent_events"), list) else []
    proxy_dot = "green-dot" if proxy_state.lower() == "running" else "yellow-dot"
    telemetry_dot = "green-dot" if telemetry_state == "On" else "yellow-dot"
    selected = str(receipt.get("selected") or "none")
    backend = str(receipt.get("backend") or "none")
    privacy = str(receipt.get("privacy") or "configured")
    compact_recent = "\n".join(
        f"""<div class="compact-recent-row"><span>{escape(str(row.get("time") or "—"))}</span><span>{escape(str(row.get("selected_engine") or ""))}</span><span>{escape(str(row.get("backend") or ""))}</span><span><i class="dot {escape(str(row.get("dot") or "yellow-dot"))}"></i>{escape(str(row.get("total_latency") or "n/a"))}</span></div>"""
        for row in recent[:2]
        if isinstance(row, dict)
    )
    if not compact_recent:
        compact_recent = '<div class="compact-recent-row"><span>—</span><span>No telemetry yet</span><span>—</span><span>—</span></div>'
    return f"""<section class="compact-window" aria-label="ModelRouter compact control panel">
      <div class="compact-header">
        <div class="compact-title">
          <div class="traffic-lights" aria-hidden="true">
            <span class="red"></span><span class="yellow"></span><span class="green"></span>
          </div>
          <span>ModelRouter</span>
        </div>
        <a class="icon-only" href="/" aria-label="Open full control center">{_icon("gear")}</a>
      </div>
      <div class="compact-body-content">
        <div class="compact-chips">
          <span class="compact-chip">{endpoint.replace("http://", "")}</span>
          <span class="compact-chip"><i class="dot {proxy_dot}"></i>{escape(proxy_state)}</span>
          <span class="compact-chip accent" id="compact-mode">{profile_label}</span>
          <span class="compact-chip"><i class="dot {telemetry_dot}"></i>{escape(telemetry_state)}</span>
        </div>
        <div class="compact-flow">
          <span class="compact-box">Request</span><span>→</span>
          <span class="compact-box selected">{escape(selected)}</span><span>→</span>
          <span class="compact-box">{escape(backend)}</span><span>→</span>
          <span class="compact-box">Response</span>
        </div>
        <div class="compact-summary">
          <div><span>Selected</span><strong class="linkish">{escape(selected)}</strong></div>
          <div><span>Routing latency</span><strong>{escape(str(receipt.get("route_latency") or "n/a"))}</strong></div>
          <div><span>Backend</span><strong>{escape(backend)}</strong></div>
          <div><span>Upstream latency</span><strong>{escape(str(receipt.get("upstream_latency") or "n/a"))}</strong></div>
          <div><span>Privacy</span><strong class="linkish">{escape(privacy)}</strong></div>
          <div><span>Safety</span><strong>{escape(str(receipt.get("confirmation") or "n/a"))}</strong></div>
        </div>
        <div class="compact-recent">
          <h3>Recent</h3>
          {compact_recent}
        </div>
        <div class="compact-actions-grid">
          <a href="/">{_icon("open")}<span>Full</span></a>
          <button type="button" onclick="postAction('/api/proxy/stop', {{confirm: true}})">{_icon("pause")}<span>Pause Proxy</span></button>
          <a href="/#receipt-title">{_icon("document")}<span>Receipt</span></a>
          <a href="/#providers">{_icon("server")}<span>Providers</span></a>
          <a href="/#safety">{_icon("shield")}<span>Safety</span></a>
        </div>
      </div>
      <div class="compact-bottom">
        <span class="compact-chip"><i class="dot {proxy_dot}"></i>Proxy {escape(proxy_state)}</span>
        <span class="compact-chip"><i class="dot {telemetry_dot}"></i>Telemetry {escape(telemetry_state)}</span>
        <span class="compact-chip"><i class="dot green-dot"></i>No chat surface</span>
      </div>
    </section>"""


def _icon(name: str) -> str:
    icons = {
        "arrow-right": '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
        "balanced": '<circle cx="12" cy="12" r="8"/><path d="M8 12h8"/><path d="M12 8v8"/>',
        "braces": '<path d="M8 4c-2 1-2 3-1 5 .5 1 .5 2 0 3-1 2-1 4 1 5"/><path d="M16 4c2 1 2 3 1 5-.5 1-.5 2 0 3 1 2 1 4-1 5"/>',
        "chevron-right": '<path d="m9 18 6-6-6-6"/>',
        "chip": '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M4 9h3M4 15h3M17 9h3M17 15h3M9 4v3M15 4v3M9 17v3M15 17v3"/>',
        "code": '<path d="m8 16-4-4 4-4"/><path d="m16 8 4 4-4 4"/><path d="m14 5-4 14"/>',
        "comment": '<path d="M21 12a7 7 0 0 1-7 7H7l-4 3 1.2-5A7 7 0 1 1 21 12Z"/>',
        "copy": '<rect x="9" y="9" width="10" height="10" rx="2"/><rect x="5" y="5" width="10" height="10" rx="2"/>',
        "cube": '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="M4 7.5 12 12l8-4.5"/><path d="M12 12v9"/>',
        "database": '<ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/><path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>',
        "document": '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/>',
        "gear": '<path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z"/><path d="M3 12h2M19 12h2M12 3v2M12 19v2M5.6 5.6 7 7M17 17l1.4 1.4M18.4 5.6 17 7M7 17l-1.4 1.4"/>',
        "image": '<rect x="4" y="5" width="16" height="14" rx="2"/><circle cx="9" cy="10" r="1.5"/><path d="m7 17 4-4 3 3 2-2 3 3"/>',
        "lock": '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
        "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4 7 7 0 1 0 20 14.5Z"/>',
        "more": '<circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>',
        "open": '<path d="M14 3h7v7"/><path d="m21 3-9 9"/><path d="M11 5H5v14h14v-6"/>',
        "overview": '<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/><rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/>',
        "pause": '<path d="M8 5v14"/><path d="M16 5v14"/>',
        "play": '<path d="m8 5 11 7-11 7Z"/>',
        "providers": '<circle cx="12" cy="7" r="3"/><circle cx="6" cy="17" r="3"/><circle cx="18" cy="17" r="3"/><path d="M10 9.5 7.5 14M14 9.5l2.5 4.5M9 17h6"/>',
        "pulse": '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
        "puzzle": '<path d="M9 3h6v4h2a2 2 0 1 1 0 4h-2v3h-3v2a2 2 0 1 1-4 0v-2H5V9h2a2 2 0 1 0 0-4h2z"/>',
        "refresh": '<path d="M20 6v5h-5"/><path d="M4 18v-5h5"/><path d="M19 11a7 7 0 0 0-12-4"/><path d="M5 13a7 7 0 0 0 12 4"/>',
        "request": '<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/><path d="m13 9 3 3-3 3"/>',
        "reasoning": '<path d="M12 3a7 7 0 0 0-4 12v3h8v-3a7 7 0 0 0-4-12Z"/><path d="M9 21h6"/><path d="M10 11h4"/>',
        "response": '<path d="M8 5H4v14h4"/><path d="M12 8l4 4-4 4"/><path d="M16 12H7"/>',
        "routing": '<path d="M6 4v5a3 3 0 0 0 3 3h6"/><path d="m12 9 3 3-3 3"/><path d="M18 20v-5a3 3 0 0 0-3-3H9"/>',
        "runtime": '<rect x="5" y="6" width="14" height="12" rx="2"/><path d="M8 10h8M8 14h5"/>',
        "server": '<rect x="4" y="5" width="16" height="6" rx="2"/><rect x="4" y="13" width="16" height="6" rx="2"/><path d="M8 8h.01M8 16h.01"/>',
        "shield": '<path d="M12 3 20 6v6c0 5-3.3 8-8 9-4.7-1-8-4-8-9V6l8-3Z"/><path d="m9 12 2 2 4-5"/>',
        "simple": '<path d="M6 12h12"/><path d="M12 6v12"/>',
        "stop": '<rect x="7" y="7" width="10" height="10" rx="1"/>',
        "terminal": '<path d="m5 8 4 4-4 4"/><path d="M11 16h8"/>',
        "vision": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    }
    path = icons.get(name, icons["overview"])
    return (
        '<svg class="icon" viewBox="0 0 24 24" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f"{path}</svg>"
    )


async def _request_payload(request: Any) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        return payload if isinstance(payload, dict) else {}
    raw = (await request.body()).decode("utf-8")
    parsed = parse_qs(raw, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


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


def _provider_policy_state(config: RoutingProxyConfig | None) -> dict[str, Any]:
    if config is None or not config.router_config:
        return {"available": False, "error": "router config unavailable"}
    try:
        router_config = load_router_config(config.router_config)
    except (RouterConfigError, OSError) as exc:
        return {"available": False, "error": str(exc)}
    return {"available": True, **router_config.provider_policy.to_dict()}


def _backend_policy_state(config: RoutingProxyConfig | None) -> dict[str, Any]:
    if config is None:
        return {"backend_allowlist": [], "backend_denylist": []}
    return config.backend_policy.to_dict()


def _verifier_state(config: RoutingProxyConfig | None) -> dict[str, Any]:
    if config is None:
        return {"version": 1, "mode": "off"}
    return config.verifier.to_dict()


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
                "runtime_adapter": runtime_state_for_backend(
                    backend,
                    timeout_seconds=min(config.health.backend_timeout_seconds, 0.2),
                ),
            }
        )
    return rows


def _runtime_models_from_backend_states(
    backend_states: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    runtime_models: dict[str, dict[str, Any]] = {}
    for backend in backend_states:
        backend_name = str(backend.get("name") or "")
        adapter = backend.get("runtime_adapter")
        if not backend_name or not isinstance(adapter, dict):
            continue
        runtime_models[backend_name] = dict(adapter)
    return runtime_models


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


def _telemetry_state(
    paths: Mapping[str, Path],
    config: RoutingProxyConfig | None,
) -> dict[str, Any]:
    events_path = paths["events"]
    feedback_path = paths["feedback"]
    event_summary = _safe_event_summary(events_path)
    try:
        summary = replay_events(
            events_path=events_path,
            feedback_path=feedback_path,
            config_path=config.router_config if config is not None else None,
            pricing_catalog_path=paths.get("pricing"),
            max_examples=10,
        )
    except Exception as exc:
        summary = {
            "events": 0,
            "feedback_labels": 0,
            "unlabeled_replayable": 0,
            "expected_mismatch_count": 0,
            "error": str(exc),
        }
    try:
        labels = feedback_summary(
            feedback_path=feedback_path,
            events_path=events_path,
            include_notes=False,
            max_rows=10,
        )
    except Exception:
        labels = {"labels": []}
    recent_ids = [
        _safe_event_string(label.get("request_id"), max_chars=120)
        for label in labels.get("labels", [])
        if label.get("request_id")
    ]
    event_recent_ids = event_summary.pop("recent_request_ids")
    return {
        **event_summary,
        **_sanitize_telemetry_summary(summary),
        "recent_request_ids": event_recent_ids or recent_ids,
    }


def _sanitize_telemetry_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in summary.items():
        if key in {
            "selected_engine_counts",
            "status_counts",
            "mismatch_groups",
            "confusion_matrix",
            "upstream_model_counts",
            "outcome_label_counts",
            "pricing_match_counts",
        }:
            payload[key] = _safe_count_mapping(value)
        elif key in {
            "usage_by_selected_engine",
            "usage_by_backend",
            "usage_by_model",
        }:
            payload[key] = _safe_usage_group_mapping(value)
        elif key == "catalog_coverage":
            payload[key] = _safe_catalog_coverage(value)
        elif key == "catalog_coverage_gaps":
            payload[key] = _safe_catalog_gap_list(value)
        elif key in {
            "unlabeled_replayable_request_ids",
            "skipped_no_prompt_request_ids",
            "feedback_without_event_request_ids",
            "feedback_for_private_event_request_ids",
        }:
            payload[key] = _safe_string_list(value, max_chars=120)
        elif key in {
            "route_changes",
            "expected_mismatches",
        }:
            payload[key] = _safe_telemetry_rows(value)
        elif isinstance(value, str):
            payload[key] = _safe_event_string(value, max_chars=320)
        else:
            payload[key] = value
    payload["pricing_override_skeleton"] = _safe_multiline_text(
        pricing_override_skeleton_from_gaps(
            payload.get("catalog_coverage_gaps", []),
        ),
        max_chars=12_000,
    )
    return payload


def _safe_count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: Counter[str] = Counter()
    for raw_key, raw_count in value.items():
        key = _safe_event_string(raw_key, max_chars=120)
        if key:
            counts[key] += _safe_int(raw_count)
    return dict(sorted(counts.items()))


def _safe_usage_group_mapping(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        return {}
    groups: dict[str, dict[str, int]] = {}
    for raw_key, raw_usage in value.items():
        key = _safe_event_string(raw_key, max_chars=160)
        usage = _safe_usage_summary(raw_usage)
        if key and _usage_has_tokens(usage):
            groups[key] = usage
    return dict(sorted(groups.items()))


def _safe_usage_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _empty_usage_summary()
    usage = _empty_usage_summary()
    for field in (
        "events",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "usage_total_tokens",
        "usage_cached_input_tokens",
        "estimated_cost_events",
    ):
        usage[field] = _safe_int(value.get(field))
    for field in (
        "estimated_input_cost",
        "estimated_output_cost",
        "estimated_cached_input_cost",
        "estimated_total_cost",
    ):
        usage[field] = _safe_float(value.get(field))
    currency = _safe_event_string(value.get("estimated_cost_currency"), max_chars=16)
    if currency:
        usage["estimated_cost_currency"] = currency
    upstream_model = _safe_event_string(value.get("upstream_model"), max_chars=160)
    if upstream_model:
        usage["upstream_model"] = upstream_model
    backend_model = _safe_event_string(value.get("backend_model"), max_chars=160)
    if backend_model:
        usage["backend_model"] = backend_model
    return usage


def _safe_cost_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    payload: dict[str, Any] = {
        "pricing_match_status": _safe_event_string(
            value.get("pricing_match_status"),
            max_chars=80,
        ),
        "estimated_cost_events": _safe_int(value.get("estimated_cost_events")),
        "estimated_input_cost": _safe_float(value.get("estimated_input_cost")),
        "estimated_output_cost": _safe_float(value.get("estimated_output_cost")),
        "estimated_cached_input_cost": _safe_float(
            value.get("estimated_cached_input_cost")
        ),
        "estimated_total_cost": _safe_float(value.get("estimated_total_cost")),
        "estimated_cost_currency": _safe_event_string(
            value.get("estimated_cost_currency"),
            max_chars=16,
        ),
        "pricing_catalog_version": _safe_int(value.get("pricing_catalog_version")),
        "pricing_catalog_source": _safe_event_string(
            value.get("pricing_catalog_source"),
            max_chars=160,
        ),
        "pricing_source": _safe_event_string(value.get("pricing_source"), max_chars=160),
        "pricing_effective_date": _safe_event_string(
            value.get("pricing_effective_date"),
            max_chars=40,
        ),
        "pricing_provider": _safe_event_string(value.get("pricing_provider")),
        "pricing_model": _safe_event_string(value.get("pricing_model")),
        "pricing_is_placeholder": value.get("pricing_is_placeholder") is True,
    }
    return payload


def _empty_usage_summary() -> dict[str, Any]:
    return {
        "events": 0,
        "usage_prompt_tokens": 0,
        "usage_completion_tokens": 0,
        "usage_total_tokens": 0,
        "usage_cached_input_tokens": 0,
        "estimated_cost_events": 0,
        "estimated_input_cost": 0.0,
        "estimated_output_cost": 0.0,
        "estimated_cached_input_cost": 0.0,
        "estimated_total_cost": 0.0,
        "estimated_cost_currency": None,
    }


def _safe_catalog_coverage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _empty_catalog_coverage()
    coverage = _empty_catalog_coverage()
    for field in (
        "total_routing_rows",
        "total_rows_with_usage",
        "rows_with_catalog_match",
        "rows_missing_provider_model_catalog_match",
        "rows_using_placeholder_pricing",
        "rows_with_estimated_cost",
        "rows_without_enough_usage_data",
        "active_catalog_version",
    ):
        coverage[field] = _safe_int(value.get(field))
    source = _safe_event_string(value.get("active_catalog_source"), max_chars=160)
    if source:
        coverage["active_catalog_source"] = source
    confidence = _safe_event_string(value.get("cost_confidence"), max_chars=80)
    if confidence:
        coverage["cost_confidence"] = confidence
    return coverage


def _safe_catalog_gap_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    gaps: list[dict[str, Any]] = []
    for item in value[:20]:
        if not isinstance(item, Mapping):
            continue
        gap = {
            "pricing_match_status": _safe_event_string(
                item.get("pricing_match_status"),
                max_chars=80,
            ),
            "provider": _safe_event_string(item.get("provider"), max_chars=120),
            "model": _safe_event_string(item.get("model"), max_chars=160),
            "backend": _safe_event_string(item.get("backend"), max_chars=120),
            "backend_model": _safe_event_string(
                item.get("backend_model"),
                max_chars=160,
            ),
            "upstream_model": _safe_event_string(
                item.get("upstream_model"),
                max_chars=160,
            ),
            "selected_engine": _safe_event_string(
                item.get("selected_engine"),
                max_chars=120,
            ),
            "events": _safe_int(item.get("events")),
            "usage_prompt_tokens": _safe_int(item.get("usage_prompt_tokens")),
            "usage_completion_tokens": _safe_int(item.get("usage_completion_tokens")),
            "usage_total_tokens": _safe_int(item.get("usage_total_tokens")),
            "usage_cached_input_tokens": _safe_int(
                item.get("usage_cached_input_tokens")
            ),
        }
        if gap["model"] or gap["backend"] or gap["provider"]:
            gaps.append(gap)
    return gaps


def _empty_catalog_coverage() -> dict[str, Any]:
    return {
        "total_routing_rows": 0,
        "total_rows_with_usage": 0,
        "rows_with_catalog_match": 0,
        "rows_missing_provider_model_catalog_match": 0,
        "rows_using_placeholder_pricing": 0,
        "rows_with_estimated_cost": 0,
        "rows_without_enough_usage_data": 0,
        "active_catalog_version": 0,
        "active_catalog_source": "",
        "cost_confidence": "no_usage",
    }


def _usage_has_tokens(usage: Mapping[str, Any]) -> bool:
    return any(
        _safe_int(usage.get(field)) > 0
        for field in (
            "usage_prompt_tokens",
            "usage_completion_tokens",
            "usage_total_tokens",
            "usage_cached_input_tokens",
        )
    )


def _safe_telemetry_rows(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, str]] = []
    for row in value[:20]:
        if not isinstance(row, Mapping):
            continue
        safe_row = {
            str(key): _safe_event_string(item, max_chars=120)
            for key, item in row.items()
            if isinstance(key, str) and isinstance(item, str)
        }
        if safe_row:
            rows.append(safe_row)
    return rows


def _recent_routing_events(events_path: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl(events_path)
    except Exception:
        rows = []
    events = [row for row in rows if row.get("event_type") == "routing_event"]
    recent: list[dict[str, Any]] = []
    for row in reversed(events):
        request_id = _safe_event_string(row.get("request_id"))
        selected_engine = _safe_event_string(row.get("selected_engine")) or "unknown"
        status = _safe_event_string(row.get("status")) or "unknown"
        backend = _safe_event_string(row.get("backend")) or "unassigned"
        backend_model = _safe_event_string(row.get("backend_model"))
        usage = _safe_usage_summary(event_usage_summary(row))
        recent.append(
            {
                "timestamp": _safe_event_string(row.get("timestamp")),
                "time": _short_event_time(row.get("timestamp")),
                "request_id": request_id,
                "route_api": _safe_event_string(row.get("route_api")),
                "selected_engine": selected_engine,
                "routing_profile": _safe_event_string(row.get("routing_profile")),
                "status": status,
                "backend": backend,
                "backend_model": backend_model,
                "upstream_model": _safe_event_string(row.get("upstream_model")),
                "usage": usage,
                "usage_tokens": _format_usage_summary(usage),
                "status_code": row.get("status_code")
                if isinstance(row.get("status_code"), int)
                else None,
                "route_latency": _format_latency(row.get("route_latency_ms")),
                "upstream_latency": _format_latency(row.get("upstream_latency_ms")),
                "total_latency": _format_latency(row.get("total_latency_ms")),
                "fallback_used": row.get("fallback_used") is True,
                "risk": _risk_label(row.get("risk_score")),
                "tools": _tools_label(row.get("requirements"), selected_engine),
                "confirmation": _confirmation_label(row, selected_engine),
                "privacy": _safe_event_string(row.get("privacy_explanation"))
                or "configured policy",
                "receipt_summary": _safe_event_string(row.get("receipt_summary")),
                "reason_codes": _safe_string_list(row.get("reason_codes"))[:8],
                "policy_explanation": _safe_event_string(row.get("policy_explanation")),
                "fallback_explanation": _safe_event_string(
                    row.get("fallback_explanation")
                ),
                "safety_explanation": _safe_event_string(row.get("safety_explanation")),
                "wrong_route_next_action": _safe_event_string(
                    row.get("wrong_route_next_action")
                ),
                "dot": _status_dot(status),
            }
        )
        if len(recent) >= limit:
            break
    return recent


def _route_map_state(
    config: RoutingProxyConfig | None,
    latest_event: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if config is None:
        return []
    selected_engine = str(latest_event.get("selected_engine") or "")
    engines = [
        engine
        for engine in DASHBOARD_ENGINE_ORDER
        if engine in config.engine_backends or engine == "human_confirm"
    ]
    engines.extend(
        sorted(
            engine
            for engine in config.engine_backends
            if engine not in engines
        )
    )
    rows: list[dict[str, Any]] = []
    for engine in engines:
        backend_name = config.engine_backends.get(engine)
        backend = config.backends.get(backend_name) if backend_name else None
        fallback_name = (
            config.fallback_backends.get(backend.name, ("",))[0]
            if backend and config.fallback_backends.get(backend.name)
            else ""
        )
        fallback_engine = _engine_for_backend(config, fallback_name) or fallback_name
        rejection = config.backend_policy_rejection_reason(backend.name) if backend else None
        rows.append(
            {
                "route_class": ROUTE_CLASS_BY_ENGINE.get(
                    engine,
                    engine.replace("_", " ").title(),
                ),
                "route_id": engine,
                "target": _route_target(engine, backend),
                "provider": _provider_runtime_label(backend),
                "latency": _configured_latency_label(backend),
                "cost": _configured_cost_label(backend),
                "privacy": _configured_privacy_label(backend),
                "tools": TOOL_HINT_BY_ENGINE.get(engine, "Limited"),
                "fallback": fallback_engine or "—",
                "selected": bool(selected_engine and engine == selected_engine),
                "policy_status": rejection or "allowed",
            }
        )
    return rows


def _provider_runtime_state(
    config: RoutingProxyConfig | None,
    latest_event: Mapping[str, Any],
) -> dict[str, Any]:
    if config is None:
        return {"providers": [], "detail": {}}
    selected_backend_name = str(latest_event.get("backend") or "")
    if latest_event.get("selected_engine") == "human_confirm":
        selected_backend_name = ""
    elif selected_backend_name not in config.backends:
        selected_backend_name = (
            config.engine_backends.get("code_agent")
            or config.engine_backends.get("balanced_local")
            or next(iter(config.backends), "")
        )
    providers: list[dict[str, Any]] = []
    for backend in config.backends.values():
        rejection = config.backend_policy_rejection_reason(backend.name)
        runtime = backend.runtime
        adapter = runtime_state_for_backend(
            backend,
            timeout_seconds=min(config.health.backend_timeout_seconds, 0.2),
        )
        health = adapter.get("health") if isinstance(adapter.get("health"), dict) else {}
        active = backend.name == selected_backend_name
        status = (
            "Policy denied"
            if rejection
            else (
                "Selected recently"
                if active and latest_event
                else str(health.get("status") or ("Managed" if runtime.enabled else "Configured"))
            )
        )
        providers.append(
            {
                "name": f"{_provider_runtime_label(backend)} / {backend.name}",
                "backend": backend.name,
                "status": status,
                "detail": _host_port_label(backend.base_url),
                "active": active,
                "dot": "red-dot"
                if rejection
                else _adapter_dot(adapter, active=active, managed=runtime.enabled),
                "icon": _provider_icon_name(_provider_runtime_label(backend)),
                "runtime_adapter": adapter,
            }
        )
    selected_backend = config.backends.get(selected_backend_name)
    return {
        "providers": providers,
        "detail": _runtime_detail_state(selected_backend, config),
        "selected_backend": selected_backend_name,
    }


def _runtime_detail_state(
    backend: Any,
    config: RoutingProxyConfig,
) -> dict[str, Any]:
    if backend is None:
        return {}
    runtime = backend.runtime
    adapter = runtime_state_for_backend(
        backend,
        timeout_seconds=min(config.health.backend_timeout_seconds, 0.2),
    )
    health = adapter.get("health") if isinstance(adapter.get("health"), dict) else {}
    capabilities = (
        adapter.get("capabilities") if isinstance(adapter.get("capabilities"), dict) else {}
    )
    models = adapter.get("models") if isinstance(adapter.get("models"), list) else []
    loaded = (
        adapter.get("loaded_models") if isinstance(adapter.get("loaded_models"), list) else []
    )
    discovered_model_ids = _runtime_model_ids(models)
    loaded_model_ids = _runtime_model_ids(loaded)
    logs = adapter.get("logs") if isinstance(adapter.get("logs"), dict) else {}
    fallback_chain = tuple(
        item.name for item in config.fallback_chain_for_backend(backend.name)
    )
    return {
        "backend": backend.name,
        "model": backend.model,
        "base_url": backend.base_url,
        "runtime_enabled": runtime.enabled,
        "runtime_kind": runtime.kind,
        "runtime_status": "managed-by-proxy" if runtime.enabled else "unmanaged",
        "adapter": adapter.get("adapter"),
        "adapter_provider": adapter.get("provider"),
        "runtime_id": adapter.get("runtime_id"),
        "runtime_mode": adapter.get("runtime_mode"),
        "detected": adapter.get("detected"),
        "endpoint": adapter.get("endpoint") or adapter.get("endpoint_url"),
        "version": adapter.get("version"),
        "missing_dependency": adapter.get("missing_dependency"),
        "install_hint": adapter.get("install_hint"),
        "last_checked_at": adapter.get("last_checked_at"),
        "health_status": health.get("status"),
        "health_detail": health.get("detail"),
        "health_ok": health.get("ok") is True,
        "discovered_models": discovered_model_ids,
        "loaded_models": loaded_model_ids,
        "model_guidance": _runtime_model_guidance(
            backend.model,
            adapter,
            discovered_model_ids,
        ),
        "capabilities": capabilities,
        "logs": logs,
        "runtime_command": " ".join(shlex.quote(part) for part in runtime.command)
        if runtime.command
        else "unmanaged backend; start it outside ModelRouter",
        "readiness_url": runtime.readiness_url,
        "idle_timeout": _idle_timeout_label(runtime.idle_timeout_seconds, runtime.enabled),
        "log_path": runtime.log_path if runtime.enabled else "",
        "fallback_chain": list(fallback_chain),
        "policy_status": config.backend_policy_rejection_reason(backend.name) or "allowed",
        "builder": _runtime_builder_fields(backend),
    }


def _runtime_model_ids(items: list[Any]) -> list[str]:
    ids: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("model_id") or item.get("id") or "").strip()
        if model_id:
            ids.append(model_id)
    return list(dict.fromkeys(ids))


def _runtime_model_ids_text(model_ids: list[str], *, limit: int = 6) -> str:
    if not model_ids:
        return "no models detected"
    shown = model_ids[:limit]
    suffix = f", +{len(model_ids) - limit} more" if len(model_ids) > limit else ""
    return ", ".join(shown) + suffix


def _runtime_model_guidance(
    configured_model: str,
    adapter: Mapping[str, Any],
    discovered_model_ids: list[str],
) -> str:
    provider = str(
        adapter.get("provider") or adapter.get("runtime_id") or adapter.get("runtime_kind") or ""
    ).lower()
    configured = configured_model.strip()
    if discovered_model_ids and configured in discovered_model_ids:
        return "Configured model is visible from the selected runtime."
    if discovered_model_ids:
        suggestions = _runtime_model_ids_text(discovered_model_ids)
        if provider == "lmstudio" or configured.startswith("lmstudio-"):
            return f"replace {configured} with one of: {suggestions}"
        if provider == "ollama":
            return (
                f"detected models: {suggestions}; replace {configured} with one "
                "of those ids or pull the configured model explicitly."
            )
        return f"configured model {configured} not detected; use one of: {suggestions}"
    if provider == "lmstudio":
        return "no models detected; start the LM Studio local server or configure endpoint."
    if provider == "ollama":
        return (
            "no models detected; start Ollama, run `ollama list`, or pull models "
            "explicitly."
        )
    return "no models detected; start the local server or configure endpoint."


def _route_receipt_state(
    config: RoutingProxyConfig | None,
    latest_event: Mapping[str, Any],
) -> dict[str, Any]:
    if not latest_event:
        return {
            "has_event": False,
            "summary": "No routing events yet. Start the proxy and send a request to see a real route receipt.",
            "selected": "none yet",
            "backend": "none yet",
            "model": "none yet",
            "reason": "waiting for first request",
            "risk": "n/a",
            "tools": "n/a",
            "fallback": "n/a",
            "confirmation": "n/a",
            "route_latency": "n/a",
            "upstream_latency": "n/a",
            "privacy": "privacy-safe defaults",
            "reason_codes": [],
            "request_id": "",
            "policy": "No request has been routed in this session.",
            "fallback_explanation": "No fallback data yet.",
            "safety": "Safety gates remain configured.",
            "wrong_route": "After a request, copy its request id and label it with model-router feedback.",
        }
    selected = str(latest_event.get("selected_engine") or "unknown")
    backend_name = str(latest_event.get("backend") or "")
    backend = config.backends.get(backend_name) if config and backend_name else None
    fallback = _fallback_for_event(config, selected, backend_name)
    return {
        "has_event": True,
        "summary": latest_event.get("receipt_summary")
        or f"Selected {selected}; status {latest_event.get('status', 'unknown')}.",
        "selected": selected,
        "backend": backend_name or "unassigned",
        "model": latest_event.get("backend_model") or (backend.model if backend else ""),
        "reason": _receipt_reason(latest_event),
        "risk": latest_event.get("risk") or "n/a",
        "tools": latest_event.get("tools") or "n/a",
        "fallback": fallback,
        "confirmation": latest_event.get("confirmation") or "n/a",
        "route_latency": latest_event.get("route_latency") or "n/a",
        "upstream_latency": latest_event.get("upstream_latency") or "n/a",
        "privacy": _privacy_short(latest_event.get("privacy")),
        "reason_codes": list(latest_event.get("reason_codes") or []),
        "request_id": latest_event.get("request_id") or "",
        "policy": latest_event.get("policy_explanation") or "Policy followed configured defaults.",
        "fallback_explanation": latest_event.get("fallback_explanation")
        or "Fallback details were not recorded for this event.",
        "safety": latest_event.get("safety_explanation")
        or "Safety details were not recorded for this event.",
        "wrong_route": latest_event.get("wrong_route_next_action")
        or "Copy this request id and label it with model-router feedback.",
    }


def _review_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    try:
        payload = review_queue(
            events_path=paths["events"],
            feedback_path=paths["feedback"],
            pricing_catalog_path=paths.get("pricing"),
            max_rows=8,
        )
        return _sanitize_review_state(payload, feedback_path=paths["feedback"])
    except Exception as exc:
        return {
            "reviewable": 0,
            "items": [],
            "truncated": False,
            "skipped_labeled": 0,
            "skipped_private": 0,
            "catalog_coverage": _empty_catalog_coverage(),
            "catalog_coverage_gaps": [],
            "pricing_override_skeleton": "",
            "error": str(exc),
            "privacy": (
                "Prompts, prompt previews, request bodies, feedback notes, and "
                "secrets are hidden by default."
            ),
        }


def _sanitize_review_state(
    payload: Mapping[str, Any],
    *,
    feedback_path: Path,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if not isinstance(item, Mapping):
            continue
        request_id = _safe_event_string(item.get("request_id"), max_chars=120)
        items.append(
            {
                "request_id": request_id,
                "selected_engine": _safe_event_string(item.get("selected_engine")),
                "status": _safe_event_string(item.get("status")),
                "backend": _safe_event_string(item.get("backend")),
                "backend_model": _safe_event_string(item.get("backend_model")),
                "upstream_model": _safe_event_string(item.get("upstream_model")),
                "usage": _safe_usage_summary(item.get("usage")),
                "usage_tokens": _format_usage_summary(item.get("usage")),
                "cost": _safe_cost_summary(item.get("cost")),
                "cost_estimate": _format_cost_summary(item.get("cost")),
                "routing_profile": _safe_event_string(item.get("routing_profile")),
                "receipt_summary": _safe_event_string(item.get("receipt_summary")),
                "reason_codes": _safe_string_list(item.get("reason_codes"))[:8],
                "replayable": item.get("replayable") is True,
                "suggested_feedback_command": (
                    "model-router feedback "
                    f"{request_id} <expected_engine> "
                    f"--output {feedback_path}"
                ),
            }
        )
    return {
        "reviewable": len(items),
        "items": items,
        "truncated": payload.get("truncated") is True,
        "skipped_labeled": _safe_int(payload.get("skipped_labeled")),
        "skipped_private": _safe_int(payload.get("skipped_private")),
        "catalog_coverage": _safe_catalog_coverage(payload.get("catalog_coverage")),
        "catalog_coverage_gaps": _safe_catalog_gap_list(
            payload.get("catalog_coverage_gaps")
        ),
        "pricing_override_skeleton": _safe_multiline_text(
            pricing_override_skeleton_from_gaps(
                _safe_catalog_gap_list(payload.get("catalog_coverage_gaps")),
            ),
            max_chars=12_000,
        ),
        "privacy": _safe_event_string(
            payload.get("privacy"),
            max_chars=320,
        )
        or (
            "Prompts, prompt previews, request bodies, feedback notes, and "
            "secrets are hidden by default."
        ),
    }


def _workflow_benchmark_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    path = paths.get("workflow_benchmarks")
    command = "model-router workflow-benchmark --json --fail-on-mismatch"
    if path is None or not path.exists():
        return {
            "status": "not run from settings",
            "path": str(path or ""),
            "command": command,
            "passed": None,
            "failed": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unreadable",
            "path": str(path),
            "command": command,
            "passed": None,
            "failed": None,
        }
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list):
        failed = sum(1 for item in results if isinstance(item, dict) and not item.get("ok"))
        passed = len(results) - failed
        status = "passing" if failed == 0 else "needs attention"
    else:
        passed = payload.get("passed") if isinstance(payload, dict) else None
        failed = payload.get("failed") if isinstance(payload, dict) else None
        status = str(payload.get("status") or "recorded") if isinstance(payload, dict) else "recorded"
    return {
        "status": status,
        "path": str(path),
        "command": command,
        "passed": passed,
        "failed": failed,
    }


def _model_options_state(discovery: Any, plan: DownloadPlan) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for model in getattr(discovery, "models", ())[:80]:
        repo_id = model.repo_id
        if repo_id in seen:
            continue
        seen.add(repo_id)
        options.append(
            {
                "value": repo_id,
                "label": f"{model.name} · local",
                "source": model.source,
                "path": model.path,
            }
        )
    for suggestion in plan.suggestions[:12]:
        repo_id = suggestion.repo_id
        if repo_id in seen:
            continue
        seen.add(repo_id)
        options.append(
            {
                "value": repo_id,
                "label": f"{repo_id} · recommended download",
                "source": suggestion.provider,
                "path": "",
            }
        )
    return options


def _safe_event_summary(events_path: Path) -> dict[str, Any]:
    engine_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    recent_request_ids: list[str] = []
    fallback_count = 0
    try:
        rows = read_jsonl(events_path)
    except Exception:
        rows = []
    for row in rows:
        if row.get("event_type") != "routing_event":
            continue
        request_id = _safe_event_string(row.get("request_id"), max_chars=120)
        if request_id:
            recent_request_ids.append(request_id)
        selected_engine = _safe_event_string(row.get("selected_engine"), max_chars=80)
        if selected_engine:
            engine_counts[selected_engine] += 1
        backend = _safe_event_string(row.get("backend"), max_chars=80)
        if backend:
            backend_counts[backend] += 1
        status = _safe_event_string(row.get("status"), max_chars=80)
        if status:
            status_counts[status] += 1
        if row.get("fallback_used") is True:
            fallback_count += 1
    return {
        "selected_engine_counts": dict(sorted(engine_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_count": fallback_count,
        "recent_request_ids": recent_request_ids[-10:],
    }


def _safe_event_string(value: Any, *, max_chars: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    text = redact_text(value).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _safe_multiline_text(value: Any, *, max_chars: int = 12_000) -> str:
    if not isinstance(value, str):
        return ""
    text = redact_text(value).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _safe_string_list(value: Any, *, max_chars: int = 120) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [
            _safe_event_string(item, max_chars=max_chars)
            for item in value
            if _safe_event_string(item, max_chars=max_chars)
        ]
    return []


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    return 0


def _safe_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return 0.0
    return round(float(value), 8)


def _short_event_time(value: Any) -> str:
    text = _safe_event_string(value)
    if "T" in text:
        text = text.split("T", 1)[1]
    if "." in text:
        text = text.split(".", 1)[0]
    if text.endswith("Z"):
        text = text[:-1]
    return text[:5] if text else "—"


def _format_latency(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    amount = float(value)
    if amount < 1:
        return f"{amount * 1000:.1f} us"
    if amount < 1000:
        return f"{amount:.1f} ms" if amount < 10 else f"{amount:.0f} ms"
    return f"{amount / 1000:.1f} s"


def _risk_label(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    score = float(value)
    if score >= 70:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _tools_label(requirements: Any, selected_engine: str) -> str:
    if selected_engine == "human_confirm":
        return "N/A"
    if isinstance(requirements, dict) and requirements.get("needs_tools") is True:
        return "required"
    return TOOL_HINT_BY_ENGINE.get(selected_engine, "limited")


def _confirmation_label(row: Mapping[str, Any], selected_engine: str) -> str:
    if selected_engine == "human_confirm":
        return "required"
    reason_codes = set(_safe_string_list(row.get("reason_codes")))
    if "safety.confirmation_required" in reason_codes:
        return "required"
    return "not required"


def _status_dot(status: str) -> str:
    lowered = status.lower()
    if lowered in {"forwarded", "completed", "ok"}:
        return "green-dot"
    if lowered in {"blocked", "confirm", "runtime_start_failed"}:
        return "yellow-dot"
    if lowered in {"error", "failed", "upstream_error"}:
        return "red-dot"
    return "yellow-dot"


def _engine_for_backend(config: RoutingProxyConfig, backend_name: str) -> str:
    for engine, mapped_backend in config.engine_backends.items():
        if mapped_backend == backend_name:
            return engine
    return ""


def _route_target(engine: str, backend: Any) -> str:
    if backend is None:
        return ROUTE_DESCRIPTION_BY_ENGINE.get(engine, engine.replace("_", " "))
    description = ROUTE_DESCRIPTION_BY_ENGINE.get(engine, engine.replace("_", " "))
    return f"{description}: {backend.model}"


def _provider_runtime_label(backend: Any) -> str:
    if backend is None:
        return "Safety Gate"
    runtime = backend.runtime
    if runtime.enabled:
        return {
            "llama-server": "llama.cpp",
            "mlx-lm": "MLX-LM",
            "generic": "Managed runtime",
        }.get(runtime.kind, runtime.kind)
    parsed = urlparse(backend.base_url)
    host = parsed.hostname or ""
    port = parsed.port
    if host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"} and port == 11434:
        return "Ollama"
    if host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"} and port == 1234:
        return "LM Studio"
    if _is_local_base_url(backend.base_url):
        return "OpenAI-compatible local"
    if "openai" in host:
        return "OpenAI-compatible hosted"
    return "OpenAI-compatible"


def _configured_latency_label(backend: Any) -> str:
    if backend is None:
        return "Very Low"
    if backend.runtime.enabled:
        return "On demand"
    if _is_local_base_url(backend.base_url):
        return "Unmeasured"
    return "Hosted"


def _configured_cost_label(backend: Any) -> str:
    if backend is None:
        return "$ Low"
    return "$ Low" if _is_local_base_url(backend.base_url) else "$$ Hosted"


def _configured_privacy_label(backend: Any) -> str:
    if backend is None:
        return "Local only"
    return "Local only" if _is_local_base_url(backend.base_url) else "Hosted"


def _is_local_base_url(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _host_port_label(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or parsed.netloc or base_url
    return f"{host}:{parsed.port}" if parsed.port else host


def _idle_timeout_label(seconds: Any, enabled: bool) -> str:
    if not enabled:
        return "not managed"
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return "managed"
    minutes = max(0, float(seconds)) / 60
    if minutes >= 1:
        return f"{minutes:g} minutes (idle unload enabled)"
    return f"{seconds:g} seconds (idle unload enabled)"


def _runtime_builder_fields(backend: Any) -> dict[str, str]:
    runtime = backend.runtime
    command = list(runtime.command)
    return {
        "kind": runtime.kind,
        "model": _flag_arg(command, "--model")
        or _flag_arg(command, "-m")
        or backend.model,
        "port": _flag_arg(command, "--port") or _port_from_url(backend.base_url),
        "context_size": _flag_arg(command, "-c")
        or _flag_arg(command, "--ctx-size")
        or "",
        "gpu_layers": _flag_arg(command, "-ngl")
        or _flag_arg(command, "--n-gpu-layers")
        or "",
        "argv_preview": " ".join(shlex.quote(part) for part in command),
    }


def _flag_arg(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return command[index + 1]


def _port_from_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    return str(parsed.port or "")


def _fallback_for_event(
    config: RoutingProxyConfig | None,
    selected_engine: str,
    backend_name: str,
) -> str:
    if config is None:
        return "not configured"
    if backend_name:
        chain = config.fallback_backends.get(backend_name, ())
        if chain:
            return _engine_for_backend(config, chain[0]) or chain[0]
    backend = config.engine_backends.get(selected_engine)
    if backend:
        chain = config.fallback_backends.get(backend, ())
        if chain:
            return _engine_for_backend(config, chain[0]) or chain[0]
    return "not configured"


def _receipt_reason(event: Mapping[str, Any]) -> str:
    codes = list(event.get("reason_codes") or [])
    for prefix, reason in (
        ("route.coding", "coding and repository intent detected"),
        ("route.research", "fresh research or current information"),
        ("route.vision", "vision or OCR handling"),
        ("route.image_generation", "image generation route"),
        ("route.confirmation", "human confirmation gate selected"),
        ("route.reasoning", "higher-complexity reasoning"),
        ("route.simple", "simple local request"),
    ):
        if prefix in codes:
            return reason
    return str(event.get("status") or "configured route selected")


def _privacy_short(value: Any) -> str:
    text = _safe_event_string(value)
    lowered = text.lower()
    if not text:
        return "configured policy"
    if "local-only" in lowered or "local only" in lowered:
        return "local-only"
    if "hosted" in lowered:
        return "hosted allowed"
    return "configured policy"


def _download_plan(
    paths: Mapping[str, Path],
    *,
    discovery=None,
    benchmark_results=None,
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


def _config_notice(config_error: str | None) -> str:
    if not config_error:
        return ""
    return (
        '<section class="notice error">'
        "<strong>Config needs attention.</strong><br>"
        f"{escape(config_error)}"
        "</section>"
    )


def _backend_row(backend: Mapping[str, Any]) -> str:
    runtime = backend["runtime"]
    command = " ".join(shlex.quote(part) for part in runtime.get("command", []))
    status_class = "ok" if runtime.get("enabled") else "warn"
    name = escape(str(backend["name"]))
    route = escape(_route_for_backend(str(backend["name"])))
    model = escape(str(backend["model"]))
    base_url = escape(str(backend["base_url"]))
    runtime_enabled = _bool_options(runtime.get("enabled"))
    runtime_kind = _options(
        ["generic", "llama-server", "mlx-lm"],
        selected=runtime.get("kind"),
    )
    readiness_url = escape(str(runtime.get("readiness_url") or ""))
    idle_timeout = escape(str(runtime.get("idle_timeout_seconds") or ""))
    log_path = escape(str(runtime.get("log_path") or ""))
    runtime_status = escape(str(runtime.get("status") or "unmanaged"))
    return f"""<tr data-backend="{name}">
      <td><strong>{name}</strong><br><span class="muted">{route}</span></td>
      <td><input data-field="model" value="{model}"></td>
      <td><input data-field="base_url" value="{base_url}"></td>
      <td>
        <select data-field="runtime_enabled">{runtime_enabled}</select>
        <select data-field="runtime_kind">{runtime_kind}</select>
        <textarea data-field="runtime_command">{escape(command)}</textarea>
        <input data-field="readiness_url" value="{readiness_url}" placeholder="readiness URL">
        <input data-field="idle_timeout_seconds" value="{idle_timeout}" placeholder="idle seconds">
        <input data-field="log_path" value="{log_path}" placeholder="log path">
      </td>
      <td><span class="badge {status_class}">{runtime_status}</span></td>
    </tr>"""


def _route_for_backend(backend_name: str) -> str:
    return ROUTE_LABELS.get(backend_name, "backend")


def _recommendation_row(item: Mapping[str, Any]) -> str:
    score = item.get("score") if isinstance(item.get("score"), dict) else {}
    label = escape(str(score.get("label", "unscored")))
    overall = escape(str(score.get("overall_score", "n/a")))
    warnings = score.get("warnings") if isinstance(score.get("warnings"), list) else []
    warning_text = ", ".join(str(warning) for warning in warnings)
    warning_html = (
        f"<br><span class=\"muted\">{escape(warning_text)}</span>"
        if warning_text
        else ""
    )
    return (
        f"<tr><td>{escape(str(item.get('route', '')))}</td>"
        f"<td><span class=\"mono\">{escape(str(item.get('repo_id', '')))}</span></td>"
        f"<td>{label}{warning_html}</td><td>{overall}</td></tr>"
    )


def _compact_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(
        f"{key}:{count}"
        for key, count in sorted(value.items(), key=lambda item: str(item[0]))
    )


def _compact_usage_groups(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts: list[str] = []
    for key, usage in sorted(value.items(), key=lambda item: str(item[0])):
        formatted = _format_usage_summary(usage)
        cost = _format_cost_summary(usage)
        if formatted != "none":
            parts.append(f"{key}:{formatted}" + (f" cost={cost}" if cost != "none" else ""))
    return ", ".join(parts) if parts else "none"


def _format_usage_summary(value: Any) -> str:
    usage = _safe_usage_summary(value)
    if not _usage_has_tokens(usage):
        return "none"
    parts = [
        f"p={usage['usage_prompt_tokens']}",
        f"c={usage['usage_completion_tokens']}",
        f"t={usage['usage_total_tokens']}",
    ]
    if usage["usage_cached_input_tokens"]:
        parts.append(f"cache={usage['usage_cached_input_tokens']}")
    return " ".join(parts)


def _format_cost_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    cost = _safe_cost_summary(value)
    status = cost.get("pricing_match_status")
    if status and status != "matched":
        return str(status)
    total = _safe_float(cost.get("estimated_total_cost"))
    events = _safe_int(cost.get("estimated_cost_events"))
    currency = _safe_event_string(cost.get("estimated_cost_currency"), max_chars=16)
    if total == 0 and events == 0:
        return "none"
    amount = f"{total:.8f}".rstrip("0").rstrip(".") or "0"
    parts = [part for part in (amount, currency) if part]
    if cost.get("pricing_is_placeholder") is True:
        parts.append("placeholder")
    return " ".join(parts)


def _format_catalog_coverage(value: Any) -> str:
    coverage = _safe_catalog_coverage(value)
    parts = [
        f"usage={coverage['total_rows_with_usage']}",
        f"matched={coverage['rows_with_catalog_match']}",
        f"missing={coverage['rows_missing_provider_model_catalog_match']}",
        f"placeholder={coverage['rows_using_placeholder_pricing']}",
        f"estimated={coverage['rows_with_estimated_cost']}",
        f"no_usage={coverage['rows_without_enough_usage_data']}",
    ]
    version = coverage.get("active_catalog_version")
    if isinstance(version, int) and version > 0:
        parts.append(f"v{version}")
    source = coverage.get("active_catalog_source")
    if isinstance(source, str) and source:
        parts.append(source)
    confidence = coverage.get("cost_confidence")
    if isinstance(confidence, str) and confidence:
        parts.append(confidence)
    return " ".join(parts)


def _format_catalog_gap_list(value: Any) -> str:
    gaps = _safe_catalog_gap_list(value)
    if not gaps:
        return "none"
    parts: list[str] = []
    for gap in gaps[:5]:
        status = gap.get("pricing_match_status") or "unknown"
        model = gap.get("model") or "unknown-model"
        backend = gap.get("backend") or "unknown-backend"
        provider = gap.get("provider") or "unknown-provider"
        events = _safe_int(gap.get("events"))
        total = _safe_int(gap.get("usage_total_tokens"))
        parts.append(
            f"{provider}/{model}@{backend} {status} events={events} t={total}"
        )
    if len(gaps) > 5:
        parts.append(f"+{len(gaps) - 5} more")
    return "; ".join(parts)


def _pricing_override_skeleton_block(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    skeleton = _safe_multiline_text(value.get("pricing_override_skeleton"))
    if not skeleton:
        return ""
    skeleton_js = json.dumps(skeleton)
    return f"""<details>
          <summary>Pricing override skeleton</summary>
          <p class="muted">Generated from catalog coverage gaps. Prices are placeholders; verify provider terms before using estimates.</p>
          <button type="button" onclick="copyText({skeleton_js})">Copy override skeleton</button>
          <pre class="catalog-output">{escape(skeleton)}</pre>
        </details>"""


def _download_row(item: Mapping[str, Any]) -> str:
    route = escape(str(item["route"]))
    repo = escape(str(item["repo_id"]))
    route_js = escape(json.dumps(str(item["route"])))
    repo_js = escape(json.dumps(str(item["repo_id"])))
    score = item.get("score") if isinstance(item.get("score"), dict) else {}
    label = score.get("label")
    score_text = (
        f"<br><span class=\"muted\">{escape(str(label))} "
        f"{escape(str(score.get('overall_score', '')))}</span>"
        if label
        else ""
    )
    return (
        f"<tr><td>{route}</td><td><span class=\"mono\">{repo}</span>{score_text}</td>"
        f"<td><button onclick=\"runDownload({route_js}, {repo_js})\">Download</button></td></tr>"
    )


def _benchmark_best_summary(benchmarks: Mapping[str, Any]) -> str:
    best = benchmarks.get("best")
    if not isinstance(best, list) or not best:
        return "none"
    first = best[0]
    if not isinstance(first, dict):
        return "none"
    model = first.get("model") or first.get("backend") or "unknown"
    tps = first.get("tokens_per_second")
    return f"{model}: {tps} tok/s" if tps is not None else str(model)


def _options(values: list[str] | tuple[str, ...], *, selected: Any) -> str:
    selected_text = str(selected or "")
    return "\n".join(
        "<option "
        + ("selected " if value == selected_text else "")
        + f'value="{escape(value)}">{escape(value)}</option>'
        for value in values
    )


def _backend_options(backends: Any, *, selected: Any) -> str:
    values = [
        str(backend.get("name"))
        for backend in backends
        if isinstance(backend, Mapping) and backend.get("name")
    ]
    return '<option value="">not set</option>' + _options(values, selected=selected)


def _bool_options(selected: Any) -> str:
    selected_bool = bool(selected)
    return (
        f'<option value="true" {"selected" if selected_bool else ""}>enabled</option>'
        f'<option value="false" {"selected" if not selected_bool else ""}>disabled</option>'
    )
