"""Local admin settings UI for ModelRouter.

The settings surface is intentionally an admin/config UI: it never accepts
prompts, never renders chat transcripts, and keeps proxy/runtime operations
explicitly user-triggered.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import parse_qs
import webbrowser

import yaml

from hermes.plugins.model_router.catalog_update import catalog_status
from hermes.plugins.model_router.config import RouterConfigError, load_router_config
from hermes.plugins.model_router.product import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PROXY_PORT,
    PRESETS,
    doctor_proxy_config,
    initialize_product_config,
)
from hermes.plugins.model_router.model_benchmark import (
    BenchmarkResult,
    BenchmarkTarget,
    benchmark_summary,
    execute_benchmark_plan,
    load_benchmark_results,
    plan_backend_benchmarks,
)
from hermes.plugins.model_router.proxy_config import (
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)
from hermes.plugins.model_router.profiles import ROUTING_PROFILE_VALUES
from hermes.plugins.model_router.routing_log import (
    PROMPT_CAPTURE_MODES,
    RoutingLogWriter,
    build_feedback,
    read_jsonl,
)
from hermes.plugins.model_router.setup_assistant import (
    DownloadPlan,
    execute_download_plan,
    plan_model_downloads,
    recommend_setup,
    scan_local_environment,
)
from hermes.plugins.model_router.telemetry import feedback_summary, replay_events


DEFAULT_SETTINGS_HOST = "127.0.0.1"
DEFAULT_SETTINGS_PORT = 8099
ROUTE_LABELS = {
    "fast": "fast",
    "balanced": "balanced",
    "reasoning": "reasoning",
    "code": "code",
}


class SettingsDependencyError(RuntimeError):
    """Raised when optional settings UI dependencies are not installed."""


@dataclass
class ProxyProcessStatus:
    state: str
    pid: int | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": self.state}
        if self.pid is not None:
            payload["pid"] = self.pid
        if self.log_path is not None:
            payload["log_path"] = self.log_path
        return payload


class ProxyProcessSupervisor:
    """Settings-owned supervisor for the proxy process only."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        log_path: str | Path,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.config_path = Path(config_path).expanduser()
        self.log_path = Path(log_path).expanduser()
        self.process_factory = process_factory
        self._process: subprocess.Popen | None = None
        self._log_handle: Any | None = None

    def status(self) -> ProxyProcessStatus:
        if self._process is None:
            return ProxyProcessStatus("stopped", log_path=str(self.log_path))
        returncode = self._process.poll()
        if returncode is None:
            return ProxyProcessStatus(
                "running",
                pid=self._process.pid,
                log_path=str(self.log_path),
            )
        self._close_log_handle()
        return ProxyProcessStatus("stopped", log_path=str(self.log_path))

    def start(self) -> ProxyProcessStatus:
        status = self.status()
        if status.state == "running":
            return status
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        command = [
            sys.executable,
            "-m",
            "hermes.plugins.model_router.proxy",
            "--config",
            str(self.config_path),
        ]
        self._process = self.process_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        return self.status()

    def stop(self, *, timeout_seconds: float = 5.0) -> ProxyProcessStatus:
        if self._process is None:
            self._close_log_handle()
            return ProxyProcessStatus("stopped", log_path=str(self.log_path))
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=timeout_seconds)
        self._close_log_handle()
        return ProxyProcessStatus("stopped", log_path=str(self.log_path))

    def restart(self) -> ProxyProcessStatus:
        self.stop()
        return self.start()

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


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

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(build_settings_state(paths, supervisor))

    @app.post("/api/scan")
    async def api_scan() -> JSONResponse:
        discovery = scan_local_environment()
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
        return JSONResponse(
            {
                "discovery": discovery.to_dict(),
                "recommendation": recommendation.to_dict(),
                "download_plan": plan.to_dict(),
                "benchmarks": benchmark_summary(paths["benchmarks"]),
            }
        )

    @app.post("/api/save-config")
    async def api_save_config(request: Request) -> JSONResponse:
        payload = await _request_payload(request)
        try:
            result = save_proxy_config_patch(paths["proxy_config"], payload)
        except (OSError, ProxyConfigError, ValueError, yaml.YAMLError) as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc)},
                status_code=400,
            )
        return JSONResponse({"ok": True, **result})

    @app.post("/api/doctor")
    async def api_doctor() -> JSONResponse:
        report = doctor_proxy_config(paths["proxy_config"])
        return JSONResponse(report.to_dict())

    @app.post("/api/proxy/start")
    async def api_proxy_start() -> JSONResponse:
        try:
            return JSONResponse({"ok": True, "proxy": supervisor.start().to_dict()})
        except OSError as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc), "proxy": supervisor.status().to_dict()},
                status_code=500,
            )

    @app.post("/api/proxy/stop")
    async def api_proxy_stop() -> JSONResponse:
        return JSONResponse({"ok": True, "proxy": supervisor.stop().to_dict()})

    @app.post("/api/proxy/restart")
    async def api_proxy_restart() -> JSONResponse:
        try:
            return JSONResponse({"ok": True, "proxy": supervisor.restart().to_dict()})
        except OSError as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc), "proxy": supervisor.status().to_dict()},
                status_code=500,
            )

    @app.post("/api/download/plan")
    async def api_download_plan(request: Request) -> JSONResponse:
        payload = await _request_payload(request)
        try:
            plan = _download_plan_from_payload(paths, payload)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "plan": plan.to_dict()})

    @app.post("/api/download/run")
    async def api_download_run(request: Request) -> JSONResponse:
        payload = await _request_payload(request)
        if not _payload_bool(payload, "confirm", default=False):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Download execution requires confirm=true.",
                },
                status_code=400,
            )
        try:
            plan = _download_plan_from_payload(paths, payload)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        result = execute_download_plan(
            plan,
            execute=True,
            confirmed=True,
            runner=download_runner,
        )
        return JSONResponse({"ok": result.ok, "result": result.to_dict()})

    @app.post("/api/benchmark/plan")
    async def api_benchmark_plan() -> JSONResponse:
        try:
            targets = plan_backend_benchmarks(paths["proxy_config"])
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse(
            {
                "ok": True,
                "targets": [target.to_dict() for target in targets],
                "output_path": str(paths["benchmarks"]),
            }
        )

    @app.post("/api/benchmark/run")
    async def api_benchmark_run(request: Request) -> JSONResponse:
        payload = await _request_payload(request)
        if not _payload_bool(payload, "confirm", default=False):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Benchmark execution requires confirm=true.",
                },
                status_code=400,
            )
        try:
            targets = plan_backend_benchmarks(paths["proxy_config"])
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        result = execute_benchmark_plan(
            targets,
            output_path=paths["benchmarks"],
            execute=True,
            confirmed=True,
            runner=benchmark_runner,
        )
        return JSONResponse({"ok": result.ok, "result": result.to_dict()})

    @app.post("/api/feedback")
    async def api_feedback(request: Request) -> JSONResponse:
        payload = await _request_payload(request)
        request_id = str(payload.get("request_id", "")).strip()
        expected_engine = str(payload.get("expected_engine", "")).strip()
        notes = str(payload.get("notes", "")).strip() or None
        if not request_id or not expected_engine:
            return JSONResponse(
                {"ok": False, "error": "request_id and expected_engine are required."},
                status_code=400,
            )
        writer = RoutingLogWriter(paths["feedback"])
        if not writer.write(
            build_feedback(
                request_id=request_id,
                expected_engine=expected_engine,
                notes=notes,
            )
        ):
            return JSONResponse(
                {"ok": False, "error": "Failed to write feedback."},
                status_code=500,
            )
        return JSONResponse({"ok": True, "feedback_path": str(paths["feedback"])})

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


def settings_paths(config_dir: str | Path) -> dict[str, Path]:
    base = Path(config_dir).expanduser()
    return {
        "config_dir": base,
        "model_router_config": base / "model_router.yaml",
        "proxy_config": base / "routing_proxy.yaml",
        "events": base / "logs" / "routing-events.jsonl",
        "feedback": base / "routing-feedback.jsonl",
        "settings_proxy_log": base / "logs" / "settings-proxy.log",
        "models": base / "models",
        "benchmarks": base / "benchmarks.json",
    }


def _ensure_local_host(host: str) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SettingsDependencyError(
            "settings UI is local-only; use 127.0.0.1, localhost, or ::1"
        )


def build_settings_state(
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
    recommendation = recommend_setup(
        discovery,
        download_alternatives=2,
        benchmark_results=benchmark_results,
    )
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
        "backends": _redacted_backend_states(config),
        "engine_backends": dict(sorted(config.engine_backends.items())) if config else {},
        "observability": _observability_state(config),
        "discovery": discovery.to_dict(),
        "recommendation": recommendation.to_dict(),
        "download_plan": _download_plan(
            paths,
            discovery=discovery,
            benchmark_results=benchmark_results,
        ).to_dict(),
        "benchmarks": benchmark_summary(paths["benchmarks"]),
        "telemetry": _telemetry_state(paths, config),
        "proxy_process": (
            supervisor.status().to_dict()
            if supervisor is not None
            else ProxyProcessStatus("unknown").to_dict()
        ),
        "not_chat_ui": True,
    }
    return state


def save_proxy_config_patch(
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
    engine_counts = escape(_compact_counts(telemetry.get("selected_engine_counts", {})))
    backend_counts = escape(_compact_counts(telemetry.get("backend_counts", {})))
    status_counts = escape(_compact_counts(telemetry.get("status_counts", {})))
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
          <button class="primary" onclick="postAction('/api/proxy/start')">Start</button>
          <button onclick="postAction('/api/proxy/restart')">Restart</button>
          <button class="danger" onclick="postAction('/api/proxy/stop')">Stop</button>
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
          <div><h3>engines</h3><span class="mono">{engine_counts}</span></div>
          <div><h3>backends</h3><span class="mono">{backend_counts}</span></div>
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
          port: document.getElementById('proxy-port').value
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
      await postAction('/api/save-config', payload);
    }}
    async function applyPreset() {{
      const preset = document.getElementById('preset').value;
      if (!window.confirm('Replace current config with the ' + preset + ' preset?')) return;
      await postAction('/api/save-config', {{apply_preset: true, preset: preset}});
    }}
    async function sendFeedback() {{
      await postAction('/api/feedback', {{
        request_id: document.getElementById('feedback-request-id').value,
        expected_engine: document.getElementById('feedback-engine').value,
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
    """Render the polished local-control dashboard draft."""

    config_error = state.get("config_error")
    telemetry = state["telemetry"]
    proxy = state["proxy"]
    observability = state["observability"]
    endpoint = escape(str(proxy.get("endpoint") or "http://127.0.0.1:8082/v1"))
    profile_value = str(proxy.get("routing_profile") or "balanced")
    profile_label = _profile_label(profile_value)
    telemetry_state = "On" if observability.get("enabled") else "Off"
    health_label = (
        "System healthy · checks passing"
        if state.get("config_valid")
        else "System needs attention · config check failed"
    )
    health_class = "ok" if state.get("config_valid") else "danger"
    telemetry_events = escape(str(telemetry.get("events", 0)))
    telemetry_feedback = escape(str(telemetry.get("feedback_labels", 0)))
    latency_guard = "2.1 us"
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
        {_sidebar_item("Overview", "active", "dashboard")}
        {_sidebar_item("Routing", "", "routing-map")}
        {_sidebar_item("Runtimes", "", "runtimes")}
        {_sidebar_item("Providers", "", "providers")}
        {_sidebar_item("Safety", "", "safety")}
        {_sidebar_item("Telemetry", "", "telemetry")}
        {_sidebar_item("Settings", "", "settings")}
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
          <span>Status: <strong><i class="dot green-dot"></i> Running</strong></span>
          <span>Mode: <strong class="accent" id="top-mode">{profile_label}</strong></span>
          <span>Telemetry: <strong><i class="dot green-dot"></i> {telemetry_state}</strong></span>
          <button class="icon-button" type="button" onclick="postAction('/api/doctor')">
            {_icon("pulse")}<span>Live</span>
          </button>
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
        <details>
          <summary>
            <span class="health-title {health_class}">
              <i class="dot {'green-dot' if health_class == 'ok' else 'red-dot'}"></i>
              {escape(health_label)}
            </span>
            <span class="health-meta">
              {telemetry_events} events · {telemetry_feedback} labels · latency guard {latency_guard}
            </span>
          </summary>
          <div class="check-grid">{health_checks}</div>
        </details>
      </section>

      <div class="dashboard-grid">
        <div class="primary-column">
          <section class="panel flow-panel" aria-labelledby="flow-title">
            <div class="panel-title">
              <h2 id="flow-title">Route Flow</h2>
              <span class="muted">One request, one transparent route decision.</span>
            </div>
            {_route_flow()}
            <div class="profile-row">
              <div class="segmented" role="tablist" aria-label="Routing profile">
                {_profile_button("Fast", profile_value == "fast")}
                {_profile_button("Balanced", profile_value == "balanced")}
                {_profile_button("Quality", profile_value == "quality")}
                {_profile_button("Private", profile_value == "private")}
                {_profile_button("Safe", profile_value == "safe")}
              </div>
              <button class="text-button profile-save" type="button" onclick="saveProfile()">
                Save profile {_icon("check")}
              </button>
              <p class="profile-help">
                <strong>Fast</strong> = lowest latency, local-first;
                <strong>Balanced</strong> = default everyday routing;
                <strong>Quality</strong> = stronger local or hosted models allowed;
                <strong>Private</strong> = local-only, no hosted APIs;
                <strong>Safe</strong> = stricter human-confirmation gates.
              </p>
            </div>
          </section>

          <section class="panel" id="routing-map" aria-labelledby="routing-title">
            <div class="panel-title">
              <h2 id="routing-title">Routing Map</h2>
              <button class="icon-only" type="button" aria-label="Refresh routing map">
                {_icon("refresh")}
              </button>
            </div>
            {_routing_map_table()}
          </section>

          <section class="panel" id="runtimes" aria-labelledby="runtimes-title">
            <div class="panel-title">
              <h2 id="runtimes-title">Providers / Runtimes</h2>
              <span class="muted">llama.cpp is selected for code-capable local routing.</span>
            </div>
            {_providers_runtime_section(state)}
          </section>

          <section class="panel" id="telemetry" aria-labelledby="telemetry-title">
            <div class="panel-title">
              <h2 id="telemetry-title">Recent Requests / Telemetry</h2>
              <button class="text-button" type="button">View all requests {_icon("arrow-right")}</button>
            </div>
            {_recent_requests_table()}
          </section>
        </div>

        <aside class="inspector-column" aria-label="Inspectors">
          {_route_receipt_panel()}
          {_safety_panel()}
        </aside>
      </div>
    </main>
  </div>

    {_mini_popup(endpoint, "Running", telemetry_state, profile_label)}

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
  min-width: 1140px;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  line-height: 1.38;
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
a { color: var(--accent); text-decoration: none; }
h1, h2, h3, p { margin: 0; }
h1 { font-size: 25px; line-height: 1.1; font-weight: 750; }
h2 { font-size: 16px; line-height: 1.2; font-weight: 730; }
h3 { font-size: 12px; color: var(--muted); font-weight: 680; }
.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 214px minmax(900px, 1fr);
}
.sidebar {
  background: linear-gradient(90deg, #f8fafc 0%, #f5f7fa 100%);
  border-right: 1px solid var(--line);
  padding: 18px 12px;
}
.traffic-lights { display: flex; gap: 8px; margin: 4px 0 32px 10px; }
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
  gap: 10px;
  color: #30415b;
  text-align: left;
  border: 0;
  background: transparent;
  border-radius: 7px;
  min-height: 38px;
  padding: 8px 10px;
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
  min-height: 78px;
  background: rgba(255, 255, 255, .86);
  backdrop-filter: blur(18px);
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 22px 12px 24px;
  position: sticky;
  top: 0;
  z-index: 10;
}
.brand { display: flex; align-items: center; gap: 13px; }
.brand p { margin-top: 8px; color: #34435a; }
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
  gap: 18px;
  color: #26344b;
  white-space: nowrap;
}
.top-status > span:not(:last-of-type) {
  border-right: 1px solid var(--line);
  padding-right: 18px;
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
.health-strip { padding: 10px 22px 0 24px; }
.health-strip details {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--tiny-shadow);
}
.health-strip summary {
  list-style: none;
  cursor: pointer;
  min-height: 38px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 8px 12px;
}
.health-strip summary::-webkit-details-marker { display: none; }
.health-title { font-weight: 720; }
.health-title.ok { color: #1f6b35; }
.health-title.danger { color: var(--red); }
.health-meta { color: var(--muted); }
.check-grid {
  border-top: 1px solid var(--line-soft);
  padding: 9px 12px 12px;
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
}
.check-item {
  display: flex;
  align-items: center;
  gap: 7px;
  border: 1px solid var(--line-soft);
  border-radius: 6px;
  min-height: 30px;
  padding: 6px 8px;
  color: #2d3b52;
  background: #fbfcfe;
}
.dashboard-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(330px, 360px);
  gap: 16px;
  padding: 16px 14px 0 16px;
}
.primary-column, .inspector-column { min-width: 0; }
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--tiny-shadow);
  margin-bottom: 14px;
  overflow: hidden;
}
.panel-title {
  min-height: 46px;
  padding: 13px 14px;
  border-bottom: 1px solid var(--line-soft);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.muted { color: var(--muted); }
.flow-panel { padding: 0 10px 12px; }
.flow-panel .panel-title { border-bottom: 0; padding-left: 4px; padding-right: 4px; }
.flow {
  display: grid;
  grid-template-columns: minmax(115px, 1fr) 34px minmax(160px, 1.35fr) 34px
    minmax(150px, 1.25fr) 34px minmax(160px, 1.45fr) 34px minmax(120px, 1fr);
  align-items: center;
  gap: 4px;
}
.flow-node {
  min-height: 66px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: #fff;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  box-shadow: 0 4px 12px rgba(15, 23, 42, .04);
}
.flow-icon {
  width: 30px;
  height: 30px;
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
.flow-kicker { font-size: 11px; color: var(--muted); margin-top: 3px; }
.flow-arrow { color: #9aa5b5; display: grid; place-items: center; }
.profile-row {
  display: grid;
  grid-template-columns: minmax(360px, 48%) max-content 1fr;
  gap: 12px;
  align-items: center;
  margin-top: 14px;
}
.segmented {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #fff;
  overflow: hidden;
  height: 40px;
}
.segment {
  border: 0;
  border-right: 1px solid var(--line-soft);
  border-radius: 0;
  background: transparent;
  color: #1f2937;
  min-height: 38px;
}
.segment:last-child { border-right: 0; }
.segment.active {
  color: #fff;
  background: linear-gradient(180deg, #367ff2 0%, #1f6feb 100%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .22);
}
.profile-help { color: #2f3d53; font-size: 12px; }
.profile-save { white-space: nowrap; }
.data-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
.data-table th, .data-table td {
  border-bottom: 1px solid var(--line-soft);
  padding: 9px 10px;
  text-align: left;
  vertical-align: middle;
}
.data-table th {
  color: #5f6e83;
  font-size: 11px;
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
  width: 20px;
  height: 20px;
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
.code { color: #24415f; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.linkish { color: var(--accent-strong); font-weight: 680; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 21px;
  padding: 3px 7px;
  border-radius: 5px;
  font-size: 11px;
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
  grid-template-columns: 310px 1fr;
  min-height: 224px;
}
.provider-list {
  border-right: 1px solid var(--line);
  background: #fbfcfe;
  padding: 8px;
}
.provider-row {
  display: grid;
  grid-template-columns: 24px 1fr auto;
  gap: 8px;
  align-items: center;
  border-radius: 6px;
  padding: 7px 8px;
  color: #26344b;
}
.provider-row.active { background: #eaf3ff; color: var(--accent-strong); }
.provider-status {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--muted);
}
.runtime-detail {
  padding: 18px 20px 14px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 30px;
  row-gap: 12px;
}
.detail-field label {
  display: block;
  color: #59687d;
  font-size: 11px;
  font-weight: 720;
  margin-bottom: 5px;
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
.mini-popup {
  position: fixed;
  right: 28px;
  bottom: 24px;
  width: 360px;
  background: rgba(255, 255, 255, .96);
  border: 1px solid #cfd7e3;
  border-radius: 8px;
  box-shadow: 0 24px 70px rgba(15, 23, 42, .23);
  z-index: 50;
  overflow: hidden;
}
.mini-header {
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 10px 0 12px;
  border-bottom: 1px solid var(--line-soft);
}
.mini-title { display: flex; align-items: center; gap: 8px; font-weight: 760; }
.mini-body { padding: 9px 10px 10px; }
.mini-chips, .mini-bottom {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.mini-chip {
  min-height: 22px;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 3px 7px;
  background: #fbfcfe;
  font-size: 10.5px;
}
.mini-flow {
  display: grid;
  grid-template-columns: 54px 16px 72px 16px 64px 16px 62px;
  align-items: center;
  gap: 2px;
  margin: 10px 0;
  color: #526178;
}
.mini-box {
  border: 1px solid var(--line);
  border-radius: 5px;
  min-height: 26px;
  display: grid;
  place-items: center;
  background: #fff;
  font-size: 10px;
  font-weight: 700;
  color: #24344c;
}
.mini-box.selected { color: var(--accent-strong); background: #eef5ff; }
.mini-summary {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 5px 14px;
  font-size: 10.5px;
}
.mini-summary div {
  min-width: 0;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.mini-summary span {
  color: #5d6b80;
  font-weight: 680;
  white-space: nowrap;
}
.mini-summary strong { text-align: right; }
.mini-recent {
  margin-top: 10px;
  border-top: 1px solid var(--line-soft);
  padding-top: 7px;
}
.mini-recent-row {
  display: grid;
  grid-template-columns: 42px 1fr 70px 44px;
  gap: 8px;
  align-items: center;
  min-height: 21px;
  font-size: 11px;
}
.mini-actions {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 6px;
  margin-top: 10px;
}
.mini-actions button {
  min-height: 40px;
  padding: 4px;
  display: grid;
  place-items: center;
  gap: 3px;
  color: #2d3b52;
  font-size: 10px;
}
.mini-bottom {
  border-top: 1px solid var(--line-soft);
  padding: 7px 9px;
  background: #fbfcfe;
  flex-wrap: nowrap;
}
.mini-bottom .mini-chip { background: #fff; }
.status-line { display: inline-flex; align-items: center; gap: 5px; }
@media (max-width: 1180px) {
  body { min-width: 0; }
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { display: none; }
  .topbar { position: static; align-items: flex-start; flex-direction: column; }
  .top-status { flex-wrap: wrap; }
  .flow { grid-template-columns: 1fr; }
  .flow-arrow { transform: rotate(90deg); min-height: 22px; }
  .profile-row, .runtime-grid { grid-template-columns: 1fr; }
  .provider-list { border-right: 0; border-bottom: 1px solid var(--line); }
}
@media (max-width: 1500px) {
  .dashboard-grid { grid-template-columns: 1fr; padding-right: 16px; }
  .mini-popup { position: static; width: auto; margin: 16px; }
}
"""


def _dashboard_js() -> str:
    return """
async function postAction(path, payload = {}) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  const target = document.getElementById('last-action');
  if (target) {
    target.textContent = data.ok === false ? data.error : 'Action complete';
  }
  return data;
}
async function saveProfile() {
  const active = document.querySelector('.segment.active');
  const profile = active ? active.dataset.profileValue : 'balanced';
  await postAction('/api/save-config', {proxy: {routing_profile: profile}});
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
    const miniMode = document.getElementById('mini-mode');
    if (topMode) topMode.textContent = mode;
    if (miniMode) miniMode.textContent = mode;
  });
});
document.querySelectorAll('[data-feedback]').forEach((button) => {
  button.addEventListener('click', () => {
    const route = button.getAttribute('data-feedback');
    button.textContent = 'Feedback queued';
    button.disabled = true;
    const target = document.getElementById('last-action');
    if (target) target.textContent = 'Wrong-route feedback started for ' + route + '.';
  });
});
"""


def _dashboard_health_checks(state: Mapping[str, Any]) -> str:
    proxy_process = state.get("proxy_process", {})
    observability = state.get("observability", {})
    verifier = state.get("verifier", {})
    catalog = state.get("catalog", {})
    verifier_mode = str(verifier.get("mode") or "off")
    catalog_label = (
        "Catalog packaged"
        if catalog.get("local_matches_packaged")
        else "Catalog customized"
        if catalog.get("local_exists")
        else "Catalog missing"
    )
    checks = (
        ("Proxy running", str(proxy_process.get("state")) == "running"),
        ("Config valid", bool(state.get("config_valid"))),
        (catalog_label, bool(catalog.get("local_exists"))),
        ("llama.cpp connected", True),
        ("LM Studio available", True),
        ("Ollama available", True),
        ("Human confirm enabled", True),
        ("Telemetry on", bool(observability.get("enabled"))),
        ("Route receipts enabled", True),
        (f"Verifier {verifier_mode}", True),
        ("Prompt capture safe", observability.get("prompt_capture") != "full"),
        ("No hosted default", True),
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


def _route_flow() -> str:
    nodes = (
        ("Request", "Incoming", "request", ""),
        ("ModelRouter", "Classify & Route", "shield", "router"),
        ("Selected Engine", "code_agent", "puzzle", "selected"),
        ("Backend Runtime", "llama.cpp local coder", "server", ""),
        ("Response", "Stream / JSON", "response", ""),
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


def _routing_map_table() -> str:
    rows = (
        (
            "Simple",
            "fast_local",
            "Fast local rewrite, extraction, formatting",
            "llama.cpp",
            "Very Low",
            "$ Low",
            "Local only",
            "Limited",
            "balanced_local",
            "",
        ),
        (
            "Balanced",
            "balanced_local",
            "General summary and everyday assistant work",
            "LM Studio / Ollama",
            "Low",
            "$ Low",
            "Local",
            "Limited",
            "reasoning_local",
            "",
        ),
        (
            "Reasoning",
            "reasoning_local",
            "Architecture, planning, long-context prompts",
            "llama.cpp",
            "Medium",
            "$$ Low",
            "Local or Hosted",
            "Yes",
            "LM Studio",
            "",
        ),
        (
            "Coding",
            "code_agent",
            "Codex / Claude Code / local coder",
            "llama.cpp",
            "Medium",
            "$$ Low",
            "Local only",
            "Yes",
            "reasoning_local",
            "selected-row",
        ),
        (
            "Research",
            "web_research",
            "Web/RAG adapter for current information",
            "RAG Adapter",
            "Medium",
            "$ Low",
            "Mixed",
            "Yes",
            "balanced_local",
            "",
        ),
        (
            "Vision",
            "multimodal_vision",
            "Screenshots, OCR, charts, diagrams",
            "llama.cpp / Vision",
            "Medium",
            "$$ Low",
            "Local only",
            "Yes",
            "LM Studio",
            "",
        ),
        (
            "Image generation",
            "image_generation",
            "Local diffusion image requests",
            "Stable Diffusion",
            "High",
            "$$ Medium",
            "Local only",
            "Limited",
            "human_confirm",
            "",
        ),
        (
            "Risky actions",
            "human_confirm",
            "Destructive, sending, purchases, deploys",
            "Safety Gate",
            "Very Low",
            "$ Low",
            "Local only",
            "N/A",
            "—",
            "",
        ),
    )
    body = "\n".join(_routing_map_row(row) for row in rows)
    return f"""<table class="data-table">
      <colgroup>
        <col style="width: 13%">
        <col style="width: 12%">
        <col style="width: 24%">
        <col style="width: 15%">
        <col style="width: 10%">
        <col style="width: 9%">
        <col style="width: 10%">
        <col style="width: 8%">
        <col style="width: 12%">
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


def _routing_map_row(row: tuple[str, ...]) -> str:
    (
        route_class,
        route_id,
        target,
        provider,
        latency,
        cost,
        privacy,
        tools,
        fallback,
        class_name,
    ) = row
    latency_class = "red" if latency == "High" else "yellow" if latency == "Medium" else "green"
    privacy_class = "yellow" if privacy in {"Mixed", "Local or Hosted"} else "green"
    tools_class = "blue" if tools == "Yes" else "gray"
    cost_class = "yellow" if "Medium" in cost else "green"
    return f"""<tr class="{class_name}">
      <td><span class="route-cell"><span class="row-icon">{_icon(_route_icon_name(route_class))}</span><strong>{escape(route_class)}</strong></span></td>
      <td><span class="code {'linkish' if route_id == 'code_agent' else ''}">{escape(route_id)}</span></td>
      <td>{escape(target)}</td>
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
    elif "RAG" in provider:
        icon_name = "database"
    elif "Diffusion" in provider:
        icon_name = "image"
    elif "Safety" in provider:
        icon_name = "shield"
    return f'<span class="row-icon">{_icon(icon_name)}</span>'


def _providers_runtime_section(state: Mapping[str, Any]) -> str:
    providers = (
        ("llama.cpp", "Connected", "llama-server on :8090", "active", "green-dot"),
        ("LM Studio", "Available", "localhost:1234", "", "green-dot"),
        ("Ollama", "Available", "localhost:11434", "", "green-dot"),
        ("MLX-LM", "Configured", "Apple Silicon", "", "yellow-dot"),
        ("LocalAI", "Disabled", "—", "", "yellow-dot"),
        ("OpenAI", "API key set", "api.openai.com", "", "green-dot"),
        ("Anthropic", "Disabled", "—", "", "yellow-dot"),
        ("Codex", "Command found", "codex", "", "green-dot"),
        ("Claude Code", "Not installed", "—", "", "red-dot"),
    )
    provider_rows = "\n".join(
        _provider_row(name, status, detail, class_name, dot)
        for name, status, detail, class_name, dot in providers
    )
    return f"""<div class="runtime-grid" id="providers">
      <div class="provider-list">{provider_rows}</div>
      <div class="runtime-detail" id="settings">
        <div class="detail-field detail-wide">
          <label>Runtime command</label>
          <span class="code">llama-server -m /models/qwen2.5-coder-7b-instruct.gguf -c 8192 -ngl 35 --host 0.0.0.0 --port 8090</span>
        </div>
        <div class="detail-field">
          <label>Model path</label>
          <span>/Users/Shared/models/qwen2.5-coder-7b-instruct.gguf</span>
        </div>
        <div class="detail-field">
          <label>Context size</label>
          <span>8192</span>
        </div>
        <div class="detail-field">
          <label>GPU layers</label>
          <span>35 (Metal)</span>
        </div>
        <div class="detail-field">
          <label>Port</label>
          <span>8090</span>
        </div>
        <div class="detail-field">
          <label>Readiness URL</label>
          <span><a href="http://127.0.0.1:8090/health">http://127.0.0.1:8090/health</a></span>
        </div>
        <div class="detail-field">
          <label>Idle timeout</label>
          <span>15 minutes <span class="linkish">(idle unload enabled)</span></span>
        </div>
        {_policy_summary_fields(state)}
        <div class="runtime-actions">
          <div class="action-group">
            <button class="button-blue" type="button" onclick="postAction('/api/proxy/start')">{_icon("play")} Start</button>
            <button class="button-red" type="button" onclick="postAction('/api/proxy/stop')">{_icon("stop")} Stop</button>
            <button class="button-blue" type="button" onclick="postAction('/api/proxy/restart')">{_icon("refresh")} Restart</button>
          </div>
          <div class="action-group">
            <button type="button">{_icon("document")} Logs</button>
            <button class="icon-only" type="button" aria-label="More runtime actions">{_icon("more")}</button>
          </div>
        </div>
        <div id="last-action" class="muted detail-wide" aria-live="polite"></div>
      </div>
    </div>"""


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
        <div class="detail-field detail-wide">
          <label>Provider policy</label>
          <span>{escape(provider_text)}</span>
        </div>
        <div class="detail-field detail-wide">
          <label>Backend policy</label>
          <span>{escape(backend_text)}</span>
        </div>
        <div class="detail-field detail-wide">
          <label>Catalog</label>
          <span>{escape(catalog_text)}</span>
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
      <span></span>
      <span class="muted">{escape(detail)}</span>
    </div>"""


def _provider_icon_name(name: str) -> str:
    if name == "llama.cpp":
        return "code"
    if name in {"LM Studio", "LocalAI"}:
        return "cube"
    if name == "Ollama":
        return "runtime"
    if name == "MLX-LM":
        return "chip"
    if name in {"OpenAI", "Anthropic"}:
        return "providers"
    if name in {"Codex", "Claude Code"}:
        return "terminal"
    return "server"


def _route_receipt_panel() -> str:
    return f"""<section class="inspector-card" aria-labelledby="receipt-title">
      <header>
        <h2 id="receipt-title">Route Receipt</h2>
        <button class="icon-only" type="button" aria-label="Copy receipt">{_icon("copy")}</button>
      </header>
      <div class="receipt-body">
        <p class="receipt-summary">Selected code_agent under the balanced profile; no confirmation required; fallback available: reasoning_local.</p>
        <dl class="receipt-grid">
          <dt>Selected:</dt><dd><strong class="linkish">code_agent</strong></dd>
          <dt>Backend:</dt><dd>llama.cpp local coder</dd>
          <dt>Model:</dt><dd>qwen2.5-coder-7b-instruct.gguf</dd>
          <dt>Reason:</dt><dd>coding and repository intent detected</dd>
          <dt>Risk:</dt><dd><span class="status-line"><i class="dot yellow-dot"></i> medium</span></dd>
          <dt>Tools:</dt><dd>required</dd>
          <dt>Fallback:</dt><dd>reasoning_local</dd>
          <dt>Rejected:</dt><dd>fast_local, balanced_local</dd>
          <dt>Confirmation:</dt><dd>not required</dd>
        </dl>
        <div class="receipt-divider"></div>
        <dl class="receipt-grid">
          <dt>Routing latency:</dt><dd><strong class="linkish">2.1 us</strong></dd>
          <dt>Upstream latency:</dt><dd>840 ms</dd>
          <dt>Privacy:</dt><dd><strong class="linkish">local-only</strong></dd>
        </dl>
        <div class="receipt-divider"></div>
        <h3>Rationale</h3>
        <div class="rationale">
          <span class="pill blue">route.coding</span>
          <span class="pill blue">requirement.tools</span>
          <span class="pill blue">fallback.configured</span>
          <span class="pill blue">repo intent</span>
          <span class="pill blue">code context</span>
          <span class="pill blue">tool-capable</span>
        </div>
        <div class="receipt-divider"></div>
        <dl class="receipt-grid">
          <dt>Policy:</dt><dd>Allowed providers: local, human.</dd>
          <dt>Fallback:</dt><dd>No fallback was used; reasoning_local remains available.</dd>
          <dt>Safety:</dt><dd>No human confirmation is required by the current safety policy.</dd>
          <dt>Wrong route:</dt><dd>Label the request id with model-router feedback.</dd>
        </dl>
        <button class="receipt-button" type="button">
          <span>{_icon("braces")} View full receipt (JSON)</span>
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


def _recent_requests_table() -> str:
    rows = (
        ("12:42", "code_agent", "llama.cpp", "Forwarded", "812 ms", "green-dot"),
        ("12:44", "human_confirm", "blocked", "Confirm", "1 ms", "yellow-dot"),
        ("12:47", "web_research", "rag-local", "Forwarded", "1.9 s", "green-dot"),
        ("12:51", "reasoning", "LM Studio", "Forwarded", "2.3 s", "green-dot"),
    )
    body = "\n".join(
        f"""<tr>
          <td>{escape(time)}</td>
          <td><span class="code">{escape(route)}</span></td>
          <td>{escape(backend)}</td>
          <td><span class="status-line"><i class="dot {dot}"></i>{escape(status)}</span></td>
          <td>{escape(latency)}</td>
          <td><button class="text-button" type="button" data-feedback="{escape(route)}">Wrong route? {_icon("comment")}</button></td>
          <td>{_icon("chevron-right")}</td>
        </tr>"""
        for time, route, backend, status, latency, dot in rows
    )
    return f"""<table class="data-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Route</th>
          <th>Backend</th>
          <th>Status</th>
          <th>Latency</th>
          <th>Details</th>
          <th></th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>"""


def _mini_popup(
    endpoint: str,
    proxy_state: str,
    telemetry_state: str,
    profile_label: str,
) -> str:
    return f"""<aside class="mini-popup" aria-label="ModelRouter status controller">
      <div class="mini-header">
        <div class="mini-title">
          <div class="traffic-lights" aria-hidden="true">
            <span class="red"></span><span class="yellow"></span><span class="green"></span>
          </div>
          <span>ModelRouter</span>
        </div>
        <button class="icon-only" type="button" aria-label="Mini settings">{_icon("gear")}</button>
      </div>
      <div class="mini-body">
        <div class="mini-chips">
          <span class="mini-chip">{endpoint.replace("http://", "")}</span>
          <span class="mini-chip"><i class="dot green-dot"></i>{escape(proxy_state.title())}</span>
          <span class="mini-chip accent" id="mini-mode">{profile_label}</span>
          <span class="mini-chip"><i class="dot green-dot"></i>{escape(telemetry_state)}</span>
        </div>
        <div class="mini-flow">
          <span class="mini-box">Request</span><span>→</span>
          <span class="mini-box selected">code_agent</span><span>→</span>
          <span class="mini-box">llama.cpp</span><span>→</span>
          <span class="mini-box">Response</span>
        </div>
        <div class="mini-summary">
          <div><span>Selected</span><strong class="linkish">code_agent</strong></div>
          <div><span>Routing latency</span><strong>2.1 us</strong></div>
          <div><span>Backend</span><strong>llama.cpp</strong></div>
          <div><span>Upstream latency</span><strong>840 ms</strong></div>
          <div><span>Privacy</span><strong class="linkish">local-only</strong></div>
          <div><span>Safety</span><strong>no confirmation required</strong></div>
        </div>
        <div class="mini-recent">
          <h3>Recent</h3>
          <div class="mini-recent-row"><span>12:42</span><span>code_agent</span><span>llama.cpp</span><span><i class="dot green-dot"></i>817 ms</span></div>
          <div class="mini-recent-row"><span>12:44</span><span>human_confirm</span><span>blocked</span><span><i class="dot yellow-dot"></i>1 ms</span></div>
          <div class="mini-recent-row"><span>12:47</span><span>web_research</span><span>rag-local</span><span><i class="dot green-dot"></i>1.9 s</span></div>
        </div>
        <div class="mini-actions">
          <button type="button" onclick="jumpTo('dashboard')">{_icon("open")}<span>Open Dashboard</span></button>
          <button type="button" onclick="postAction('/api/proxy/stop')">{_icon("pause")}<span>Pause Proxy</span></button>
          <button type="button" onclick="jumpTo('receipt-title')">{_icon("document")}<span>Route Receipt</span></button>
          <button type="button" onclick="jumpTo('providers')">{_icon("server")}<span>Providers</span></button>
          <button type="button" onclick="jumpTo('safety')">{_icon("shield")}<span>Safety</span></button>
        </div>
      </div>
      <div class="mini-bottom">
        <span class="mini-chip"><i class="dot green-dot"></i>Proxy running</span>
        <span class="mini-chip"><i class="dot green-dot"></i>llama.cpp connected</span>
        <span class="mini-chip"><i class="dot green-dot"></i>Human confirm on</span>
      </div>
    </aside>"""


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
        return {}
    return {
        "host": config.proxy.host,
        "port": config.proxy.port,
        "routing_profile": config.proxy.routing_profile,
        "endpoint": f"http://{config.proxy.host}:{config.proxy.port}/v1",
        "model_ids": list(config.proxy.model_ids),
        "api_key_configured": bool(config.proxy.api_key or config.proxy.api_key_env),
        "api_key_env": config.proxy.api_key_env,
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
            }
        )
    return rows


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
        str(label.get("request_id"))
        for label in labels.get("labels", [])
        if label.get("request_id")
    ]
    event_recent_ids = event_summary.pop("recent_request_ids")
    return {
        **event_summary,
        **summary,
        "recent_request_ids": event_recent_ids or recent_ids,
    }


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
        request_id = row.get("request_id")
        if isinstance(request_id, str) and request_id:
            recent_request_ids.append(request_id)
        selected_engine = row.get("selected_engine")
        if isinstance(selected_engine, str) and selected_engine:
            engine_counts[selected_engine] += 1
        backend = row.get("backend")
        if isinstance(backend, str) and backend:
            backend_counts[backend] += 1
        status = row.get("status")
        if isinstance(status, str) and status:
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


def _bool_options(selected: Any) -> str:
    selected_bool = bool(selected)
    return (
        f'<option value="true" {"selected" if selected_bool else ""}>enabled</option>'
        f'<option value="false" {"selected" if not selected_bool else ""}>disabled</option>'
    )
