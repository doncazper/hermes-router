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

from hermes.plugins.model_router.product import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PROXY_PORT,
    PRESETS,
    doctor_proxy_config,
    initialize_product_config,
)
from hermes.plugins.model_router.proxy_config import (
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)
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
        return HTMLResponse(render_settings_page(build_settings_state(paths, supervisor)))

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(build_settings_state(paths, supervisor))

    @app.post("/api/scan")
    async def api_scan() -> JSONResponse:
        discovery = scan_local_environment()
        recommendation = recommend_setup(discovery, download_alternatives=2)
        plan = _download_plan(paths)
        return JSONResponse(
            {
                "discovery": discovery.to_dict(),
                "recommendation": recommendation.to_dict(),
                "download_plan": plan.to_dict(),
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
    recommendation = recommend_setup(discovery, download_alternatives=2)
    state: dict[str, Any] = {
        "product": "ModelRouter",
        "paths": {name: str(path) for name, path in paths.items()},
        "presets": list(PRESETS),
        "prompt_capture_modes": list(PROMPT_CAPTURE_MODES),
        "config_exists": proxy_config_path.exists(),
        "config_valid": config is not None,
        "config_error": config_error,
        "proxy": _redacted_proxy_state(config),
        "backends": _redacted_backend_states(config),
        "engine_backends": dict(sorted(config.engine_backends.items())) if config else {},
        "observability": _observability_state(config),
        "discovery": discovery.to_dict(),
        "recommendation": recommendation.to_dict(),
        "download_plan": _download_plan(paths, discovery=discovery).to_dict(),
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
    telemetry = state["telemetry"]
    proxy = state["proxy"]
    observability = state["observability"]
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
        <h2>Per-route Backends</h2>
        <table id="backend-table">
          <thead>
            <tr><th>Route</th><th>Model</th><th>Base URL</th><th>Runtime</th><th>Status</th></tr>
          </thead>
          <tbody>{backend_rows}</tbody>
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
  </script>
</body>
</html>"""


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
        "endpoint": f"http://{config.proxy.host}:{config.proxy.port}/v1",
        "model_ids": list(config.proxy.model_ids),
        "api_key_configured": bool(config.proxy.api_key or config.proxy.api_key_env),
        "api_key_env": config.proxy.api_key_env,
    }


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
) -> DownloadPlan:
    return plan_model_downloads(
        discovery=discovery,
        alternatives=2,
        local_root=paths["models"],
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
    )


def _patch_proxy_config_data(data: dict[str, Any], payload: Mapping[str, Any]) -> None:
    proxy_patch = _mapping(payload.get("proxy"))
    proxy = data.setdefault("proxy", {})
    if proxy_patch:
        _patch_string(proxy, proxy_patch, "host")
        _patch_int(proxy, proxy_patch, "port")
        if "model_ids" in proxy_patch:
            proxy["model_ids"] = _string_list(proxy_patch["model_ids"])

    obs_patch = _mapping(payload.get("observability"))
    observability = data.setdefault("observability", {})
    if obs_patch:
        _patch_bool(observability, obs_patch, "enabled")
        _patch_string(observability, obs_patch, "log_path")
        _patch_string(observability, obs_patch, "prompt_capture")

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
    return (
        f"<tr><td>{route}</td><td><span class=\"mono\">{repo}</span></td>"
        f"<td><button onclick=\"runDownload({route_js}, {repo_js})\">Download</button></td></tr>"
    )


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
