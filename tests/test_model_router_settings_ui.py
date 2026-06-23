import json
from pathlib import Path
import subprocess
from typing import Any

from fastapi.testclient import TestClient
import yaml

import hermes.plugins.model_router.settings_ui as settings_ui
from hermes.plugins.model_router.model_benchmark import BenchmarkResult, BenchmarkTarget
from hermes.plugins.model_router.product import initialize_product_config
from hermes.plugins.model_router.proxy_config import load_proxy_config
from hermes.plugins.model_router.setup_assistant import DiscoveredModel, SetupDiscovery


class _FakeProcess:
    pid = 4321

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _DoctorReport:
    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "proxy_config_valid": True,
            "router_config_valid": True,
            "proxy_config": "/tmp/routing_proxy.yaml",
            "router_config": "/tmp/model_router.yaml",
            "backends": [],
            "errors": [],
            "proxy_endpoint": "http://127.0.0.1:8082/v1",
            "telemetry_log_path": "/tmp/events.jsonl",
            "remediation": ["No action needed."],
        }


def _init_config(tmp_path: Path) -> None:
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )


def _stub_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_ui,
        "scan_local_environment",
        lambda: SetupDiscovery(
            commands={"ollama": False, "hf": True},
            env_vars={"OPENAI_API_KEY": False},
            model_dirs=(),
            models=(
                DiscoveredModel(
                    name="Qwen3-4B",
                    repo_id="mlx-community/Qwen3-4B-4bit",
                    path="/models/qwen",
                    source="local_directory",
                    roles=("balanced_local",),
                ),
            ),
        ),
    )


def test_settings_state_redacts_literal_api_keys(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    config_path = tmp_path / "routing_proxy.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["proxy"]["api_key"] = "secret-proxy-key"
    data["backends"]["fast"]["api_key"] = "secret-backend-key"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    state = settings_ui.build_settings_state(settings_ui.settings_paths(tmp_path))
    serialized = json.dumps(state, sort_keys=True)

    assert "secret-proxy-key" not in serialized
    assert "secret-backend-key" not in serialized
    assert state["proxy"]["api_key_configured"] is True
    fast = next(backend for backend in state["backends"] if backend["name"] == "fast")
    assert fast["api_key_configured"] is True


def test_settings_home_page_loads_without_chat_surface(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "ModelRouter Settings" in response.text
    assert "No chat surface" in response.text
    assert "chat transcript" not in response.text.lower()
    assert "textarea id=\"chat" not in response.text.lower()


def test_settings_state_api_includes_models_and_downloads(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    events_path = tmp_path / "logs" / "routing-events.jsonl"
    events_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event_type": "routing_event",
                    "request_id": "req-1",
                    "selected_engine": "fast_local",
                    "backend": "fast",
                    "status": "forwarded",
                    "fallback_used": False,
                },
                {
                    "event_type": "routing_event",
                    "request_id": "req-2",
                    "selected_engine": "code_agent",
                    "backend": "code",
                    "status": "forwarded",
                    "fallback_used": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    payload = TestClient(app).get("/api/state").json()

    assert payload["config_valid"] is True
    assert payload["discovery"]["models"][0]["repo_id"] == "mlx-community/Qwen3-4B-4bit"
    assert payload["download_plan"]["suggestions"]
    assert payload["recommendation"]["local_model_recommendations"]
    assert "score" in payload["recommendation"]["local_model_recommendations"][0]
    assert "score" in payload["download_plan"]["suggestions"][0]
    assert payload["benchmarks"]["results"] == 0
    assert payload["telemetry"]["backend_counts"] == {"code": 1, "fast": 1}
    assert payload["telemetry"]["fallback_count"] == 1
    assert payload["telemetry"]["recent_request_ids"] == ["req-1", "req-2"]


def test_save_config_can_apply_preset_template_explicitly(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post(
        "/api/save-config",
        json={"apply_preset": True, "preset": "ollama"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["preset"] == "ollama"
    config = load_proxy_config(tmp_path / "routing_proxy.yaml")
    assert config.backends["fast"].base_url == "http://127.0.0.1:11434/v1"


def test_save_config_patches_structured_fields_and_validates(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post(
        "/api/save-config",
        json={
            "proxy": {"host": "127.0.0.1", "port": "9099"},
            "observability": {
                "enabled": True,
                "prompt_capture": "off",
                "log_path": str(tmp_path / "events.jsonl"),
            },
            "backends": {
                "fast": {
                    "model": "new-fast-model",
                    "base_url": "http://127.0.0.1:9999/v1",
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    config = load_proxy_config(tmp_path / "routing_proxy.yaml")
    assert config.proxy.port == 9099
    assert config.backends["fast"].model == "new-fast-model"
    assert config.backends["fast"].base_url == "http://127.0.0.1:9999/v1"
    assert config.observability.prompt_capture == "off"


def test_doctor_api_returns_structured_report(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    monkeypatch.setattr(settings_ui, "doctor_proxy_config", lambda _path: _DoctorReport())
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post("/api/doctor")

    assert response.status_code == 200
    assert response.json()["remediation"] == ["No action needed."]


def test_proxy_supervisor_endpoints_start_stop_restart_without_shell(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    commands: list[list[str]] = []
    kwargs: list[dict[str, Any]] = []
    processes: list[_FakeProcess] = []

    def factory(command: list[str], **process_kwargs: Any) -> _FakeProcess:
        process = _FakeProcess()
        commands.append(command)
        kwargs.append(process_kwargs)
        processes.append(process)
        return process

    supervisor = settings_ui.ProxyProcessSupervisor(
        config_path=tmp_path / "routing_proxy.yaml",
        log_path=tmp_path / "logs" / "settings-proxy.log",
        process_factory=factory,
    )
    app = settings_ui.create_settings_app(
        config_dir=tmp_path,
        proxy_supervisor=supervisor,
    )
    client = TestClient(app)

    started = client.post("/api/proxy/start").json()
    restarted = client.post("/api/proxy/restart").json()
    stopped = client.post("/api/proxy/stop").json()

    assert started["proxy"]["state"] == "running"
    assert restarted["proxy"]["state"] == "running"
    assert stopped["proxy"]["state"] == "stopped"
    assert commands[0][0].endswith("python") or "python" in commands[0][0]
    assert commands[0][1:3] == ["-m", "hermes.plugins.model_router.proxy"]
    assert kwargs[0]["shell"] is False
    assert kwargs[0]["stdin"] == subprocess.DEVNULL
    assert processes[0].terminated is True


def test_download_run_requires_explicit_confirmation(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    calls: list[tuple[str, ...]] = []
    app = settings_ui.create_settings_app(
        config_dir=tmp_path,
        download_runner=lambda command: calls.append(command) or 0,
    )
    client = TestClient(app)

    blocked = client.post(
        "/api/download/run",
        json={"route": "balanced_local", "repo_id": "org/model"},
    )
    allowed = client.post(
        "/api/download/run",
        json={"confirm": True, "route": "balanced_local", "repo_id": "org/model"},
    )

    assert blocked.status_code == 400
    assert allowed.status_code == 200
    assert calls
    assert calls[0][:3] == ("hf", "download", "org/model")


def test_benchmark_run_requires_confirmation_and_stores_metrics(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    calls: list[str] = []

    def runner(target: BenchmarkTarget, _timeout: float) -> BenchmarkResult:
        calls.append(target.backend)
        return BenchmarkResult(
            backend=target.backend,
            route=target.route,
            model=target.model,
            base_url=target.base_url,
            runtime_kind=target.runtime_kind,
            managed_runtime=target.managed_runtime,
            status="completed",
            timestamp="2026-06-22T00:00:00.000Z",
            total_latency_ms=100.0,
            tokens_per_second=25.0,
            measured_tokens=10,
        )

    app = settings_ui.create_settings_app(
        config_dir=tmp_path,
        benchmark_runner=runner,
    )
    client = TestClient(app)

    blocked = client.post("/api/benchmark/run", json={})
    planned = client.post("/api/benchmark/plan")
    allowed = client.post("/api/benchmark/run", json={"confirm": True})
    text = (tmp_path / "benchmarks.json").read_text(encoding="utf-8")

    assert blocked.status_code == 400
    assert planned.status_code == 200
    assert planned.json()["targets"]
    assert allowed.status_code == 200
    assert calls == ["fast", "balanced", "reasoning", "code"]
    assert "completed" in text
    assert "Reply with one short sentence" not in text


def test_feedback_api_writes_cli_compatible_jsonl(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post(
        "/api/feedback",
        json={
            "request_id": "req-123",
            "expected_engine": "code_agent",
            "notes": "should have used code",
        },
    )

    row = json.loads((tmp_path / "routing-feedback.jsonl").read_text(encoding="utf-8"))
    assert response.status_code == 200
    assert row["event_type"] == "routing_feedback"
    assert row["request_id"] == "req-123"
    assert row["expected_engine"] == "code_agent"
    assert row["notes"] == "should have used code"
