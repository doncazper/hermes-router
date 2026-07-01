import json
from pathlib import Path
import subprocess
from typing import Any

from fastapi.testclient import TestClient
import pytest
import yaml

import hermes.plugins.model_router.settings_ui as settings_ui
from hermes.plugins.model_router.admin.actions import (
    AdminActionError,
    action_descriptors,
    run_admin_action,
)
from hermes.plugins.model_router.admin.state import build_admin_state
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


def _stub_empty_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_ui,
        "scan_local_environment",
        lambda: SetupDiscovery(
            commands={"ollama": False, "hf": False},
            env_vars={"OPENAI_API_KEY": False},
            model_dirs=(),
            models=(),
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
    assert state["proxy"]["routing_mode"] == "decision"
    assert state["proxy"]["decision_layer_enabled"] is True
    assert state["proxy"]["default_backend"] is None
    assert state["proxy"]["default_model"] is None
    assert state["proxy"]["respect_client_model"] is False
    assert state["proxy"]["unknown_model_behavior"] == "fallback_to_default"
    assert state["proxy"]["safety_gate_mode"] == "decision_only"
    assert any(action["id"] == "config.save_proxy_patch" for action in state["actions"])
    fast = next(backend for backend in state["backends"] if backend["name"] == "fast")
    assert fast["api_key_configured"] is True


def test_admin_state_entrypoint_matches_settings_state(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    paths = settings_ui.settings_paths(tmp_path)

    assert build_admin_state(paths)["proxy"] == settings_ui.build_settings_state(paths)["proxy"]
    assert "proxy.start" in {action["id"] for action in action_descriptors()}


def test_settings_home_page_loads_without_chat_surface(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "ModelRouter Settings" in response.text
    assert "Provider policy" in response.text
    assert "Backend policy" in response.text
    assert "Catalog" in response.text
    assert "showCatalogDiff" in response.text
    assert "applyCatalogUpdate" in response.text
    assert 'id="pricing"' in response.text
    assert "showPricingStatus" in response.text
    assert "showPricingDiff" in response.text
    assert "applyPricingCatalog" in response.text
    assert 'href="/compact"' in response.text
    assert "Open compact windowed mode" in response.text
    assert '<section class="compact-window"' not in response.text
    assert "No routing events yet" in response.text
    assert "Settings UI Follow-Through" in response.text
    assert "Telemetry Review" in response.text
    assert "Catalog coverage" in response.text
    assert "Maturity" in response.text
    assert "TUI control center" in response.text
    assert "code_agent" in response.text
    assert 'id="feedback-outcome"' in response.text
    assert "failed_verification" in response.text
    assert response.text.count('id="settings"') == 1
    assert 'id="runtime-detail"' in response.text
    assert 'id="catalog"' in response.text
    assert "LM Studio / code" in response.text
    assert "No chat surface" in response.text
    assert "chat transcript" not in response.text.lower()
    assert "textarea id=\"chat" not in response.text.lower()


def test_compact_page_renders_standalone_control_panel(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).get("/compact")

    assert response.status_code == 200
    assert "ModelRouter Compact" in response.text
    assert 'class="compact-body"' in response.text
    assert 'aria-label="ModelRouter compact control panel"' in response.text
    assert 'href="/"' in response.text
    assert "Full" in response.text
    assert "No chat surface" in response.text
    assert '<div class="dashboard-grid">' not in response.text
    assert "overlay" not in response.text.lower()


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
                    "backend_model": "configured-code",
                    "upstream_model": "actual-code",
                    "status": "forwarded",
                    "fallback_used": True,
                    "usage_prompt_tokens": 21,
                    "usage_completion_tokens": 9,
                    "usage_total_tokens": 30,
                    "usage_cached_input_tokens": 4,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "routing-feedback.jsonl").write_text(
        json.dumps(
            {
                "event_type": "routing_feedback",
                "request_id": "req-2",
                "expected_engine": "code_agent",
                "outcome_label": "accepted",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "pricing_catalog.yaml").write_text(
        """catalog_version: 4
updated_at: "2026-06-30T00:00:00Z"
entries:
  - provider: test
    model: actual-code
    input_per_1m: 2
    output_per_1m: 4
    cached_input_per_1m: 0.5
    currency: USD
    effective_date: "2026-06-30"
    source: settings-test
""",
        encoding="utf-8",
    )
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    payload = TestClient(app).get("/api/state").json()

    assert payload["config_valid"] is True
    assert payload["provider_policy"]["available"] is True
    assert payload["provider_policy"]["version"] == 1
    assert payload["backend_policy"] == {
        "version": 1,
        "backend_allowlist": [],
        "backend_denylist": [],
    }
    assert payload["pricing_catalog"]["override_path"].endswith("pricing_catalog.yaml")
    assert payload["pricing_catalog"]["active_catalog_version"] == 4
    assert payload["pricing_catalog"]["active_catalog_source"].startswith(
        "packaged+override:"
    )
    assert payload["verifier"]["mode"] == "off"
    assert payload["discovery"]["models"][0]["repo_id"] == "mlx-community/Qwen3-4B-4bit"
    assert payload["download_plan"]["suggestions"]
    assert payload["model_library"]["installed"][0]["model_id"] == (
        "mlx-community/Qwen3-4B-4bit"
    )
    assert payload["model_library"]["registry"]["count"] >= 1
    assert any(
        model["model_id"] == "mlx-community/Qwen3-4B-4bit"
        for model in payload["model_library"]["registry"]["models"]
    )
    assert payload["model_library"]["discover"]["source"] == "curated_catalog"
    assert payload["model_library"]["discover"]["results"]
    assert payload["model_library"]["recommended"]
    assert payload["model_library"]["downloads"]
    assert any(
        assignment["route_id"] == "code_agent"
        for assignment in payload["model_library"]["assignments"]
    )
    assert payload["recommendation"]["local_model_recommendations"]
    assert "score" in payload["recommendation"]["local_model_recommendations"][0]
    assert "score" in payload["download_plan"]["suggestions"][0]
    assert payload["benchmarks"]["results"] == 0
    assert payload["maturity"]["status"] == "release_candidate"
    assert {
        feature["feature_id"]: feature["maturity"]
        for feature in payload["maturity"]["features"]
    }["tui"] == "experimental"
    assert payload["telemetry"]["backend_counts"] == {"code": 1, "fast": 1}
    assert payload["telemetry"]["fallback_count"] == 1
    assert payload["telemetry"]["feedback_labels"] == 1
    assert payload["telemetry"]["outcome_label_counts"] == {"accepted": 1}
    assert payload["telemetry"]["usage_events"] == 1
    assert payload["telemetry"]["usage_prompt_tokens"] == 21
    assert payload["telemetry"]["usage_completion_tokens"] == 9
    assert payload["telemetry"]["usage_total_tokens"] == 30
    assert payload["telemetry"]["usage_cached_input_tokens"] == 4
    assert payload["telemetry"]["usage_by_backend"]["code"]["usage_total_tokens"] == 30
    assert payload["telemetry"]["usage_by_model"]["actual-code"][
        "usage_total_tokens"
    ] == 30
    assert payload["telemetry"]["pricing_match_counts"] == {"matched": 1}
    assert payload["telemetry"]["catalog_coverage"] == {
        "active_catalog_source": "packaged+override:"
        + str(tmp_path / "pricing_catalog.yaml"),
        "active_catalog_version": 4,
        "cost_confidence": "catalog_matched",
        "rows_missing_provider_model_catalog_match": 0,
        "rows_using_placeholder_pricing": 0,
        "rows_with_catalog_match": 1,
        "rows_with_estimated_cost": 1,
        "rows_without_enough_usage_data": 1,
        "total_routing_rows": 2,
        "total_rows_with_usage": 1,
    }
    assert payload["telemetry"]["catalog_coverage_gaps"] == []
    assert payload["telemetry"]["pricing_override_skeleton"] == ""
    assert payload["telemetry"]["estimated_total_cost"] == 0.000072
    assert payload["telemetry"]["estimated_cost_currency"] == "USD"
    assert payload["telemetry"]["recent_request_ids"] == ["req-1", "req-2"]
    assert payload["route_receipt"]["request_id"] == "req-2"
    assert payload["route_receipt"]["selected"] == "code_agent"
    assert payload["provider_runtime"]["selected_backend"] == "code"
    assert payload["recent_events"][0]["usage_tokens"] == "p=21 c=9 t=30 cache=4"
    fast_backend = next(row for row in payload["backends"] if row["name"] == "fast")
    fast_adapter = fast_backend["runtime_adapter"]
    assert fast_adapter["provider"] == "lmstudio"
    assert fast_adapter["capabilities"]["load_model"]["supported"] is False
    assert "disabled_reason" in fast_adapter["capabilities"]["load_model"]
    assert any(row["route_id"] == "code_agent" for row in payload["route_map"])


def test_model_library_dashboard_renders_populated_surfaces(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)

    state = settings_ui.build_settings_state(settings_ui.settings_paths(tmp_path))
    html = settings_ui.render_dashboard_page(state)

    assert 'id="models"' in html
    assert "Installed" in html
    assert "Discover" in html
    assert "Recommended" in html
    assert "Downloads" in html
    assert "Assignments" in html
    assert 'class="model-ops-strip"' in html
    assert "compact by default" in html
    assert "expand for commands" in html
    assert '<details class="model-card">' in html
    assert "mlx-community/Qwen3-4B-4bit" in html
    assert "Qwen2.5-Coder" in html
    assert "/api/model/assign-route" in html
    assert "Plan downloads" in html
    assert "No local models found yet" not in html


def test_settings_state_feeds_runtime_models_into_registry(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_empty_scan(monkeypatch)

    def fake_runtime_state(backend, *, timeout_seconds=0.25):
        del timeout_seconds
        return {
            "adapter": "FakeRuntimeAdapter",
            "provider": "lmstudio",
            "runtime_kind": "lmstudio",
            "endpoint_url": backend.base_url,
            "detection": {
                "provider": "lmstudio",
                "runtime_kind": "lmstudio",
                "endpoint_url": backend.base_url,
                "installed": True,
                "available": True,
                "detail": "fake runtime available",
                "command": ["lms"],
            },
            "health": {
                "status": "ready",
                "reachable": True,
                "ok": True,
                "detail": "ready",
                "status_code": 200,
                "checked_url": backend.base_url.rstrip("/") + "/models",
            },
            "models": [
                {
                    "model_id": f"runtime-visible-{backend.name}",
                    "loaded": None,
                    "source": "runtime",
                }
            ],
            "loaded_models": [],
            "capabilities": {},
            "logs": {"supported": False},
        }

    monkeypatch.setattr(settings_ui, "runtime_state_for_backend", fake_runtime_state)

    state = settings_ui.build_settings_state(settings_ui.settings_paths(tmp_path))
    registry_models = state["model_library"]["registry"]["models"]

    runtime_model = next(
        model for model in registry_models if model["model_id"] == "runtime-visible-fast"
    )
    assert runtime_model["source"] == "runtime"
    assert runtime_model["provider"] == "lmstudio"
    assert runtime_model["backend"] == "fast"


def test_model_library_dashboard_renders_useful_empty_states(tmp_path, monkeypatch):
    _stub_empty_scan(monkeypatch)

    state = settings_ui.build_settings_state(settings_ui.settings_paths(tmp_path))
    html = settings_ui.render_dashboard_page(state)

    assert state["config_valid"] is False
    assert state["model_library"]["installed"] == []
    assert state["model_library"]["assignments"] == []
    assert state["model_library"]["discover"]["results"]
    assert "No local models found yet" in html
    assert "No route assignments are available" in html
    assert "Curated catalog did not return candidates" not in html


def test_dashboard_uses_latest_safe_route_receipt_without_prompt_leakage(
    tmp_path,
    monkeypatch,
):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    events_path = tmp_path / "logs" / "routing-events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "event_type": "routing_event",
                "timestamp": "2026-06-23T12:42:00.000Z",
                "request_id": "req-secret",
                "route_api": "chat_completions",
                "selected_engine": "code_agent",
                "routing_profile": "balanced",
                "status": "forwarded",
                "route_latency_ms": 0.0021,
                "upstream_latency_ms": 840.0,
                "total_latency_ms": 841.0,
                "fallback_used": False,
                "backend": "code",
                "backend_model": "qwen-code-local",
                "upstream_model": "actual-code-model",
                "usage_prompt_tokens": 40,
                "usage_completion_tokens": 12,
                "usage_total_tokens": 52,
                "usage_cached_input_tokens": 8,
                "risk_score": 50,
                "requirements": {"needs_tools": True},
                "receipt_summary": (
                    "Selected code_agent under the balanced profile; "
                    "api_key=metadata-secret; fallback available: reasoning_local."
                ),
                "reason_codes": [
                    "profile.balanced",
                    "route.coding",
                    "requirement.tools",
                    "token=metadata-secret",
                ],
                "policy_explanation": "Allowed providers: token=metadata-secret.",
                "fallback_explanation": (
                    "No fallback was used; reasoning_local remains available."
                ),
                "safety_explanation": "No human confirmation is required.",
                "privacy_explanation": "Local-only routing is active.",
                "wrong_route_next_action": "Label the request id with feedback.",
                "prompt_preview": "api_key=super-secret fix this repository",
                "prompt": "api_key=super-secret fix this repository",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = settings_ui.build_settings_state(settings_ui.settings_paths(tmp_path))
    serialized = json.dumps(state, sort_keys=True)
    html = settings_ui.render_dashboard_page(state)

    assert state["route_receipt"]["request_id"] == "req-secret"
    assert state["route_receipt"]["selected"] == "code_agent"
    assert state["route_receipt"]["backend"] == "code"
    assert state["route_receipt"]["privacy"] == "local-only"
    assert state["recent_events"][0]["request_id"] == "req-secret"
    assert state["recent_events"][0]["usage_tokens"] == "p=40 c=12 t=52 cache=8"
    assert state["review"]["items"][0]["request_id"] == "req-secret"
    assert state["review"]["items"][0]["usage_tokens"] == "p=40 c=12 t=52 cache=8"
    assert state["review"]["catalog_coverage"]["total_rows_with_usage"] == 1
    assert state["review"]["catalog_coverage"]["rows_missing_provider_model_catalog_match"] == 1
    assert state["review"]["catalog_coverage_gaps"][0]["model"] == "actual-code-model"
    assert state["review"]["catalog_coverage_gaps"][0]["backend"] == "code"
    assert state["review"]["catalog_coverage_gaps"][0]["usage_total_tokens"] == 52
    assert 'model: "actual-code-model"' in state["review"]["pricing_override_skeleton"]
    assert "input_per_1m: 0.0" in state["review"]["pricing_override_skeleton"]
    assert 'model: "actual-code-model"' in state["telemetry"]["pricing_override_skeleton"]
    assert "route.coding" in html
    assert "req-secret" in html
    assert "p=40 c=12 t=52 cache=8" in html
    assert "Catalog coverage" in html
    assert "Coverage gaps" in html
    assert "Copy override skeleton" in html
    assert "placeholder-generated-from-telemetry-gap" in html
    assert "api_key=super-secret" not in serialized
    assert "api_key=super-secret" not in html
    assert "fix this repository" not in serialized
    assert "fix this repository" not in html
    assert "metadata-secret" not in serialized
    assert "metadata-secret" not in html
    assert "[REDACTED]" in html


def test_dashboard_route_map_reflects_configured_backends(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    config_path = tmp_path / "routing_proxy.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["backends"]["code"]["base_url"] = "http://127.0.0.1:8093/v1"
    data["backends"]["code"]["model"] = "local-code.gguf"
    data["backends"]["code"]["runtime"] = {
        "enabled": True,
        "kind": "llama-server",
        "command": [
            "llama-server",
            "-m",
            "/models/local-code.gguf",
            "--port",
            "8093",
        ],
        "readiness_url": "http://127.0.0.1:8093/health",
        "idle_timeout_seconds": 900,
        "log_path": "~/.model-router/logs/code.log",
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    state = settings_ui.build_settings_state(settings_ui.settings_paths(tmp_path))
    code_row = next(row for row in state["route_map"] if row["route_id"] == "code_agent")

    assert code_row["target"].endswith("local-code.gguf")
    assert code_row["provider"] == "llama.cpp"
    assert code_row["latency"] == "On demand"
    assert state["provider_runtime"]["detail"]["builder"]["port"] == "8093"
    assert state["provider_runtime"]["detail"]["builder"]["model"] == "/models/local-code.gguf"
    assert state["provider_runtime"]["detail"]["adapter_provider"] == "llamacpp"
    assert state["provider_runtime"]["detail"]["capabilities"]["load_model"] == {
        "supported": False,
        "disabled_reason": "Managed runtimes load by starting their configured process.",
    }


def test_human_confirm_latest_event_does_not_select_backend(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    events_path = tmp_path / "logs" / "routing-events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "event_type": "routing_event",
                "request_id": "req-confirm",
                "selected_engine": "human_confirm",
                "status": "blocked",
                "fallback_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = settings_ui.build_settings_state(settings_ui.settings_paths(tmp_path))

    assert state["route_receipt"]["selected"] == "human_confirm"
    assert state["route_receipt"]["backend"] == "unassigned"
    assert state["provider_runtime"]["selected_backend"] == ""
    assert state["provider_runtime"]["detail"] == {}


def test_save_config_can_apply_preset_template_explicitly(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post(
        "/api/save-config",
        json={"confirm": True, "apply_preset": True, "preset": "ollama"},
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
            "confirm": True,
            "proxy": {
                "host": "127.0.0.1",
                "port": "9099",
                "routing_mode": "manual",
                "default_backend": "fast",
                "default_model": "new-fast-model",
                "respect_client_model": True,
                "unknown_model_behavior": "reject_404",
                "safety_gate_mode": "always_static",
            },
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
    assert config.proxy.routing_mode == "manual"
    assert config.proxy.default_backend == "fast"
    assert config.proxy.default_model == "new-fast-model"
    assert config.proxy.respect_client_model is True
    assert config.proxy.unknown_model_behavior == "reject_404"
    assert config.proxy.safety_gate_mode == "always_static"
    assert config.backends["fast"].model == "new-fast-model"
    assert config.backends["fast"].base_url == "http://127.0.0.1:9999/v1"
    assert config.observability.prompt_capture == "off"


def test_save_config_patches_routing_profile(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post(
        "/api/save-config",
        json={"confirm": True, "proxy": {"routing_profile": "private"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["proxy"]["routing_profile"] == "private"
    config = load_proxy_config(tmp_path / "routing_proxy.yaml")
    assert config.proxy.routing_profile == "private"


def test_model_assign_route_api_requires_confirmation_and_updates_config(
    tmp_path,
    monkeypatch,
):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    missing_confirm = TestClient(app).post(
        "/api/model/assign-route",
        json={"route_id": "fast_local", "model": "new-fast-model"},
    )
    assert missing_confirm.status_code == 400

    response = TestClient(app).post(
        "/api/model/assign-route",
        json={
            "confirm": True,
            "route_id": "fast_local",
            "model": "new-fast-model",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["restart_recommended"] is True
    assert payload["assignment"] == {
        "route_id": "fast_local",
        "backend": "fast",
        "model": "new-fast-model",
    }
    config = load_proxy_config(tmp_path / "routing_proxy.yaml")
    assert config.backends["fast"].model == "new-fast-model"


def test_save_config_patches_backend_policy(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post(
        "/api/save-config",
        json={
            "confirm": True,
            "backend_policy": {
                "backend_allowlist": "fast, balanced",
                "backend_denylist": ["reasoning"],
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["backend_policy"] == {
        "version": 1,
        "backend_allowlist": ["fast", "balanced"],
        "backend_denylist": ["reasoning"],
    }
    config = load_proxy_config(tmp_path / "routing_proxy.yaml")
    assert config.backend_policy.backend_allowlist == ("fast", "balanced")
    assert config.backend_policy.backend_denylist == ("reasoning",)


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

    started = client.post("/api/proxy/start", json={"confirm": True}).json()
    restarted = client.post("/api/proxy/restart", json={"confirm": True}).json()
    stopped = client.post("/api/proxy/stop", json={"confirm": True}).json()

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


def test_catalog_api_requires_confirmation_before_apply(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)
    client = TestClient(app)

    diff = client.post("/api/catalog/diff")
    blocked = client.post("/api/catalog/apply", json={})
    applied = client.post("/api/catalog/apply", json={"confirm": True})

    assert diff.status_code == 200
    assert diff.json()["ok"] is True
    assert "diff" in diff.json()
    assert blocked.status_code == 400
    assert blocked.json()["error"] == "Catalog apply requires confirm=true."
    assert applied.status_code == 200
    assert applied.json()["ok"] is True
    assert "migration_log" in applied.json()["result"]
    assert applied.json()["catalog"]["local_config"].endswith("model_router.yaml")


def test_pricing_api_requires_confirmation_before_apply(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)
    client = TestClient(app)

    status = client.post("/api/pricing/status")
    diff = client.post("/api/pricing/diff")
    blocked = client.post("/api/pricing/apply", json={})
    applied = client.post("/api/pricing/apply", json={"confirm": True})

    assert status.status_code == 200
    assert status.json()["status"]["remote_checks_enabled"] is False
    assert diff.status_code == 200
    assert diff.json()["ok"] is True
    assert "diff" in diff.json()
    assert blocked.status_code == 400
    assert blocked.json()["error"] == "Pricing catalog apply requires confirm=true."
    assert applied.status_code == 200
    assert applied.json()["ok"] is True
    assert applied.json()["status"]["override_exists"] is True
    assert (tmp_path / "pricing_catalog.yaml").exists()


def test_feedback_api_writes_cli_compatible_jsonl(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    app = settings_ui.create_settings_app(config_dir=tmp_path)

    response = TestClient(app).post(
        "/api/feedback",
        json={
            "confirm": True,
            "request_id": "req-123",
            "expected_engine": "code_agent",
            "outcome_label": "failed_verification",
            "notes": "should have used code",
        },
    )

    row = json.loads((tmp_path / "routing-feedback.jsonl").read_text(encoding="utf-8"))
    assert response.status_code == 200
    assert row["event_type"] == "routing_feedback"
    assert row["request_id"] == "req-123"
    assert row["expected_engine"] == "code_agent"
    assert row["outcome_label"] == "failed_verification"
    assert row["notes"] == "should have used code"

    invalid = TestClient(app).post(
        "/api/feedback",
        json={
            "confirm": True,
            "request_id": "req-124",
            "expected_engine": "code_agent",
            "outcome_label": "automatic_success",
        },
    )

    assert invalid.status_code == 400
    assert "invalid outcome_label" in invalid.json()["error"]


def test_mutating_admin_actions_require_confirmation(tmp_path, monkeypatch):
    _init_config(tmp_path)
    _stub_scan(monkeypatch)
    paths = settings_ui.settings_paths(tmp_path)
    app = settings_ui.create_settings_app(config_dir=tmp_path)
    client = TestClient(app)

    blocked_save = client.post(
        "/api/save-config",
        json={"proxy": {"routing_profile": "private"}},
    )
    blocked_proxy = client.post("/api/proxy/start", json={})
    blocked_feedback = client.post(
        "/api/feedback",
        json={"request_id": "req-123", "expected_engine": "code_agent"},
    )

    assert blocked_save.status_code == 400
    assert blocked_save.json()["error"] == "Config save requires confirm=true."
    assert blocked_proxy.status_code == 400
    assert blocked_proxy.json()["error"] == "Proxy start requires confirm=true."
    assert blocked_feedback.status_code == 400
    assert blocked_feedback.json()["error"] == "Feedback submission requires confirm=true."
    with pytest.raises(AdminActionError, match="Config save requires confirm=true"):
        run_admin_action(
            "config.save_proxy_patch",
            paths,
            {"proxy": {"routing_profile": "private"}},
        )
