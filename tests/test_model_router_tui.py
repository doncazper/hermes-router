from io import StringIO
from pathlib import Path

import pytest

from hermes.plugins.model_router import tui


def _empty_state(tmp_path: Path) -> dict:
    return {
        "product": "ModelRouter",
        "config_valid": False,
        "paths": {
            "config_dir": str(tmp_path),
            "proxy_config": str(tmp_path / "routing_proxy.yaml"),
            "model_router_config": str(tmp_path / "model_router.yaml"),
            "events": str(tmp_path / "logs" / "routing-events.jsonl"),
            "feedback": str(tmp_path / "routing-feedback.jsonl"),
            "settings_proxy_log": str(tmp_path / "logs" / "settings-proxy.log"),
        },
        "proxy": {
            "endpoint": "http://127.0.0.1:8082/v1",
            "routing_mode": "decision",
            "routing_profile": "balanced",
            "decision_layer_enabled": True,
            "respect_client_model": False,
            "safety_gate_mode": "decision_only",
        },
        "proxy_process": {"state": "stopped"},
        "observability": {
            "enabled": False,
            "log_path": str(tmp_path / "logs" / "routing-events.jsonl"),
            "prompt_capture": "redacted_preview",
        },
        "route_receipt": {},
        "model_library": {
            "installed": [],
            "recommended": [],
            "downloads": [],
            "assignments": [],
        },
        "route_map": [],
        "backends": [],
        "provider_runtime": {"providers": []},
        "telemetry": {
            "events": 0,
            "feedback_labels": 0,
            "unlabeled_replayable": 0,
            "expected_mismatch_count": 0,
            "fallback_count": 0,
        },
        "recent_events": [],
        "actions": [
            {
                "id": "doctor.run",
                "label": "Run doctor",
                "requires_confirm": False,
            },
            {
                "id": "proxy.start",
                "label": "Start proxy",
                "requires_confirm": True,
            },
        ],
    }


def _populated_state(tmp_path: Path) -> dict:
    state = _empty_state(tmp_path)
    state.update(
        {
            "config_valid": True,
            "proxy_process": {"state": "running", "pid": 1234},
            "observability": {
                **state["observability"],
                "enabled": True,
            },
            "route_receipt": {
                "request_id": "req-1",
                "selected": "code_agent",
                "backend": "code",
                "model": "qwen-code",
            },
            "model_library": {
                "installed": [
                    {
                        "display_name": "Qwen Local",
                        "model_id": "qwen-local",
                        "source": "local_directory",
                    }
                ],
                "recommended": [
                    {
                        "model_id": "mlx-community/Qwen3-4B-4bit",
                        "route_fit": ["balanced_local"],
                        "score_label": "recommended",
                    }
                ],
                "downloads": [
                    {
                        "route": "balanced_local",
                        "model_id": "mlx-community/Qwen3-4B-4bit",
                        "status": "planned",
                    }
                ],
                "assignments": [
                    {
                        "route_id": "code_agent",
                        "backend": "code",
                        "model": "qwen-code",
                    }
                ],
            },
            "route_map": [
                {
                    "route_id": "code_agent",
                    "provider": "llama.cpp",
                    "target": "Code backend",
                    "fallback": "reasoning_local",
                    "selected": True,
                }
            ],
            "backends": [
                {
                    "name": "code",
                    "runtime_adapter": {
                        "provider": "llamacpp",
                        "health": {"status": "ready"},
                        "capabilities": {
                            "load_model": {
                                "supported": False,
                                "disabled_reason": "process-owned",
                            }
                        },
                        "logs": {"paths": [str(tmp_path / "logs" / "code.log")]},
                    },
                }
            ],
            "provider_runtime": {
                "providers": [
                    {
                        "name": "llama.cpp / code",
                        "status": "ready",
                        "detail": "127.0.0.1:8090",
                    }
                ]
            },
            "telemetry": {
                "events": 2,
                "feedback_labels": 1,
                "unlabeled_replayable": 1,
                "expected_mismatch_count": 0,
                "fallback_count": 1,
                "selected_engine_counts": {"code_agent": 2},
                "backend_counts": {"code": 2},
            },
            "recent_events": [
                {
                    "request_id": "req-1",
                    "selected_engine": "code_agent",
                    "backend": "code",
                    "status": "forwarded",
                }
            ],
        }
    )
    return state


def test_tui_missing_dependency_prints_install_hint(monkeypatch, tmp_path):
    def missing_textual():
        raise tui.TuiDependencyError(tui.TEXTUAL_INSTALL_HINT)

    monkeypatch.setattr(tui, "_load_textual_symbols", missing_textual)
    output = StringIO()

    assert tui.run_tui(config_dir=tmp_path, output=output) == 1
    assert "ModelRouter TUI requires Textual" in output.getvalue()
    assert 'python -m pip install "hermes-router[tui]"' in output.getvalue()


def test_tui_snapshot_renders_empty_state(tmp_path):
    snapshot = tui.render_tui_snapshot(_empty_state(tmp_path))

    for tab in tui.TUI_TABS:
        assert f"## {tab}" in snapshot
    assert "No local models found" in snapshot
    assert "No routing map" in snapshot
    assert "No backends configured" in snapshot
    assert "Mutating shared actions require confirm=true" in snapshot
    assert "proxy.start: Start proxy (requires confirm)" in snapshot


def test_tui_snapshot_renders_populated_state(tmp_path):
    snapshot = tui.render_tui_snapshot(_populated_state(tmp_path))

    assert "Config valid: yes" in snapshot
    assert "Request: req-1" in snapshot
    assert "Qwen Local" in snapshot
    assert "* code_agent: llama.cpp -> Code backend" in snapshot
    assert "code: llamacpp ready; load=disabled" in snapshot
    assert "Events: 2" in snapshot
    assert "code.log" in snapshot
    assert "No chat surface" in snapshot


def test_build_tui_state_uses_shared_admin_state(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    expected_state = _empty_state(tmp_path)

    def fake_settings_paths(config_dir):
        captured["config_dir"] = config_dir
        return {"config_dir": Path(config_dir)}

    def fake_build_admin_state(paths, supervisor=None):
        captured["paths"] = paths
        captured["supervisor"] = supervisor
        return expected_state

    monkeypatch.setattr(tui, "settings_paths", fake_settings_paths)
    monkeypatch.setattr(tui, "build_admin_state", fake_build_admin_state)

    assert tui.build_tui_state(tmp_path) is expected_state
    assert captured["config_dir"] == tmp_path
    assert captured["paths"] == {"config_dir": tmp_path}


def test_tui_dependency_loader_reports_textual_only(monkeypatch):
    def missing_import(_name, *_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'textual'", name="textual")

    monkeypatch.setattr("builtins.__import__", missing_import)

    with pytest.raises(tui.TuiDependencyError):
        tui._load_textual_symbols()
