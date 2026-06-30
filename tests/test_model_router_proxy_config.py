from pathlib import Path

import pytest
import yaml

from hermes.plugins.model_router.proxy_config import (
    ProxyConfigError,
    RUNTIME_KINDS,
    default_proxy_config_source,
    load_proxy_config,
)


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "routing_proxy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _valid_config() -> dict:
    return {
        "proxy": {
            "host": "127.0.0.1",
            "port": 8082,
            "api_key_env": "MODEL_ROUTER_PROXY_API_KEY",
            "model_ids": ["model-router"],
        },
        "router_config": "configs/model_router.yaml",
        "backends": {
            "fast": {
                "base_url": "http://fast.test/v1",
                "model": "fast-model",
                "strip_tools": True,
            },
            "deep": {
                "base_url": "http://deep.test/v1",
                "model": "deep-model",
                "api_key_env": "DEEP_API_KEY",
                "timeout_seconds": 20,
            },
        },
        "engine_backends": {
            "fast_local": "fast",
            "balanced_local": "fast",
            "reasoning_local": "deep",
            "code_agent": "deep",
        },
        "fallback_backends": {
            "fast": ["deep"],
        },
    }


def test_load_proxy_config_accepts_valid_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    config = load_proxy_config(_write_config(tmp_path, _valid_config()))

    assert config.proxy.host == "127.0.0.1"
    assert config.proxy.port == 8082
    assert config.proxy.routing_profile == "balanced"
    assert config.proxy.routing_mode == "decision"
    assert config.proxy.default_backend is None
    assert config.proxy.default_model is None
    assert config.proxy.respect_client_model is False
    assert config.proxy.unknown_model_behavior == "fallback_to_default"
    assert config.proxy.safety_gate_mode == "decision_only"
    assert config.proxy.resolved_api_key == "proxy-secret"
    assert config.backends["fast"].strip_tools is True
    assert config.backends["deep"].resolved_api_key == "deep-secret"
    assert config.observability.enabled is False
    assert config.backend_for_engine("fast_local") == config.backends["fast"]
    assert config.fallback_chain_for_backend("fast") == (config.backends["deep"],)
    assert config.backend_policy.to_dict() == {
        "version": 1,
        "backend_allowlist": [],
        "backend_denylist": [],
    }
    assert config.verifier.mode == "off"


def test_load_proxy_config_accepts_routing_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["proxy"]["routing_profile"] = "private"

    config = load_proxy_config(_write_config(tmp_path, data))

    assert config.proxy.routing_profile == "private"


def test_load_proxy_config_accepts_manual_routing_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["proxy"].update(
        {
            "routing_mode": "manual",
            "default_backend": "deep",
            "default_model": "manual-model",
            "respect_client_model": True,
            "unknown_model_behavior": "reject_404",
            "safety_gate_mode": "always_static",
        }
    )

    config = load_proxy_config(_write_config(tmp_path, data))

    assert config.proxy.routing_mode == "manual"
    assert config.proxy.default_backend == "deep"
    assert config.proxy.default_model == "manual-model"
    assert config.proxy.respect_client_model is True
    assert config.proxy.unknown_model_behavior == "reject_404"
    assert config.proxy.safety_gate_mode == "always_static"


def test_load_proxy_config_decision_mode_ignores_unused_manual_default_backend(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["proxy"]["default_backend"] = "missing"

    config = load_proxy_config(_write_config(tmp_path, data))

    assert config.proxy.routing_mode == "decision"
    assert config.proxy.default_backend == "missing"


def test_load_proxy_config_rejects_incomplete_manual_routing_mode(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["proxy"]["routing_mode"] = "manual"

    with pytest.raises(ProxyConfigError, match="default_backend"):
        load_proxy_config(_write_config(tmp_path, data))

    data["proxy"]["default_backend"] = "missing"
    data["proxy"]["default_model"] = "manual-model"
    with pytest.raises(ProxyConfigError, match="undefined backend"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_deferred_routing_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["proxy"]["routing_mode"] = "model_map"

    with pytest.raises(ProxyConfigError, match="not implemented"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_accepts_backend_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["backend_policy"] = {
        "version": 1,
        "backend_allowlist": ["fast"],
        "backend_denylist": ["deep"],
    }

    config = load_proxy_config(_write_config(tmp_path, data))

    assert config.backend_policy.backend_allowlist == ("fast",)
    assert config.backend_policy.backend_denylist == ("deep",)
    assert config.backend_for_engine("fast_local") == config.backends["fast"]
    assert config.backend_for_engine("reasoning_local") is None
    assert config.fallback_chain_for_backend("fast") == ()


def test_load_proxy_config_accepts_verifier_block(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["verifier"] = {
        "version": 1,
        "mode": "sampled",
        "backend": "deep",
        "sample_rate": 0.25,
        "route_codes": ["route.coding"],
        "timeout_seconds": 3,
        "failure_behavior": "fail_closed",
        "prompt_template": "Verify {selected_engine}: {receipt_summary}",
        "include_response_preview": True,
        "max_response_preview_chars": 120,
    }

    config = load_proxy_config(_write_config(tmp_path, data))

    assert config.verifier.mode == "sampled"
    assert config.verifier.backend == "deep"
    assert config.verifier.sample_rate == 0.25
    assert config.verifier.route_codes == ("route.coding",)
    assert config.verifier.failure_behavior == "fail_closed"
    assert config.verifier.include_response_preview is True


def test_load_proxy_config_rejects_invalid_verifier_block(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["verifier"] = {"mode": "sampled", "backend": "missing", "sample_rate": 1.0}

    with pytest.raises(ProxyConfigError, match="undefined backend"):
        load_proxy_config(_write_config(tmp_path, data))

    data["verifier"] = {"mode": "sampled", "backend": "deep", "sample_rate": 0.0}
    with pytest.raises(ProxyConfigError, match="sample_rate"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_invalid_backend_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["backend_policy"] = {"backend_allowlist": ["missing"]}

    with pytest.raises(ProxyConfigError, match="undefined backend"):
        load_proxy_config(_write_config(tmp_path, data))

    data["backend_policy"] = {"version": 2}
    with pytest.raises(ProxyConfigError, match="version"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_invalid_routing_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["proxy"]["routing_profile"] = "mystery"

    with pytest.raises(ProxyConfigError, match="routing profile"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_uses_packaged_example_outside_repo_cwd(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    config = load_proxy_config()

    assert config.source_path == default_proxy_config_source()
    assert config.proxy.host == "127.0.0.1"
    assert config.proxy.port == 8082
    assert config.backend_for_engine("fast_local") == config.backends["fast"]


def test_load_proxy_config_rejects_undefined_engine_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["engine_backends"]["fast_local"] = "missing"

    with pytest.raises(ProxyConfigError, match="undefined backend"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_fallback_cycles(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["fallback_backends"] = {"fast": ["deep"], "deep": ["fast"]}

    with pytest.raises(ProxyConfigError, match="cycle"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_ambiguous_auth_settings(tmp_path):
    data = _valid_config()
    data["proxy"]["api_key"] = "literal"

    with pytest.raises(ProxyConfigError, match="api_key or api_key_env"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_missing_proxy_api_key_env(tmp_path):
    data = _valid_config()

    with pytest.raises(ProxyConfigError, match="MODEL_ROUTER_PROXY_API_KEY"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_missing_backend_api_key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    data = _valid_config()

    with pytest.raises(ProxyConfigError, match="DEEP_API_KEY"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_accepts_observability_block(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["observability"] = {
        "enabled": True,
        "log_path": "~/.model-router/test.jsonl",
        "prompt_capture": "redacted_preview",
        "max_bytes": 1024,
        "backups": 2,
    }
    data["health"] = {"backend_timeout_seconds": 2.5}

    config = load_proxy_config(_write_config(tmp_path, data))

    assert config.observability.enabled is True
    assert config.observability.log_path == "~/.model-router/test.jsonl"
    assert config.observability.prompt_capture == "redacted_preview"
    assert config.observability.max_bytes == 1024
    assert config.observability.backups == 2
    assert config.health.backend_timeout_seconds == 2.5


def test_load_proxy_config_accepts_backend_runtime_block(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["backends"]["fast"]["runtime"] = {
        "enabled": True,
        "kind": "llama-server",
        "command": [
            "llama-server",
            "-m",
            "/models/fast.gguf",
            "--alias",
            "fast",
            "--alias",
            "fast",
            "--port",
            "8090",
        ],
        "readiness_url": "http://127.0.0.1:8090/v1/models",
        "readiness_timeout_seconds": 10,
        "idle_timeout_seconds": 900,
        "shutdown_timeout_seconds": 5,
        "log_path": "~/.model-router/logs/llama-fast.log",
    }

    config = load_proxy_config(_write_config(tmp_path, data))
    runtime = config.backends["fast"].runtime

    assert runtime.enabled is True
    assert runtime.kind == "llama-server"
    assert runtime.command == (
        "llama-server",
        "-m",
        "/models/fast.gguf",
        "--alias",
        "fast",
        "--alias",
        "fast",
        "--port",
        "8090",
    )
    assert runtime.readiness_url == "http://127.0.0.1:8090/v1/models"
    assert runtime.idle_timeout_seconds == 900
    assert "llama-server" in RUNTIME_KINDS


def test_load_proxy_config_rejects_runtime_shell_string_command(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["backends"]["fast"]["runtime"] = {
        "enabled": True,
        "kind": "llama-server",
        "command": "llama-server -m /models/fast.gguf --port 8090",
        "readiness_url": "http://127.0.0.1:8090/v1/models",
        "log_path": "~/.model-router/logs/llama-fast.log",
    }

    with pytest.raises(ProxyConfigError, match="command"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_invalid_runtime_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["backends"]["fast"]["runtime"] = {
        "enabled": True,
        "kind": "made-up",
        "command": ["runtime"],
        "readiness_url": "http://127.0.0.1:8090/v1/models",
        "readiness_timeout_seconds": 0,
        "log_path": "~/.model-router/logs/runtime.log",
    }

    with pytest.raises(ProxyConfigError, match="runtime kind"):
        load_proxy_config(_write_config(tmp_path, data))

    data["backends"]["fast"]["runtime"]["kind"] = "generic"
    with pytest.raises(ProxyConfigError, match="readiness_timeout_seconds"):
        load_proxy_config(_write_config(tmp_path, data))

    data["backends"]["fast"]["runtime"]["readiness_timeout_seconds"] = 1
    data["backends"]["fast"]["runtime"]["readiness_url"] = "127.0.0.1:8090/v1/models"
    with pytest.raises(ProxyConfigError, match="readiness_url"):
        load_proxy_config(_write_config(tmp_path, data))

    data["backends"]["fast"]["runtime"]["readiness_url"] = (
        "http://127.0.0.1:8090/v1/models"
    )
    data["backends"]["fast"]["runtime"]["log_path"] = "bad\x00path.log"
    with pytest.raises(ProxyConfigError, match="log_path"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_invalid_prompt_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["observability"] = {"prompt_capture": "rawish"}

    with pytest.raises(ProxyConfigError, match="prompt_capture"):
        load_proxy_config(_write_config(tmp_path, data))


def test_load_proxy_config_rejects_invalid_observability_rotation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MODEL_ROUTER_PROXY_API_KEY", "proxy-secret")
    monkeypatch.setenv("DEEP_API_KEY", "deep-secret")
    data = _valid_config()
    data["observability"] = {"max_bytes": -1}

    with pytest.raises(ProxyConfigError, match="max_bytes"):
        load_proxy_config(_write_config(tmp_path, data))
