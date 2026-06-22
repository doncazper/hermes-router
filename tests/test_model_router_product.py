import json

import yaml

from hermes.plugins.model_router.config import load_router_config
import hermes.plugins.model_router.product as product
from hermes.plugins.model_router.product import (
    PRESETS,
    BackendHealth,
    FirstRunSignals,
    OLLAMA_RECOMMENDED_MODELS,
    doctor_proxy_config,
    initialize_product_config,
)
from hermes.plugins.model_router.proxy_config import load_proxy_config


def test_init_noninteractive_lmstudio_writes_runnable_configs(tmp_path):
    result = initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        proxy_port=9090,
        force=False,
        interactive=False,
    )

    assert result.ok is True
    assert (tmp_path / "model_router.yaml").is_file()
    assert (tmp_path / "routing_proxy.yaml").is_file()
    assert (tmp_path / "logs").is_dir()

    router_config = load_router_config(tmp_path / "model_router.yaml")
    proxy_config = load_proxy_config(tmp_path / "routing_proxy.yaml")

    assert router_config.get_engine("fast_local") is not None
    assert proxy_config.proxy.port == 9090
    assert proxy_config.router_config == str(tmp_path / "model_router.yaml")
    assert proxy_config.observability.enabled is True
    assert proxy_config.health.backend_timeout_seconds == 1.0


def test_init_auto_selects_ollama_and_reports_missing_model_pulls(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        product,
        "detect_first_run_environment",
        lambda: FirstRunSignals(
            ollama_installed=True,
            ollama_running=True,
            lmstudio_running=False,
            ollama_models=("qwen3:0.6b",),
            recommended_preset="ollama",
            notes=("Recommended preset: ollama.", "Ollama is reachable."),
        ),
    )

    result = initialize_product_config(
        auto_detect=True,
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    config = load_proxy_config(tmp_path / "routing_proxy.yaml")

    assert result.ok is True
    assert result.preset == "ollama"
    assert result.detection["ollama_running"] is True
    assert config.backends["fast"].base_url == "http://127.0.0.1:11434/v1"
    missing = set(OLLAMA_RECOMMENDED_MODELS) - {"qwen3:0.6b"}
    for model in missing:
        assert f"- ollama pull {model}" in result.messages


def test_init_auto_reports_ollama_start_when_installed_but_stopped(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        product,
        "detect_first_run_environment",
        lambda: FirstRunSignals(
            ollama_installed=True,
            ollama_running=False,
            lmstudio_running=False,
            recommended_preset="ollama",
            notes=("Recommended preset: ollama.",),
        ),
    )

    result = initialize_product_config(
        auto_detect=True,
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    assert result.preset == "ollama"
    assert "Start Ollama before running the proxy: ollama serve" in result.messages
    assert "- ollama pull qwen3:0.6b" in result.messages


def test_init_rejects_preset_and_auto_together(tmp_path):
    try:
        initialize_product_config(
            preset="lmstudio",
            auto_detect=True,
            config_dir=tmp_path,
            interactive=False,
        )
    except ValueError as exc:
        assert "either --preset or --auto" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_init_does_not_overwrite_without_force(tmp_path):
    initialize_product_config(
        preset="ollama",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    result = initialize_product_config(
        preset="ollama",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    assert result.ok is False
    assert str(tmp_path / "model_router.yaml") in result.skipped
    assert str(tmp_path / "routing_proxy.yaml") in result.skipped


def test_all_presets_generate_valid_proxy_configs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for preset in PRESETS:
        config_dir = tmp_path / preset
        initialize_product_config(
            preset=preset,
            config_dir=config_dir,
            force=False,
            interactive=False,
        )

        load_router_config(config_dir / "model_router.yaml")
        config = load_proxy_config(config_dir / "routing_proxy.yaml")
        assert config.backend_for_engine("fast_local") is not None


def test_init_interactive_customizes_backend_values(tmp_path):
    answers = iter(
        [
            "1",
            "9091",
            "http://fast.local/v1",
            "fast-custom",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    )

    result = initialize_product_config(
        config_dir=tmp_path,
        interactive=True,
        input_func=lambda _prompt: next(answers),
    )
    data = yaml.safe_load((tmp_path / "routing_proxy.yaml").read_text())

    assert result.ok is True
    assert data["proxy"]["port"] == 9091
    assert data["backends"]["fast"]["base_url"] == "http://fast.local/v1"
    assert data["backends"]["fast"]["model"] == "fast-custom"


def test_doctor_reports_backend_health(tmp_path, monkeypatch):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    def fake_health(backend, *, timeout_seconds):
        return BackendHealth(
            backend=backend.name,
            reachable=backend.name != "reasoning",
            ok=backend.name != "reasoning",
            status_code=200 if backend.name != "reasoning" else None,
            detail="mocked",
        )

    monkeypatch.setattr(product, "check_backend_health", fake_health)
    report = doctor_proxy_config(tmp_path / "routing_proxy.yaml")

    assert report.proxy_config_valid is True
    assert report.router_config_valid is True
    assert report.ok is False
    assert any(not backend.reachable for backend in report.backends)
    assert json.dumps(report.to_dict())


def test_doctor_includes_first_run_remediation_for_ollama(tmp_path, monkeypatch):
    initialize_product_config(
        preset="ollama",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    def fake_health(backend, *, timeout_seconds):
        if backend.name == "fast":
            return BackendHealth(
                backend=backend.name,
                reachable=True,
                ok=False,
                status_code=200,
                detail=f"reachable: HTTP 200; configured model {backend.model!r} not listed",
            )
        return BackendHealth(
            backend=backend.name,
            reachable=False,
            ok=False,
            status_code=None,
            detail="connection refused",
        )

    monkeypatch.setattr(product, "check_backend_health", fake_health)
    report = doctor_proxy_config(tmp_path / "routing_proxy.yaml")

    assert report.ok is False
    assert report.proxy_endpoint == "http://127.0.0.1:8082/v1"
    assert report.telemetry_log_path == str(tmp_path / "logs" / "routing-events.jsonl")
    assert "Ollama backend unreachable; start Ollama with `ollama serve`." in (
        report.remediation
    )
    assert "Backend fast model missing; run `ollama pull qwen3:0.6b`." in (
        report.remediation
    )
    assert any("model-router telemetry summary" in item for item in report.remediation)


def test_check_backend_health_reports_listed_model(monkeypatch):
    backend = product.ProxyBackendConfig(
        name="fast",
        base_url="http://backend.test/v1",
        model="fast-model",
    )

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"data":[{"id":"fast-model"}]}'

    monkeypatch.setattr(product, "urlopen", lambda *_args, **_kwargs: Response())

    health = product.check_backend_health(backend, timeout_seconds=1.0)

    assert health.ok is True
    assert "configured model 'fast-model' listed" in health.detail


def test_check_backend_health_reports_missing_model(monkeypatch):
    backend = product.ProxyBackendConfig(
        name="fast",
        base_url="http://backend.test/v1",
        model="missing-model",
    )

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"data":[{"id":"other-model"}]}'

    monkeypatch.setattr(product, "urlopen", lambda *_args, **_kwargs: Response())

    health = product.check_backend_health(backend, timeout_seconds=1.0)

    assert health.reachable is True
    assert health.ok is False
    assert "configured model 'missing-model' not listed" in health.detail
