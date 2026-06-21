import json

import yaml

from hermes.plugins.model_router.config import load_router_config
import hermes.plugins.model_router.product as product
from hermes.plugins.model_router.product import (
    PRESETS,
    BackendHealth,
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
