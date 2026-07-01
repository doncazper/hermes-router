import json

from hermes.plugins.model_router.config import load_router_config
from hermes.plugins.model_router.model_registry import build_model_registry
from hermes.plugins.model_router.policy import ModelRouter
from hermes.plugins.model_router.product import initialize_product_config
from hermes.plugins.model_router.proxy_config import load_proxy_config
from hermes.plugins.model_router.runtime_adapters import RuntimeModel
from hermes.plugins.model_router.setup_assistant import DiscoveredModel, SetupDiscovery


def test_registry_tracks_local_model_with_json_safe_metadata(tmp_path):
    model_file = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    model_file.write_bytes(b"placeholder")
    discovery = SetupDiscovery(
        commands={},
        model_dirs=(str(tmp_path),),
        models=(
            DiscoveredModel(
                name="Qwen3 4B",
                repo_id="Qwen/Qwen3-4B-GGUF",
                path=str(model_file),
                source="local_directory",
                roles=("balanced_local", "code_agent"),
            ),
        ),
    )

    registry = build_model_registry(discovery=discovery)
    payload = registry.to_dict()

    assert payload["count"] == 1
    model = payload["models"][0]
    assert model["model_id"] == "Qwen/Qwen3-4B-GGUF"
    assert model["provider"] == "local"
    assert model["runtime"] == "llama.cpp"
    assert model["local_path"] == str(model_file)
    assert model["format"] == "GGUF"
    assert model["quantization"] == "Q4_K_M"
    assert model["size_bytes"] == len(b"placeholder")
    assert model["install_state"] == "installed"
    assert model["assigned_routes"] == ["balanced_local", "code_agent"]
    assert "code" in model["capabilities"]
    json.dumps(payload)


def test_registry_tracks_user_declared_hosted_model_with_missing_metadata():
    registry = build_model_registry(
        user_models=(
            {
                "provider": "openai",
                "model": "gpt-example",
                "capabilities": ("chat", "tools"),
                "routing_eligible": False,
                "custom_note": object(),
            },
        )
    )

    model = registry.to_dict()["models"][0]

    assert model["provider"] == "openai"
    assert model["runtime"] == "api"
    assert model["model_id"] == "gpt-example"
    assert model["local_path"] is None
    assert model["format"] == "API"
    assert model["context_length"] is None
    assert model["license"] is None
    assert model["install_state"] == "remote"
    assert model["routing_eligible"] is False
    assert model["metadata"]["custom_note"].startswith("<object object")
    json.dumps(model)


def test_registry_imports_models_from_router_and_proxy_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    router_config = load_router_config(tmp_path / "model_router.yaml")
    proxy_config = load_proxy_config(tmp_path / "routing_proxy.yaml")

    registry = build_model_registry(
        router_config=router_config,
        proxy_config=proxy_config,
        runtime_models={"fast": (RuntimeModel("runtime-visible", loaded=True),)},
    )
    models = registry.to_dict()["models"]

    router_model = next(
        model
        for model in models
        if model["source"] == "router_config" and model["assigned_routes"]
    )
    proxy_model = next(
        model
        for model in models
        if model["source"] == "proxy_config" and model["backend"] == "fast"
    )
    runtime_model = next(model for model in models if model["model_id"] == "runtime-visible")

    assert router_model["provider"]
    assert router_model["context_length"] is not None
    assert router_model["assigned_routes"]
    assert proxy_model["provider"] == "lmstudio"
    assert proxy_model["runtime"] == "lmstudio"
    assert proxy_model["backend"] == "fast"
    assert proxy_model["routing_eligible"] is True
    assert runtime_model["source"] == "runtime"
    assert runtime_model["load_state"] == "loaded"
    assert runtime_model["backend"] == "fast"


def test_registry_detects_lmstudio_imported_model_folder():
    discovery = SetupDiscovery(
        commands={},
        model_dirs=("/Users/example/.lmstudio/models",),
        models=(
            DiscoveredModel(
                name="Local LM Studio model",
                repo_id="publisher/model",
                path="/Users/example/.lmstudio/models/publisher/model",
                source="lmstudio",
                roles=("balanced_local",),
            ),
        ),
    )

    model = build_model_registry(discovery=discovery).to_dict()["models"][0]

    assert model["provider"] == "lmstudio"
    assert model["runtime"] == "lmstudio"
    assert model["source"] == "lmstudio"
    assert model["install_state"] == "installed"


def test_route_fast_does_not_depend_on_model_registry(monkeypatch, tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    router = ModelRouter.from_config(tmp_path / "model_router.yaml")

    import hermes.plugins.model_router.model_registry as model_registry

    def fail_registry(*_args, **_kwargs):
        raise AssertionError("route_fast must not build the model registry")

    monkeypatch.setattr(model_registry, "build_model_registry", fail_registry)

    assert router.route_fast("rewrite this text") == "fast_local"
