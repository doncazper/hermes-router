import json

from hermes.plugins.model_router.config import load_router_config
from hermes.plugins.model_router.evals import (
    EVAL_FIXTURE_SCHEMA_VERSION,
    EVAL_SCORER_VERSION,
)
from hermes.plugins.model_router.model_registry import build_model_registry
from hermes.plugins.model_router.policy import ModelRouter
from hermes.plugins.model_router.product import initialize_product_config
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyRuntimeConfig,
    ProxyServerConfig,
    RoutingProxyConfig,
    load_proxy_config,
)
from hermes.plugins.model_router.runtime_adapters import RuntimeModel
from hermes.plugins.model_router.setup_assistant import DiscoveredModel, SetupDiscovery


def _eval_row(**overrides):
    row = {
        "version": 1,
        "run_id": "evalrun_registry",
        "created_at": "2026-06-30T12:00:00.000Z",
        "backend": "fast",
        "model": "mock-model",
        "selected_engine": "fast_local",
        "fixture_id": "strict_json_routing_control_decision",
        "category": "structured_output",
        "score_percent": 100.0,
        "weighted_score": 1.0,
        "exit_status": "passed",
        "status": "completed",
        "timeout": False,
        "scorer_version": EVAL_SCORER_VERSION,
        "fixture_version": EVAL_FIXTURE_SCHEMA_VERSION,
        "failure_reasons": [],
    }
    row.update(overrides)
    return row


def _proxy_config(*backends: ProxyBackendConfig) -> RoutingProxyConfig:
    return RoutingProxyConfig(
        proxy=ProxyServerConfig(),
        router_config=None,
        backends={backend.name: backend for backend in backends},
        engine_backends={
            route: backend.name for route, backend in zip(
                ("fast_local", "balanced_local", "code_agent"),
                backends,
                strict=False,
            )
        }
        or {"fast_local": backends[0].name},
        fallback_backends={},
        source_path="test",
    )


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
    assert runtime_model["source"] == "runtime_import"
    assert runtime_model["load_state"] == "loaded"
    assert runtime_model["backend"] == "fast"
    assert runtime_model["routing_eligible"] is False


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


def test_registry_imports_lmstudio_runtime_state_idempotently():
    config = _proxy_config(
        ProxyBackendConfig(
            name="fast",
            base_url="http://127.0.0.1:1234/v1",
            model="configured-model",
        )
    )
    runtime_state = {
        "runtime_id": "lmstudio",
        "provider": "lmstudio",
        "runtime_kind": "lmstudio",
        "endpoint": "http://127.0.0.1:1234/v1",
        "detected": True,
        "last_checked_at": "2026-06-30T12:00:00Z",
        "health": {"status": "ready", "ok": True},
        "models": [
            {
                "id": "publisher/runtime-fast",
                "display_name": "Runtime Fast",
                "loaded": True,
                "context_length": 8192,
                "capabilities": {"vision": True, "tool_calls": False},
                "owned_by": "lm-studio",
            }
        ],
        "loaded_models": [],
        "capabilities": {
            "discover_models": {"supported": True},
            "list_loaded_models": {"supported": False},
        },
    }

    first = build_model_registry(
        proxy_config=config,
        runtime_models={"fast": runtime_state},
    ).to_dict()
    second = build_model_registry(
        proxy_config=config,
        runtime_models={"fast": runtime_state},
    ).to_dict()

    assert first == second
    model = next(item for item in first["models"] if item["model_id"] == "publisher/runtime-fast")
    assert model["source"] == "runtime_import"
    assert model["runtime_id"] == "lmstudio"
    assert model["name"] == "Runtime Fast"
    assert model["context_length"] == 8192
    assert model["load_state"] == "loaded"
    assert model["routing_eligible"] is True
    assert model["last_seen_at"] == "2026-06-30T12:00:00Z"
    assert "models" in model["capabilities"]
    assert "vision" in model["capabilities"]
    assert model["metadata"]["owned_by"] == "lm-studio"
    json.dumps(first)


def test_registry_imports_ollama_models_and_loaded_state():
    config = _proxy_config(
        ProxyBackendConfig(
            name="fast",
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3:4b",
        )
    )
    registry = build_model_registry(
        proxy_config=config,
        runtime_models={
            "fast": {
                "runtime_id": "ollama",
                "provider": "ollama",
                "runtime_kind": "ollama",
                "endpoint": "http://127.0.0.1:11434/v1",
                "detected": True,
                "last_checked_at": "2026-06-30T12:05:00Z",
                "health": {"status": "ready", "ok": True},
                "models": [
                    {"model_id": "qwen3:4b", "loaded": False, "source": "ollama_cli"},
                    {
                        "model_id": "llama3.2:latest",
                        "loaded": False,
                        "source": "ollama_cli",
                    },
                ],
                "loaded_models": [
                    {"model_id": "qwen3:4b", "source": "ollama_cli"},
                ],
                "capabilities": {
                    "discover_models": {"supported": True},
                    "list_loaded_models": {"supported": True},
                    "unload_model": {"supported": True},
                },
            }
        },
    )
    models = registry.to_dict()["models"]

    qwen = next(item for item in models if item["model_id"] == "qwen3:4b")
    other = next(item for item in models if item["model_id"] == "llama3.2:latest")

    assert qwen["source"] == "proxy_config+runtime_import"
    assert qwen["load_state"] == "loaded"
    assert qwen["metadata"]["runtime_source"] == "ollama_cli"
    assert qwen["runtime_id"] == "ollama"
    assert other["source"] == "runtime_import"
    assert other["load_state"] == "unloaded"
    assert other["routing_eligible"] is True
    assert "model_unload" in other["capabilities"]


def test_registry_imports_configured_llamacpp_and_mlx_model_paths():
    config = _proxy_config(
        ProxyBackendConfig(
            name="fast",
            base_url="http://127.0.0.1:8090/v1",
            model="fast-local",
            runtime=ProxyRuntimeConfig(
                enabled=True,
                kind="llama-server",
                command=("llama-server", "-m", "/models/Fast-Q4_K_M.gguf", "--port", "8090"),
                readiness_url="http://127.0.0.1:8090/v1/models",
            ),
        ),
        ProxyBackendConfig(
            name="deep",
            base_url="http://127.0.0.1:8091/v1",
            model="mlx-local",
            runtime=ProxyRuntimeConfig(
                enabled=True,
                kind="mlx-lm",
                command=("python", "-m", "mlx_lm.server", "--model", "/models/mlx/Qwen"),
                readiness_url="http://127.0.0.1:8091/v1/models",
            ),
        ),
    )

    models = build_model_registry(proxy_config=config).to_dict()["models"]
    fast = next(item for item in models if item["backend"] == "fast")
    deep = next(item for item in models if item["backend"] == "deep")

    assert fast["source"] == "proxy_config"
    assert fast["local_path"] == "/models/Fast-Q4_K_M.gguf"
    assert fast["format"] == "GGUF"
    assert fast["quantization"] == "Q4_K_M"
    assert deep["local_path"] == "/models/mlx/Qwen"
    assert deep["format"] == "MLX"


def test_registry_allows_missing_runtime_metadata_and_marks_stale_models():
    config = _proxy_config(
        ProxyBackendConfig(
            name="fast",
            base_url="http://127.0.0.1:1234/v1",
            model="configured-model",
        )
    )
    registry = build_model_registry(
        proxy_config=config,
        runtime_models={
            "fast": {
                "runtime_id": "lmstudio",
                "runtime_kind": "lmstudio",
                "health": {"status": "ready", "ok": True},
                "models": [{"id": "metadata-light"}],
                "stale_models": ["stale-runtime-model"],
            }
        },
    )

    models = registry.to_dict()["models"]
    imported = next(item for item in models if item["model_id"] == "metadata-light")
    stale = next(item for item in models if item["model_id"] == "stale-runtime-model")

    assert imported["source"] == "runtime_import"
    assert imported["routing_eligible"] is True
    assert imported["metadata"]["stale"] is False
    assert stale["install_state"] == "stale"
    assert stale["load_state"] == "stale"
    assert stale["routing_eligible"] is False
    assert stale["metadata"]["stale"] is True
    json.dumps(registry.to_dict())


def test_registry_can_attach_cached_eval_evidence_without_running_evals():
    registry = build_model_registry(
        user_models=(
            {
                "provider": "lmstudio",
                "model": "mock-model",
                "backend": "fast",
                "runtime": "lmstudio",
            },
        ),
        eval_results=(
            _eval_row(),
            _eval_row(
                fixture_id="code_review_judgment",
                category="code_review_judgment",
                score_percent=50.0,
                weighted_score=0.5,
                exit_status="failed",
                failure_reasons=("Required pattern was missing.",),
            ),
        ),
    )

    model = registry.to_dict()["models"][0]
    summary = model["metadata"]["latest_eval_summary"]

    assert summary["status"] == "evaluated"
    assert summary["fixture_count"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["score_mean_percent"] == 75.0
    assert any("advisory" in note for note in summary["notes"])
    json.dumps(model)


def test_registry_marks_missing_eval_evidence_as_not_evaluated():
    registry = build_model_registry(
        user_models=(
            {
                "provider": "lmstudio",
                "model": "unevaluated-model",
                "backend": "fast",
                "runtime": "lmstudio",
            },
        ),
        eval_results=(),
    )

    summary = registry.to_dict()["models"][0]["metadata"]["latest_eval_summary"]

    assert summary["status"] == "not_evaluated"
    assert summary["fixture_count"] == 0
    assert any("does not block routing" in note for note in summary["notes"])


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


def test_route_fast_does_not_load_eval_evidence(monkeypatch, tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    router = ModelRouter.from_config(tmp_path / "model_router.yaml")

    import hermes.plugins.model_router.eval_runner as eval_runner

    def fail_eval(*_args, **_kwargs):
        raise AssertionError("route_fast must not load or summarize eval evidence")

    monkeypatch.setattr(eval_runner, "load_eval_results", fail_eval)
    monkeypatch.setattr(eval_runner, "eval_evidence_for_model", fail_eval)
    monkeypatch.setattr(eval_runner, "eval_evidence_from_rows", fail_eval)

    assert router.route_fast("rewrite this text") == "fast_local"
