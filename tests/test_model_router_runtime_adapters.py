from urllib.error import URLError

from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyRuntimeConfig,
)
import hermes.plugins.model_router.runtime_adapters as runtime_adapters
from hermes.plugins.model_router.runtime_adapters import (
    AdapterSupport,
    GenericOpenAICompatibleAdapter,
    LMStudioAdapter,
    ManagedRuntimeAdapter,
    OllamaAdapter,
    RuntimeCapabilities,
    RuntimeAdapter,
    adapter_for_backend,
    runtime_state_for_backend,
)


def _backend(
    *,
    name: str = "fast",
    base_url: str = "http://127.0.0.1:1234/v1",
    model: str = "fast-model",
    runtime: ProxyRuntimeConfig | None = None,
) -> ProxyBackendConfig:
    return ProxyBackendConfig(
        name=name,
        base_url=base_url,
        model=model,
        runtime=runtime or ProxyRuntimeConfig(),
    )


def _uses_protocol(adapter: RuntimeAdapter) -> bool:
    health = adapter.health(timeout_seconds=0.01)
    return health.status in {"ready", "degraded", "unsupported", "unreachable", "error"}


def test_generic_openai_adapter_health_and_model_discovery_use_fake_http():
    calls: list[tuple[str, dict[str, str], float]] = []

    def requester(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        return 200, {"data": [{"id": "fast-model"}, {"id": "other-model"}]}

    backend = _backend(base_url="http://127.0.0.1:8090/v1", model="fast-model")
    adapter = GenericOpenAICompatibleAdapter(backend, requester=requester)

    assert _uses_protocol(adapter) is True
    health = adapter.health(timeout_seconds=0.05)
    models = adapter.discover_models(timeout_seconds=0.05)
    capabilities = adapter.capabilities().to_dict()
    load = adapter.load_model("fast-model")

    assert health.ok is True
    assert health.status == "ready"
    assert [model.model_id for model in models] == ["fast-model", "other-model"]
    assert calls[0][0] == "http://127.0.0.1:8090/v1/models"
    assert capabilities["load_model"]["supported"] is False
    assert "standard load action" in capabilities["load_model"]["disabled_reason"]
    assert load.ok is False
    assert load.status == "unsupported"


def test_lmstudio_and_ollama_adapters_are_detected_from_local_ports():
    lmstudio = adapter_for_backend(_backend(base_url="http://127.0.0.1:1234/v1"))
    ollama = adapter_for_backend(_backend(base_url="http://127.0.0.1:11434/v1"))

    assert isinstance(lmstudio, LMStudioAdapter)
    assert lmstudio.capabilities().provider == "lmstudio"
    assert isinstance(ollama, OllamaAdapter)
    assert ollama.capabilities().provider == "ollama"


def test_hosted_generic_adapter_does_not_call_network_by_default():
    def requester(_url, _headers, _timeout):
        raise AssertionError("hosted health should not be checked implicitly")

    backend = _backend(base_url="https://api.openai.example/v1")
    adapter = GenericOpenAICompatibleAdapter(backend, requester=requester)

    health = adapter.health()

    assert health.status == "unsupported"
    assert adapter.discover_models() == ()


def test_managed_runtime_adapter_reports_logs_and_process_owned_actions(tmp_path):
    backend = _backend(
        base_url="http://127.0.0.1:8090/v1",
        runtime=ProxyRuntimeConfig(
            enabled=True,
            kind="llama-server",
            command=("llama-server", "-m", "/models/fast.gguf", "--port", "8090"),
            readiness_url="http://127.0.0.1:8090/v1/models",
            log_path=str(tmp_path / "fast.log"),
        ),
    )
    adapter = adapter_for_backend(backend, requester=lambda *_args: (200, {"data": []}))

    assert isinstance(adapter, ManagedRuntimeAdapter)
    capabilities = adapter.capabilities().to_dict()
    logs = adapter.logs().to_dict()

    assert capabilities["runtime_kind"] == "llama-server"
    assert capabilities["load_model"]["supported"] is False
    assert "starting their configured process" in capabilities["load_model"][
        "disabled_reason"
    ]
    assert logs["supported"] is True
    assert logs["paths"] == [str(tmp_path / "fast.log")]


def test_runtime_state_catches_adapter_failures():
    backend = _backend(base_url="http://127.0.0.1:8090/v1")

    def requester(_url, _headers, _timeout):
        raise URLError("offline")

    state = runtime_state_for_backend(backend, requester=requester)

    assert state["health"]["status"] == "unreachable"
    assert state["models"] == []
    assert state["capabilities"]["load_model"]["supported"] is False


def test_runtime_state_catches_fake_adapter_method_failures(monkeypatch):
    class BrokenAdapter:
        def capabilities(self):
            return RuntimeCapabilities(
                provider="fake",
                runtime_kind="fake",
                health=AdapterSupport(True),
                discover_models=AdapterSupport(True),
                list_loaded_models=AdapterSupport(True),
                load_model=AdapterSupport(False, "fake load disabled"),
                unload_model=AdapterSupport(False, "fake unload disabled"),
                logs=AdapterSupport(True),
            )

        def health(self, *, timeout_seconds: float = 0.25):
            raise RuntimeError("health failed")

        def discover_models(self, *, timeout_seconds: float = 0.25):
            raise RuntimeError("models failed")

        def list_loaded_models(self, *, timeout_seconds: float = 0.25):
            raise RuntimeError("loaded failed")

        def load_model(self, model_id: str):
            raise RuntimeError("load failed")

        def unload_model(self, model_id: str):
            raise RuntimeError("unload failed")

        def logs(self):
            raise RuntimeError("logs failed")

    def fake_adapter_for_backend(_backend, *, requester=None):
        del requester
        return BrokenAdapter()

    monkeypatch.setattr(
        runtime_adapters,
        "adapter_for_backend",
        fake_adapter_for_backend,
    )

    state = runtime_adapters.runtime_state_for_backend(_backend())

    assert state["adapter"] == "BrokenAdapter"
    assert state["health"]["status"] == "error"
    assert state["health"]["detail"] == "RuntimeError"
    assert state["models"] == []
    assert state["loaded_models"] == []
    assert state["logs"]["error"] == "RuntimeError"


def test_runtime_state_catches_adapter_construction_failures(monkeypatch):
    def fake_adapter_for_backend(_backend, *, requester=None):
        del requester
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr(
        runtime_adapters,
        "adapter_for_backend",
        fake_adapter_for_backend,
    )

    state = runtime_adapters.runtime_state_for_backend(_backend())

    assert state["adapter"] == "error"
    assert state["health"]["status"] == "error"
    assert state["capabilities"]["health"]["supported"] is False
    assert state["logs"]["error"] == "RuntimeError"
