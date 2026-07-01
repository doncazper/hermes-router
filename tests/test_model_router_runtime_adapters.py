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
    RuntimeActionResult,
    RuntimeCapabilities,
    RuntimeAdapter,
    RuntimeDetection,
    RuntimeHealth,
    RuntimeLogInfo,
    RuntimeModel,
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
    assert adapter.endpoint_url() == "http://127.0.0.1:8090/v1"
    assert adapter.detect().available is True
    assert capabilities["endpoint_url"] == "http://127.0.0.1:8090/v1"
    assert capabilities["detect_runtime"]["supported"] is True
    assert capabilities["load_model"]["supported"] is False
    assert "standard load action" in capabilities["load_model"]["disabled_reason"]
    assert capabilities["start_server"]["supported"] is False
    assert "outside ModelRouter" in capabilities["start_server"]["disabled_reason"]
    assert load.ok is False
    assert load.status == "unsupported"


def test_runtime_state_supports_mocked_full_support_adapter(monkeypatch):
    captured: dict[str, object] = {}

    class FullSupportAdapter:
        def __init__(self, backend):
            self.backend = backend
            self.started = False
            self.loaded: list[str] = []

        def endpoint_url(self):
            return "http://127.0.0.1:7777/v1"

        def capabilities(self):
            support = AdapterSupport(True)
            return RuntimeCapabilities(
                provider="mock",
                runtime_kind="mock-runtime",
                endpoint_url=self.endpoint_url(),
                detect_runtime=support,
                health=support,
                discover_models=support,
                list_loaded_models=support,
                start_server=support,
                stop_server=support,
                load_model=support,
                unload_model=support,
                logs=support,
            )

        def detect(self):
            return RuntimeDetection(
                provider="mock",
                runtime_kind="mock-runtime",
                endpoint_url=self.endpoint_url(),
                installed=True,
                available=True,
                detail="mock runtime available",
                command=("mock-runtime",),
            )

        def health(self, *, timeout_seconds: float = 0.25):
            return RuntimeHealth(
                status="ready",
                reachable=True,
                ok=True,
                detail=f"ready within {timeout_seconds}",
                checked_url=self.endpoint_url() + "/models",
            )

        def discover_models(self, *, timeout_seconds: float = 0.25):
            del timeout_seconds
            return (
                RuntimeModel("mock-small", loaded=True),
                RuntimeModel("mock-large", loaded=False),
            )

        def list_loaded_models(self, *, timeout_seconds: float = 0.25):
            del timeout_seconds
            return (RuntimeModel("mock-small", loaded=True),)

        def start_server(self):
            self.started = True
            return RuntimeActionResult(True, "started", "mock server started")

        def stop_server(self):
            self.started = False
            return RuntimeActionResult(True, "stopped", "mock server stopped")

        def load_model(self, model_id: str):
            self.loaded.append(model_id)
            return RuntimeActionResult(True, "loaded", f"loaded {model_id}")

        def unload_model(self, model_id: str):
            return RuntimeActionResult(True, "unloaded", f"unloaded {model_id}")

        def logs(self):
            return RuntimeLogInfo(supported=True, paths=("/tmp/mock.log",))

    def fake_adapter_for_backend(backend, *, requester=None):
        del requester
        adapter = FullSupportAdapter(backend)
        captured["adapter"] = adapter
        return adapter

    monkeypatch.setattr(
        runtime_adapters,
        "adapter_for_backend",
        fake_adapter_for_backend,
    )

    state = runtime_adapters.runtime_state_for_backend(_backend())
    adapter = captured["adapter"]

    assert state["endpoint_url"] == "http://127.0.0.1:7777/v1"
    assert state["detection"]["installed"] is True
    assert state["detection"]["available"] is True
    assert state["health"]["status"] == "ready"
    assert [model["model_id"] for model in state["models"]] == [
        "mock-small",
        "mock-large",
    ]
    assert state["loaded_models"] == [
        {"model_id": "mock-small", "loaded": True, "source": "runtime"}
    ]
    assert state["capabilities"]["start_server"]["supported"] is True
    assert adapter.start_server().ok is True
    assert adapter.load_model("mock-large").status == "loaded"
    assert adapter.stop_server().status == "stopped"


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
    capabilities = adapter.capabilities().to_dict()
    start = adapter.start_server()

    assert health.status == "unsupported"
    assert adapter.detect().available is None
    assert adapter.discover_models() == ()
    assert capabilities["health"]["supported"] is False
    assert capabilities["discover_models"]["supported"] is False
    assert capabilities["start_server"]["supported"] is False
    assert "provider" in capabilities["start_server"]["disabled_reason"]
    assert start.status == "unsupported"
    assert start.disabled_reason == capabilities["start_server"]["disabled_reason"]


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
    assert capabilities["start_server"]["supported"] is False
    assert "model-router-proxy" in capabilities["start_server"]["disabled_reason"]
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
