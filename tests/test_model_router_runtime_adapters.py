import json
import sys
from urllib.error import URLError

from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyRuntimeConfig,
)
import hermes.plugins.model_router.runtime_adapters as runtime_adapters
from hermes.plugins.model_router.runtime_adapters import (
    AdapterSupport,
    CommandResult,
    GenericOpenAICompatibleAdapter,
    LMStudioAdapter,
    LocalAIAdapter,
    ManagedRuntimeAdapter,
    OllamaAdapter,
    RuntimeActionResult,
    RuntimeCapabilities,
    RuntimeAdapter,
    RuntimeDetection,
    RuntimeHealth,
    RuntimeLogInfo,
    RuntimeModel,
    VLLMAdapter,
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


def test_runtime_detection_report_contains_mvp_fields():
    def requester(_url, _headers, _timeout):
        return 200, {"data": [{"id": "fast-model"}]}

    state = runtime_state_for_backend(
        _backend(base_url="http://127.0.0.1:8090/v1", model="fast-model"),
        requester=requester,
        checked_at="2026-06-30T12:00:00Z",
    )

    assert state["runtime_id"] == "openai_compatible_local"
    assert state["runtime_kind"] == "openai_compatible_local"
    assert state["runtime_mode"] == "external_managed"
    assert state["detected"] is True
    assert state["endpoint"] == "http://127.0.0.1:8090/v1"
    assert state["version"] is None
    assert state["health_status"] == "ready"
    assert state["missing_dependency"] is None
    assert state["install_hint"] is None
    assert state["last_checked_at"] == "2026-06-30T12:00:00Z"
    assert state["detection"]["runtime_id"] == "openai_compatible_local"
    assert state["detection"]["detected"] is True


def test_runtime_detection_reports_missing_ollama_cli_and_unreachable_server(
    monkeypatch,
):
    def requester(_url, _headers, _timeout):
        raise URLError("offline")

    monkeypatch.setattr(runtime_adapters, "_resolve_command", lambda _command: None)

    state = runtime_state_for_backend(
        _backend(base_url="http://127.0.0.1:11434/v1", model="qwen3:4b"),
        requester=requester,
        checked_at="2026-06-30T12:00:00Z",
    )

    assert state["runtime_id"] == "ollama"
    assert state["detected"] is True
    assert state["health"]["status"] == "unreachable"
    assert state["missing_dependency"] == "ollama CLI"
    assert "Start Ollama" in state["install_hint"]


def test_runtime_detection_reports_unhealthy_but_reachable_runtime():
    def requester(_url, _headers, _timeout):
        return 500, {"data": [{"id": "other-model"}]}

    state = runtime_state_for_backend(
        _backend(base_url="http://127.0.0.1:8090/v1", model="fast-model"),
        requester=requester,
        checked_at="2026-06-30T12:00:00Z",
    )

    assert state["detected"] is True
    assert state["health"]["status"] == "degraded"
    assert state["health"]["reachable"] is True
    assert state["health"]["ok"] is False
    assert state["health"]["status_code"] == 500


def test_runtime_detection_reports_configured_but_unreachable_runtime():
    def requester(_url, _headers, _timeout):
        raise URLError("offline")

    state = runtime_state_for_backend(
        _backend(base_url="http://127.0.0.1:8090/v1", model="fast-model"),
        requester=requester,
        checked_at="2026-06-30T12:00:00Z",
    )

    assert state["detected"] is True
    assert state["health"]["status"] == "unreachable"
    assert state["install_hint"] == (
        "Start the configured local OpenAI-compatible server or update backend base_url."
    )


def test_runtime_detection_reports_hosted_backend_without_network_call():
    def requester(_url, _headers, _timeout):
        raise AssertionError("hosted detection should not call the network")

    state = runtime_state_for_backend(
        _backend(base_url="https://api.openai.example/v1", model="hosted-model"),
        requester=requester,
        checked_at="2026-06-30T12:00:00Z",
    )

    assert state["runtime_id"] == "openai_compatible_hosted"
    assert state["runtime_mode"] == "external_managed"
    assert state["detected"] is True
    assert state["health"]["status"] == "unsupported"
    assert "does not probe provider availability" in state["install_hint"]


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
    assert state["runtime_id"] == "mock"
    assert state["runtime_mode"] == "external_managed"
    assert state["detected"] is True
    assert state["last_checked_at"]
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


def test_localai_adapter_detected_from_template_model_hint():
    calls: list[tuple[str, dict[str, str], float]] = []

    def requester(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        return 200, {"data": [{"id": "localai-fast-model"}]}

    backend = _backend(
        name="fast",
        base_url="http://127.0.0.1:8080/v1",
        model="localai-fast-model",
    )
    adapter = adapter_for_backend(backend, requester=requester)

    assert isinstance(adapter, LocalAIAdapter)
    assert adapter.detect().provider == "localai"
    assert adapter.capabilities().provider == "localai"
    assert adapter.capabilities().runtime_kind == "localai"
    assert adapter.health(timeout_seconds=0.05).ok is True
    assert [model.model_id for model in adapter.discover_models()] == [
        "localai-fast-model"
    ]
    assert adapter.list_loaded_models() == ()
    assert adapter.load_model("localai-fast-model").status == "unsupported"
    assert "LocalAI lifecycle" in adapter.capabilities().load_model.disabled_reason
    assert calls[0][0] == "http://127.0.0.1:8080/v1/models"


def test_vllm_adapter_detected_from_backend_name_or_host_hint():
    calls: list[tuple[str, dict[str, str], float]] = []

    def requester(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        return 200, {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    by_name = _backend(
        name="vllm",
        base_url="http://127.0.0.1:8000/v1",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    by_host = _backend(
        name="fast",
        base_url="http://vllm.local/v1",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )

    adapter = adapter_for_backend(by_name, requester=requester)
    host_adapter = adapter_for_backend(by_host, requester=requester)
    state = runtime_state_for_backend(by_name, requester=requester)

    assert isinstance(adapter, VLLMAdapter)
    assert isinstance(host_adapter, VLLMAdapter)
    assert adapter.detect().provider == "vllm"
    assert adapter.capabilities().runtime_kind == "vllm"
    assert adapter.health(timeout_seconds=0.05).ok is True
    assert [model.model_id for model in adapter.discover_models()] == [
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
    assert adapter.unload_model("meta-llama/Llama-3.1-8B-Instruct").status == (
        "unsupported"
    )
    assert "vLLM lifecycle" in adapter.capabilities().unload_model.disabled_reason
    assert state["provider"] == "vllm"
    assert state["runtime_kind"] == "vllm"
    assert calls[0][0] == "http://127.0.0.1:8000/v1/models"


def test_unknown_local_openai_backend_keeps_generic_adapter():
    backend = _backend(
        name="custom",
        base_url="http://127.0.0.1:8000/v1",
        model="custom-model",
    )
    adapter = adapter_for_backend(backend, requester=lambda *_args: (200, {"data": []}))

    assert isinstance(adapter, GenericOpenAICompatibleAdapter)
    assert not isinstance(adapter, VLLMAdapter)
    assert adapter.capabilities().provider == "openai_compatible_local"
    assert adapter.capabilities().runtime_kind == "openai_compatible_local"


def test_lmstudio_adapter_detects_cli_and_keeps_native_lifecycle_disabled():
    commands: list[tuple[str, ...]] = []

    def requester(_url, _headers, _timeout):
        return 200, {"data": [{"id": "lmstudio-fast"}, {"id": "lmstudio-code"}]}

    def command_runner(command, timeout):
        del timeout
        commands.append(tuple(command))
        return CommandResult(0)

    adapter = LMStudioAdapter(
        _backend(base_url="http://127.0.0.1:1234/v1", model="lmstudio-fast"),
        requester=requester,
        command_runner=command_runner,
        command_resolver=lambda command: f"/usr/local/bin/{command}"
        if command == "lms"
        else None,
    )

    detection = adapter.detect()
    capabilities = adapter.capabilities().to_dict()
    models = adapter.discover_models()
    loaded = adapter.list_loaded_models()
    load = adapter.load_model("lmstudio-fast")
    unload = adapter.unload_model("lmstudio-fast")
    start = adapter.start_server()

    assert detection.installed is True
    assert detection.command == ("lms",)
    assert [model.model_id for model in models] == ["lmstudio-fast", "lmstudio-code"]
    assert loaded == ()
    assert capabilities["list_loaded_models"]["supported"] is False
    assert "stable OpenAI-compatible API" in capabilities["list_loaded_models"][
        "disabled_reason"
    ]
    assert capabilities["load_model"]["supported"] is False
    assert "stable local CLI/API contract" in capabilities["load_model"][
        "disabled_reason"
    ]
    assert load.status == "unsupported"
    assert unload.status == "unsupported"
    assert start.status == "unsupported"
    assert commands == []


def test_lmstudio_runtime_state_imports_real_model_ids_with_source_metadata():
    def requester(_url, _headers, _timeout):
        return 200, {
            "data": [
                {
                    "id": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                    "object": "model",
                    "owned_by": "lm-studio",
                },
                {"id": "qwen2.5-coder-7b-instruct"},
            ]
        }

    state = runtime_state_for_backend(
        _backend(
            base_url="http://127.0.0.1:1234/v1",
            model="lmstudio-fast-model",
        ),
        requester=requester,
        checked_at="2026-07-01T12:00:00Z",
    )

    models = state["models"]
    assert state["runtime_id"] == "lmstudio"
    assert [model["model_id"] for model in models] == [
        "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "qwen2.5-coder-7b-instruct",
    ]
    assert models[0]["source"] == "lmstudio_models_api"
    assert models[0]["metadata"]["owned_by"] == "lm-studio"
    assert state["health"]["detail"] == "configured model 'lmstudio-fast-model' not listed"


def test_ollama_adapter_lists_models_and_loaded_models_with_mocked_cli():
    commands: list[tuple[str, ...]] = []

    def requester(_url, _headers, _timeout):
        raise AssertionError("ollama CLI inventory should avoid HTTP fallback")

    def command_runner(command, timeout):
        del timeout
        commands.append(tuple(command))
        if tuple(command) == ("ollama", "list"):
            return CommandResult(
                0,
                stdout=(
                    "NAME              ID          SIZE      MODIFIED\n"
                    "qwen3:4b          abc123      2.5 GB    1 hour ago\n"
                    "llama3.2:latest   def456      2.0 GB    2 days ago\n"
                ),
            )
        if tuple(command) == ("ollama", "ps"):
            return CommandResult(
                0,
                stdout=(
                    "NAME              ID          SIZE      PROCESSOR    UNTIL\n"
                    "qwen3:4b          abc123      2.5 GB    100% GPU      4 minutes\n"
                ),
            )
        raise AssertionError(f"unexpected command: {command!r}")

    adapter = OllamaAdapter(
        _backend(base_url="http://127.0.0.1:11434/v1", model="qwen3:4b"),
        requester=requester,
        command_runner=command_runner,
        command_resolver=lambda command: f"/usr/local/bin/{command}"
        if command == "ollama"
        else None,
    )

    detection = adapter.detect()
    capabilities = adapter.capabilities().to_dict()
    models = adapter.discover_models()
    loaded = adapter.list_loaded_models()

    assert detection.installed is True
    assert detection.command == ("ollama",)
    assert capabilities["discover_models"]["supported"] is True
    assert capabilities["list_loaded_models"]["supported"] is True
    assert capabilities["unload_model"]["supported"] is True
    assert capabilities["load_model"]["supported"] is False
    assert "does not run or pull models silently" in capabilities["load_model"][
        "disabled_reason"
    ]
    assert [model.model_id for model in models] == ["qwen3:4b", "llama3.2:latest"]
    assert models[0].tags == ("4b",)
    assert models[0].metadata == {
        "ollama_id": "abc123",
        "ollama_size": "2.5 GB",
        "ollama_modified": "1 hour ago",
    }
    assert models[1].tags == ("latest",)
    assert [model.to_dict() for model in loaded] == [
        {
            "model_id": "qwen3:4b",
            "loaded": True,
            "source": "ollama_cli",
            "tags": ["4b"],
            "metadata": {
                "ollama_id": "abc123",
                "ollama_size": "2.5 GB",
                "ollama_processor": "100% GPU",
                "ollama_until": "4 minutes",
            },
        }
    ]
    assert commands == [("ollama", "list"), ("ollama", "ps")]


def test_ollama_unload_is_explicit_and_does_not_load_or_pull():
    commands: list[tuple[str, ...]] = []

    def command_runner(command, timeout):
        del timeout
        commands.append(tuple(command))
        if tuple(command) == ("ollama", "stop", "qwen3:4b"):
            return CommandResult(0, stdout="")
        raise AssertionError(f"unexpected command: {command!r}")

    adapter = OllamaAdapter(
        _backend(base_url="http://127.0.0.1:11434/v1", model="qwen3:4b"),
        command_runner=command_runner,
        command_resolver=lambda command: f"/usr/local/bin/{command}"
        if command == "ollama"
        else None,
    )

    load = adapter.load_model("qwen3:4b")
    start = adapter.start_server()
    unload = adapter.unload_model("qwen3:4b")

    assert load.status == "unsupported"
    assert start.status == "unsupported"
    assert unload.ok is True
    assert unload.status == "unloaded"
    assert commands == [("ollama", "stop", "qwen3:4b")]


def test_ollama_adapter_reports_disabled_reasons_without_cli():
    adapter = OllamaAdapter(
        _backend(base_url="http://127.0.0.1:11434/v1", model="qwen3:4b"),
        command_resolver=lambda _command: None,
    )

    capabilities = adapter.capabilities().to_dict()
    loaded = adapter.list_loaded_models()
    unload = adapter.unload_model("qwen3:4b")

    assert adapter.detect().installed is False
    assert capabilities["discover_models"]["supported"] is True
    assert capabilities["list_loaded_models"]["supported"] is False
    assert "Ollama CLI not found" in capabilities["list_loaded_models"][
        "disabled_reason"
    ]
    assert capabilities["unload_model"]["supported"] is False
    assert "ollama CLI" in capabilities["unload_model"]["disabled_reason"]
    assert loaded == ()
    assert unload.status == "unsupported"


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


def test_managed_runtime_adapter_reports_missing_command_reasons(tmp_path):
    backend = _backend(
        base_url="http://127.0.0.1:8090/v1",
        runtime=ProxyRuntimeConfig(
            enabled=True,
            kind="llama-server",
            command=(
                "definitely-missing-model-router-runtime",
                "-m",
                "/models/fast.gguf",
                "--port",
                "8090",
            ),
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
    assert "command missing" in capabilities["start_server"]["disabled_reason"]
    assert capabilities["load_model"]["supported"] is False
    assert "command missing" in capabilities["load_model"]["disabled_reason"]
    assert logs["supported"] is True
    assert logs["paths"] == [str(tmp_path / "fast.log")]


def test_managed_runtime_adapter_start_stop_load_unload_are_pid_owned(
    tmp_path,
    monkeypatch,
):
    running_pids: set[int] = set()
    process_calls: list[tuple[list[str], dict[str, object]]] = []
    request_calls: list[str] = []

    class FakeProcess:
        pid = 43210

        def poll(self):
            return None

    def requester(url, _headers, _timeout):
        request_calls.append(url)
        if len(request_calls) == 1:
            raise URLError("offline before start")
        return 200, {"data": [{"id": "fast-model"}]}

    def process_factory(command, **kwargs):
        process_calls.append((list(command), dict(kwargs)))
        running_pids.add(FakeProcess.pid)
        return FakeProcess()

    def fake_pid_running(pid: int) -> bool:
        return pid in running_pids

    def fake_terminate_pid(pid: int, *, timeout_seconds: float) -> bool:
        assert timeout_seconds == 0.2
        running_pids.discard(pid)
        return True

    monkeypatch.setattr(runtime_adapters, "_pid_running", fake_pid_running)
    monkeypatch.setattr(runtime_adapters, "_terminate_pid", fake_terminate_pid)

    backend = _backend(
        base_url="http://127.0.0.1:8090/v1",
        model="fast-model",
        runtime=ProxyRuntimeConfig(
            enabled=True,
            kind="llama-server",
            command=(sys.executable, "-m", "fake_runtime"),
            readiness_url="http://127.0.0.1:8090/v1/models",
            readiness_timeout_seconds=0.2,
            shutdown_timeout_seconds=0.2,
            log_path=str(tmp_path / "fast.log"),
        ),
    )
    adapter = ManagedRuntimeAdapter(
        backend,
        requester=requester,
        process_factory=process_factory,
    )

    capabilities = adapter.capabilities().to_dict()
    started = adapter.start_server()
    pid_marker = json.loads((tmp_path / "fast.log.fast.pid").read_text(encoding="utf-8"))
    loaded_models = adapter.list_loaded_models()
    loaded = adapter.load_model("fast-model")
    unloaded = adapter.unload_model("fast-model")
    stopped_again = adapter.stop_server()
    payload = started.to_dict()

    assert capabilities["start_server"]["supported"] is True
    assert capabilities["stop_server"]["supported"] is True
    assert capabilities["load_model"]["supported"] is True
    assert capabilities["unload_model"]["supported"] is True
    assert started.status == "started"
    assert pid_marker["pid"] == 43210
    assert pid_marker["command"] == [sys.executable, "-m", "fake_runtime"]
    assert process_calls[0][0] == [sys.executable, "-m", "fake_runtime"]
    assert process_calls[0][1]["shell"] is False
    assert process_calls[0][1]["stdin"] == runtime_adapters.subprocess.DEVNULL
    assert loaded_models == (
        RuntimeModel("fast-model", loaded=True, source="modelrouter_managed"),
    )
    assert loaded.status == "already_loaded"
    assert unloaded.status == "unloaded"
    assert stopped_again.status == "not_running"
    assert "secret" not in str(payload).lower()


def test_managed_runtime_adapter_start_detects_already_running_pid(
    tmp_path,
    monkeypatch,
):
    calls: list[list[str]] = []
    monkeypatch.setattr(runtime_adapters, "_pid_running", lambda pid: pid == 43210)
    backend = _backend(
        base_url="http://127.0.0.1:8090/v1",
        model="fast-model",
        runtime=ProxyRuntimeConfig(
            enabled=True,
            kind="llama-server",
            command=(sys.executable, "-m", "fake_runtime"),
            readiness_url="http://127.0.0.1:8090/v1/models",
            log_path=str(tmp_path / "fast.log"),
        ),
    )
    (tmp_path / "fast.log.fast.pid").write_text(
        json.dumps({"pid": 43210, "command": [sys.executable, "-m", "fake_runtime"]}),
        encoding="utf-8",
    )
    adapter = ManagedRuntimeAdapter(
        backend,
        requester=lambda *_args: (_ for _ in ()).throw(URLError("should not probe")),
        process_factory=lambda command, **_kwargs: calls.append(list(command)),
    )

    result = adapter.start_server()

    assert result.ok is True
    assert result.status == "already_running"
    assert calls == []


def test_managed_runtime_adapter_mismatched_pid_marker_leaves_process_untouched(
    tmp_path,
    monkeypatch,
):
    terminate_calls: list[int] = []
    process_calls: list[list[str]] = []
    monkeypatch.setattr(runtime_adapters, "_pid_running", lambda pid: pid == 43210)
    monkeypatch.setattr(
        runtime_adapters,
        "_terminate_pid",
        lambda pid, *, timeout_seconds: terminate_calls.append(pid) or True,
    )
    backend = _backend(
        base_url="http://127.0.0.1:8090/v1",
        model="fast-model",
        runtime=ProxyRuntimeConfig(
            enabled=True,
            kind="llama-server",
            command=(sys.executable, "-m", "new_runtime"),
            readiness_url="http://127.0.0.1:8090/v1/models",
            log_path=str(tmp_path / "fast.log"),
        ),
    )
    (tmp_path / "fast.log.fast.pid").write_text(
        json.dumps({"pid": 43210, "command": [sys.executable, "-m", "old_runtime"]}),
        encoding="utf-8",
    )
    adapter = ManagedRuntimeAdapter(
        backend,
        requester=lambda *_args: (_ for _ in ()).throw(URLError("should not probe")),
        process_factory=lambda command, **_kwargs: process_calls.append(list(command)),
    )

    start = adapter.start_server()
    stop = adapter.stop_server()
    loaded_models = adapter.list_loaded_models()

    assert start.ok is False
    assert start.status == "blocked"
    assert stop.ok is False
    assert stop.status == "blocked"
    assert "does not match" in stop.disabled_reason
    assert loaded_models == ()
    assert process_calls == []
    assert terminate_calls == []


def test_managed_runtime_adapter_stop_without_pid_leaves_external_runtime_alone(
    tmp_path,
    monkeypatch,
):
    terminate_calls: list[int] = []
    monkeypatch.setattr(
        runtime_adapters,
        "_terminate_pid",
        lambda pid, *, timeout_seconds: terminate_calls.append(pid) or True,
    )
    backend = _backend(
        base_url="http://127.0.0.1:8090/v1",
        model="fast-model",
        runtime=ProxyRuntimeConfig(
            enabled=True,
            kind="llama-server",
            command=(sys.executable, "-m", "fake_runtime"),
            readiness_url="http://127.0.0.1:8090/v1/models",
            log_path=str(tmp_path / "fast.log"),
        ),
    )

    result = ManagedRuntimeAdapter(backend).stop_server()

    assert result.ok is True
    assert result.status == "not_running"
    assert "externally managed runtimes were left untouched" in result.message
    assert terminate_calls == []


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
