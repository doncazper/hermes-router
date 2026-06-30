import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
import logging
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

import hermes.plugins.model_router.proxy as proxy_module
from hermes.plugins.model_router.proxy import (
    ProxySessionStats,
    create_app,
    _format_proxy_session_summary,
    _stream_response_bytes,
)
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyBackendPolicyConfig,
    ProxyObservabilityConfig,
    ProxyRuntimeConfig,
    ProxyServerConfig,
    ProxyVerifierConfig,
    RoutingProxyConfig,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeStream:
    exits = 0

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.status_code = response.status_code
        self.headers = response.headers
        self.raise_after = response.extensions.get("raise_after")

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        type(self).exits += 1
        return None

    async def aiter_bytes(self):
        yield self.response.content
        if self.raise_after is not None:
            raise self.raise_after


class _FakeAsyncClient:
    responses: dict[str, list[httpx.Response | Exception]] = {}
    requests: list[dict[str, Any]] = []

    def __init__(self, *, base_url: str, headers: dict[str, str], timeout: float) -> None:
        self.base_url = base_url
        self.headers = headers
        self.timeout = timeout
        self.backend_name = str(base_url).split("//", 1)[-1].split(".", 1)[0]

    async def aclose(self) -> None:
        return None

    async def post(self, path: str, *, content: bytes, timeout: float | None = None):
        self.requests.append(
            {
                "backend": self.backend_name,
                "path": path,
                "headers": self.headers,
                "body": json.loads(content.decode("utf-8")),
                "timeout": timeout,
            }
        )
        response = self._next_response()
        if isinstance(response, Exception):
            raise response
        return response

    async def get(self, path: str, *, timeout: float):
        self.requests.append(
            {
                "backend": self.backend_name,
                "path": path,
                "headers": self.headers,
                "timeout": timeout,
                "method": "GET",
            }
        )
        response = self._next_response()
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, method: str, path: str, *, content: bytes):
        assert method == "POST"
        self.requests.append(
            {
                "backend": self.backend_name,
                "path": path,
                "headers": self.headers,
                "body": json.loads(content.decode("utf-8")),
                "stream": True,
            }
        )
        response = self._next_response()
        if isinstance(response, Exception):
            raise response
        return _FakeStream(response)

    def _next_response(self) -> httpx.Response | Exception:
        responses = self.responses.setdefault(self.backend_name, [])
        if responses:
            return responses.pop(0)
        return httpx.Response(
            200,
            json={
                "id": f"cmpl-{self.backend_name}",
                "object": "chat.completion",
                "model": f"{self.backend_name}-model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )


class _LiveStreamState:
    def __init__(self) -> None:
        self.request_seen = threading.Event()
        self.first_chunk_sent = threading.Event()
        self.stream_closed = threading.Event()
        self.headers: dict[str, str] = {}
        self.body: dict[str, Any] = {}


def _response(status_code: int, payload: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload
        or {
            "id": f"status-{status_code}",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
    )


def _stream_response(
    content: bytes = b"data: hello\n\n",
    *,
    raise_after: Exception | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        content=content,
        headers={"content-type": "text/event-stream"},
        extensions={"raise_after": raise_after} if raise_after else {},
    )


def _config(
    *,
    api_key: str | None = None,
    fallback: bool = True,
    log_path: Path | None = None,
    fast_runtime: ProxyRuntimeConfig | None = None,
    routing_profile: str = "balanced",
    backend_policy: ProxyBackendPolicyConfig | None = None,
    verifier: ProxyVerifierConfig | None = None,
) -> RoutingProxyConfig:
    fallback_backends = {"fast": ("deep",)} if fallback else {}
    return RoutingProxyConfig(
        proxy=ProxyServerConfig(api_key=api_key, routing_profile=routing_profile),
        router_config=None,
        source_path="test.yaml",
        backends={
            "fast": ProxyBackendConfig(
                name="fast",
                base_url="http://fast.test/v1",
                model="fast-model",
                strip_tools=True,
                runtime=fast_runtime or ProxyRuntimeConfig(),
            ),
            "deep": ProxyBackendConfig(
                name="deep",
                base_url="http://deep.test/v1",
                model="deep-model",
                api_key="deep-secret",
            ),
        },
        engine_backends={
            "fast_local": "fast",
            "balanced_local": "fast",
            "reasoning_local": "deep",
            "code_agent": "deep",
            "web_research": "deep",
            "multimodal_vision": "deep",
            "image_generation": "deep",
        },
        fallback_backends=fallback_backends,
        observability=ProxyObservabilityConfig(
            enabled=log_path is not None,
            log_path=str(log_path or "~/.model-router/routing-events.jsonl"),
        ),
        backend_policy=backend_policy or ProxyBackendPolicyConfig(),
        verifier=verifier or ProxyVerifierConfig(),
    )


def _managed_runtime(tmp_path: Path, *, kind: str = "llama-server") -> ProxyRuntimeConfig:
    return ProxyRuntimeConfig(
        enabled=True,
        kind=kind,
        command=("fake-runtime", "--port", "8090"),
        readiness_url="http://127.0.0.1:8090/v1/models",
        readiness_timeout_seconds=1.0,
        idle_timeout_seconds=900.0,
        shutdown_timeout_seconds=1.0,
        log_path=str(tmp_path / "runtime.log"),
    )


def _patch_runtime_manager(monkeypatch, *, start_error: Exception | None = None):
    class FakeRuntimeManager:
        instances: list["FakeRuntimeManager"] = []

        def __init__(self, backends, **_kwargs):
            self.backends = backends
            self.has_managed_backends = any(
                backend.runtime.enabled for backend in backends.values()
            )
            self.ensure_calls: list[str] = []
            self.begin_calls: list[str] = []
            self.end_calls: list[str] = []
            self.touch_calls: list[str] = []
            self.stop_all_called = False
            type(self).instances.append(self)

        async def ensure_running(self, backend):
            self.ensure_calls.append(backend.name)
            if start_error is not None:
                raise start_error

        def begin_request(self, backend_name: str) -> None:
            self.begin_calls.append(backend_name)

        def end_request(self, backend_name: str) -> None:
            self.end_calls.append(backend_name)

        def touch(self, backend_name: str) -> None:
            self.touch_calls.append(backend_name)

        async def reap_idle_forever(self):
            await asyncio.Event().wait()

        async def stop_all(self):
            self.stop_all_called = True

    monkeypatch.setattr(proxy_module, "ManagedRuntimeManager", FakeRuntimeManager)
    return FakeRuntimeManager


def _client(monkeypatch, config: RoutingProxyConfig) -> TestClient:
    _FakeAsyncClient.responses = {}
    _FakeAsyncClient.requests = []
    _FakeStream.exits = 0
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    return TestClient(create_app(config))


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _uvicorn_server(app):
    import uvicorn

    port = _unused_tcp_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive():
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for test ASGI server")
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("test ASGI server exited before startup")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("test ASGI server did not stop")


def _live_streaming_upstream(state: _LiveStreamState):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI(docs_url=None, redoc_url=None)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        state.headers = dict(request.headers)
        body = await request.json()
        state.body = body if isinstance(body, dict) else {}
        state.request_seen.set()

        async def stream_body():
            try:
                yield b"data: first\n\n"
                state.first_chunk_sent.set()
                while True:
                    await asyncio.sleep(0.05)
                    yield b"data: still-open\n\n"
            finally:
                state.stream_closed.set()

        return StreamingResponse(stream_body(), media_type="text/event-stream")

    return app


def _live_proxy_config(
    *,
    upstream_base_url: str,
    log_path: Path,
) -> RoutingProxyConfig:
    engine_backends = {
        engine: "fast"
        for engine in (
            "fast_local",
            "balanced_local",
            "reasoning_local",
            "code_agent",
            "web_research",
            "multimodal_vision",
            "image_generation",
        )
    }
    return RoutingProxyConfig(
        proxy=ProxyServerConfig(api_key="client-secret"),
        router_config=None,
        source_path="live-disconnect-test.yaml",
        backends={
            "fast": ProxyBackendConfig(
                name="fast",
                base_url=f"{upstream_base_url}/v1",
                model="fast-live-model",
                api_key="backend-secret",
                timeout_seconds=5.0,
            )
        },
        engine_backends=engine_backends,
        fallback_backends={},
        observability=ProxyObservabilityConfig(
            enabled=True,
            log_path=str(log_path),
            prompt_capture="off",
        ),
    )


def _stream_and_disconnect(proxy_base_url: str, payload: dict[str, Any]) -> bytes:
    parsed = urlparse(proxy_base_url)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {parsed.hostname}:{parsed.port}\r\n"
        "Authorization: Bearer client-secret\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii") + body

    data = b""
    with socket.create_connection((parsed.hostname, parsed.port), timeout=5) as sock:
        sock.settimeout(5)
        sock.sendall(request)
        deadline = time.monotonic() + 5
        while b"data: first" not in data:
            if time.monotonic() > deadline:
                raise AssertionError(f"stream did not produce first chunk: {data!r}")
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return data


def _wait_for_log_row(path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                return json.loads(lines[-1])
        time.sleep(0.02)
    raise AssertionError(f"routing log row was not written to {path}")


def _assert_route_headers(
    response,
    *,
    engine: str,
    backend: str | None = None,
    model: str | None = None,
    fallback: str = "false",
    profile: str = "balanced",
    mode: str = "decision",
    decision_layer: str = "on",
    route_api: str = "route_fast",
) -> None:
    assert response.headers["x-modelrouter-request-id"]
    assert response.headers["x-modelrouter-engine"] == engine
    assert response.headers["x-modelrouter-mode"] == mode
    assert response.headers["x-modelrouter-decision-layer"] == decision_layer
    assert response.headers["x-modelrouter-profile"] == profile
    assert response.headers["x-modelrouter-route-api"] == route_api
    assert response.headers["x-modelrouter-fallback"] == fallback
    assert response.headers["x-request-id"] == response.headers[
        "x-modelrouter-request-id"
    ]
    assert response.headers["x-routed-engine"] == engine
    if backend is None:
        assert "x-modelrouter-backend" not in response.headers
        assert "x-routed-backend" not in response.headers
    else:
        assert response.headers["x-modelrouter-backend"] == backend
        assert response.headers["x-routed-backend"] == backend
    if model is not None:
        assert response.headers["x-modelrouter-model"] == model


def _serialized_route_headers(response) -> str:
    return json.dumps(
        {
            key: value
            for key, value in response.headers.items()
            if key.startswith("x-modelrouter")
            or key.startswith("x-routed")
            or key == "x-request-id"
        }
    )


def test_proxy_routes_to_backend_and_overrides_model(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "client-visible-model",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 200
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert response.headers["x-routed-engine"] == "fast_local"
    assert response.headers["x-routed-backend"] == "fast"
    assert response.headers["x-modelrouter-model"] == "fast-model"
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "fast-model"


def test_proxy_manual_mode_forwards_default_backend_without_route_fast(
    monkeypatch,
    tmp_path,
):
    def fail_route(*_args, **_kwargs):
        raise AssertionError("decision layer should be disabled")

    monkeypatch.setattr(proxy_module.ModelRouter, "route_fast", fail_route)
    monkeypatch.setattr(proxy_module.ModelRouter, "route", fail_route)
    log_path = tmp_path / "routing-events.jsonl"
    config = replace(
        _config(log_path=log_path),
        proxy=ProxyServerConfig(
            routing_mode="manual",
            default_backend="deep",
            default_model="manual-model",
            respect_client_model=False,
        ),
    )

    with _client(monkeypatch, config) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "client-visible-model",
                "messages": [
                    {"role": "user", "content": "api_key=secret route this"}
                ],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert response.status_code == 200
    _assert_route_headers(
        response,
        engine="manual",
        backend="deep",
        model="manual-model",
        mode="manual",
        decision_layer="off",
        route_api="manual",
    )
    assert _FakeAsyncClient.requests[0]["backend"] == "deep"
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "manual-model"
    assert row["routing_mode"] == "manual"
    assert row["decision_layer_enabled"] is False
    assert row["selected_engine"] == "manual"
    assert row["selected_backend"] == "deep"
    assert row["selected_model"] == "manual-model"
    assert row["route_api"] == "manual"
    assert row["prompt_length"] == 0
    assert row["receipt_summary"].startswith("Manual routing selected deep")
    assert "secret" not in json.dumps(row)


def test_proxy_manual_mode_static_safety_blocks_without_route_fast(
    monkeypatch,
    tmp_path,
):
    def fail_route(*_args, **_kwargs):
        raise AssertionError("decision layer should be disabled")

    monkeypatch.setattr(proxy_module.ModelRouter, "route_fast", fail_route)
    monkeypatch.setattr(proxy_module.ModelRouter, "route", fail_route)
    log_path = tmp_path / "routing-events.jsonl"
    config = replace(
        _config(log_path=log_path),
        proxy=ProxyServerConfig(
            routing_mode="manual",
            default_backend="deep",
            default_model="manual-model",
            respect_client_model=False,
            safety_gate_mode="always_static",
        ),
    )

    with _client(monkeypatch, config) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "client-visible-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "api_key=secret drop production database",
                    }
                ],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert response.status_code == 409
    assert response.json()["selected_engine"] == "human_confirm"
    _assert_route_headers(
        response,
        engine="human_confirm",
        mode="manual",
        decision_layer="off",
        route_api="manual",
    )
    assert _FakeAsyncClient.requests == []
    assert row["routing_mode"] == "manual"
    assert row["decision_layer_enabled"] is False
    assert row["status"] == "human_confirm"
    assert row["route_api"] == "manual"
    assert "secret" not in json.dumps(row)


def test_proxy_manual_mode_respects_known_client_model(monkeypatch):
    config = replace(
        _config(),
        proxy=ProxyServerConfig(
            routing_mode="manual",
            default_backend="deep",
            default_model="manual-model",
            respect_client_model=True,
        ),
    )

    with _client(monkeypatch, config) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "deep-model",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 200
    _assert_route_headers(
        response,
        engine="manual",
        backend="deep",
        model="deep-model",
        mode="manual",
        decision_layer="off",
        route_api="manual",
    )
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "deep-model"


def test_proxy_manual_mode_does_not_forward_public_proxy_model_id(monkeypatch):
    config = replace(
        _config(),
        proxy=ProxyServerConfig(
            routing_mode="manual",
            default_backend="deep",
            default_model="manual-model",
            model_ids=("model-router",),
            respect_client_model=True,
            unknown_model_behavior="fallback_to_default",
        ),
    )

    with _client(monkeypatch, config) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 200
    _assert_route_headers(
        response,
        engine="manual",
        backend="deep",
        model="manual-model",
        mode="manual",
        decision_layer="off",
        route_api="manual",
    )
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "manual-model"


def test_proxy_manual_mode_rejects_unknown_client_model_when_configured(monkeypatch):
    config = replace(
        _config(),
        proxy=ProxyServerConfig(
            routing_mode="manual",
            default_backend="deep",
            default_model="manual-model",
            respect_client_model=True,
            unknown_model_behavior="reject_404",
        ),
    )

    with _client(monkeypatch, config) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "unknown-client-model",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 404
    _assert_route_headers(
        response,
        engine="manual",
        backend="deep",
        model="manual-model",
        mode="manual",
        decision_layer="off",
        route_api="manual",
    )
    assert response.json()["error"]["type"] == "unknown_model"
    assert _FakeAsyncClient.requests == []


def test_proxy_applies_configured_routing_profile(monkeypatch):
    with _client(monkeypatch, _config(routing_profile="private")) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "client-visible-model",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )
        health = client.get("/health")

    assert response.status_code == 200
    _assert_route_headers(
        response,
        engine="fast_local",
        backend="fast",
        profile="private",
    )
    assert health.json()["routing_profile"] == "private"


def test_proxy_starts_managed_runtime_on_first_routed_request(monkeypatch, tmp_path):
    fake_manager = _patch_runtime_manager(monkeypatch)
    with _client(
        monkeypatch,
        _config(fast_runtime=_managed_runtime(tmp_path)),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    manager = fake_manager.instances[-1]
    assert response.status_code == 200
    assert manager.ensure_calls == ["fast"]
    assert manager.begin_calls == ["fast"]
    assert manager.end_calls == ["fast"]
    assert manager.touch_calls == ["fast"]
    assert manager.stop_all_called is True
    assert _FakeAsyncClient.requests[0]["backend"] == "fast"


def test_proxy_runtime_start_failure_returns_safe_502(
    monkeypatch,
    tmp_path,
):
    log_path = tmp_path / "routing-events.jsonl"
    _patch_runtime_manager(
        monkeypatch,
        start_error=proxy_module.RuntimeStartError("runtime fast command not found"),
    )
    with _client(
        monkeypatch,
        _config(
            log_path=log_path,
            fast_runtime=_managed_runtime(tmp_path),
        ),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [
                    {"role": "user", "content": "rewrite this api_key=secret"}
                ],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    serialized = json.dumps(response.json())
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "runtime_start_failed"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert _FakeAsyncClient.requests == []
    assert row["status"] == "runtime_start_failed"
    assert "secret" not in serialized
    assert "messages" not in serialized


def test_proxy_blocks_human_confirm_without_upstream_call(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "drop production database"}],
            },
        )

    assert response.status_code == 409
    assert response.json()["selected_engine"] == "human_confirm"
    _assert_route_headers(response, engine="human_confirm")
    assert _FakeAsyncClient.requests == []


def test_proxy_strips_tools_only_for_configured_backend(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "functions": [{"name": "legacy"}],
            },
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "fix the repo"}],
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "functions": [{"name": "legacy"}],
            },
        )

    fast_body = _FakeAsyncClient.requests[0]["body"]
    deep_body = _FakeAsyncClient.requests[1]["body"]
    assert "tools" not in fast_body
    assert "tool_choice" not in fast_body
    assert "parallel_tool_calls" not in fast_body
    assert "functions" not in fast_body
    assert deep_body["tools"] == [{"type": "function"}]
    assert deep_body["tool_choice"] == "auto"
    assert deep_body["parallel_tool_calls"] is True
    assert deep_body["functions"] == [{"name": "legacy"}]


def test_proxy_responses_routes_to_backend_and_preserves_common_shape(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "client-visible-model",
                "input": [
                    {"role": "system", "content": "be concise"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "fix the repo and run tests",
                            }
                        ],
                    },
                ],
                "instructions": "Return a compact answer.",
                "tools": [{"type": "function", "name": "run_tests"}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "metadata": {"client": "compat-test"},
                "previous_response_id": "resp_previous",
            },
        )

    body = _FakeAsyncClient.requests[0]["body"]
    assert response.status_code == 200
    _assert_route_headers(response, engine="code_agent", backend="deep")
    assert response.headers["x-routed-engine"] == "code_agent"
    assert response.headers["x-routed-backend"] == "deep"
    assert _FakeAsyncClient.requests[0]["path"] == "/responses"
    assert body["model"] == "deep-model"
    assert body["input"][1]["content"][0]["text"] == "fix the repo and run tests"
    assert body["instructions"] == "Return a compact answer."
    assert body["tools"] == [{"type": "function", "name": "run_tests"}]
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True
    assert body["metadata"] == {"client": "compat-test"}
    assert body["previous_response_id"] == "resp_previous"


def test_proxy_responses_streaming_preserves_sse_bytes(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        _FakeAsyncClient.responses = {
            "fast": [_stream_response(b"event: response.output_text.delta\n\n")]
        }
        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "model-router",
                "stream": True,
                "input": "rewrite this text",
            },
        ) as response:
            body = response.read()

    assert response.status_code == 200
    assert body == b"event: response.output_text.delta\n\n"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert response.headers["x-routed-backend"] == "fast"
    assert _FakeAsyncClient.requests[0]["path"] == "/responses"
    assert _FakeAsyncClient.requests[0]["stream"] is True
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "fast-model"


def test_proxy_mlx_lm_runtime_rejects_responses_without_bridge(
    monkeypatch,
    tmp_path,
):
    fake_manager = _patch_runtime_manager(monkeypatch)
    with _client(
        monkeypatch,
        _config(fast_runtime=_managed_runtime(tmp_path, kind="mlx-lm")),
    ) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "model-router",
                "input": "rewrite this text",
            },
        )

    manager = fake_manager.instances[-1]
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_api_unsupported"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert manager.ensure_calls == []
    assert _FakeAsyncClient.requests == []


def test_proxy_responses_blocks_human_confirm_without_upstream_call(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "model-router",
                "input": "delete all production records",
            },
        )

    assert response.status_code == 409
    assert response.json()["selected_engine"] == "human_confirm"
    _assert_route_headers(response, engine="human_confirm")
    assert _FakeAsyncClient.requests == []


def test_proxy_embeddings_routes_to_backend_and_preserves_input(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "client-visible-model",
                "input": ["rewrite this text", "another string"],
                "encoding_format": "float",
            },
        )

    body = _FakeAsyncClient.requests[0]["body"]
    assert response.status_code == 200
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert _FakeAsyncClient.requests[0]["path"] == "/embeddings"
    assert body["model"] == "fast-model"
    assert body["input"] == ["rewrite this text", "another string"]
    assert body["encoding_format"] == "float"


def test_proxy_embeddings_manual_mode_does_not_call_route_fast(monkeypatch):
    def fail_route(*_args, **_kwargs):
        raise AssertionError("decision layer should be disabled")

    monkeypatch.setattr(proxy_module.ModelRouter, "route_fast", fail_route)
    monkeypatch.setattr(proxy_module.ModelRouter, "route", fail_route)
    config = replace(
        _config(),
        proxy=ProxyServerConfig(
            routing_mode="manual",
            default_backend="deep",
            default_model="manual-embedding-model",
            respect_client_model=False,
        ),
    )

    with _client(monkeypatch, config) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "client-visible-model",
                "input": "api_key=secret embed this",
            },
        )

    assert response.status_code == 200
    _assert_route_headers(
        response,
        engine="manual",
        backend="deep",
        model="manual-embedding-model",
        mode="manual",
        decision_layer="off",
        route_api="manual",
    )
    assert _FakeAsyncClient.requests[0]["path"] == "/embeddings"
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "manual-embedding-model"


def test_proxy_completions_routes_to_backend_and_preserves_prompt(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "client-visible-model",
                "prompt": "fix the repo and run tests",
                "max_tokens": 64,
            },
        )

    body = _FakeAsyncClient.requests[0]["body"]
    assert response.status_code == 200
    _assert_route_headers(response, engine="code_agent", backend="deep")
    assert _FakeAsyncClient.requests[0]["path"] == "/completions"
    assert body["model"] == "deep-model"
    assert body["prompt"] == "fix the repo and run tests"
    assert body["max_tokens"] == 64


def test_proxy_mlx_lm_runtime_rejects_unsupported_compat_endpoints(
    monkeypatch,
    tmp_path,
):
    fake_manager = _patch_runtime_manager(monkeypatch)
    config = _config(fast_runtime=_managed_runtime(tmp_path, kind="mlx-lm"))

    with _client(monkeypatch, config) as client:
        embeddings = client.post(
            "/v1/embeddings",
            json={"model": "model-router", "input": "rewrite this text"},
        )
        completions = client.post(
            "/v1/completions",
            json={"model": "model-router", "prompt": "rewrite this text"},
        )

    manager = fake_manager.instances[-1]
    assert embeddings.status_code == 502
    assert embeddings.json()["error"]["type"] == "upstream_api_unsupported"
    assert completions.status_code == 502
    assert completions.json()["error"]["type"] == "upstream_api_unsupported"
    assert manager.ensure_calls == []
    assert _FakeAsyncClient.requests == []


def test_proxy_messages_returns_shaped_unsupported_error(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/messages",
            json={"model": "model-router", "messages": []},
        )

    payload = response.json()
    assert response.status_code == 501
    assert payload["error"]["type"] == "unsupported_endpoint"
    assert payload["modelrouter"]["endpoint"] == "/v1/messages"
    assert "/v1/messages" in payload["modelrouter"]["planned_endpoints"]
    assert _FakeAsyncClient.requests == []


def test_proxy_experimental_endpoint_failure_does_not_break_decision_chat(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        unsupported = client.post(
            "/v1/messages",
            json={"model": "model-router", "messages": []},
        )
        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert unsupported.status_code == 501
    assert unsupported.json()["error"]["type"] == "unsupported_endpoint"
    assert chat.status_code == 200
    _assert_route_headers(chat, engine="fast_local", backend="fast")
    assert [request["path"] for request in _FakeAsyncClient.requests] == [
        "/chat/completions"
    ]


def test_proxy_unknown_v1_endpoint_returns_shaped_unsupported_error(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            json={"model": "model-router"},
        )

    payload = response.json()
    assert response.status_code == 404
    assert payload["error"]["type"] == "unsupported_endpoint"
    assert payload["modelrouter"]["endpoint"] == "/v1/audio/transcriptions"
    assert "/v1/embeddings" in payload["modelrouter"]["supported_endpoints"]
    assert _FakeAsyncClient.requests == []


def test_proxy_models_exposes_aliases_and_capability_hints(monkeypatch):
    config = replace(
        _config(),
        proxy=ProxyServerConfig(
            model_ids=("model-router", "model-router-fast"),
        ),
    )

    with _client(monkeypatch, config) as client:
        response = client.get("/v1/models")

    payload = response.json()
    models = {item["id"]: item for item in payload["data"]}
    assert response.status_code == 200
    assert list(models)[:2] == ["model-router", "model-router-fast"]
    assert models["model-router"]["modelrouter"]["kind"] == "proxy_alias"
    assert models["model-router"]["capabilities"]["chat_completions"] is True
    assert models["model-router"]["capabilities"]["embeddings"] is True
    assert models["model-router"]["capabilities"]["completions"] is True
    assert models["model-router"]["capabilities"]["messages"] is False
    assert models["fast-model"]["modelrouter"] == {
        "kind": "backend_model",
        "backend": "fast",
        "runtime_kind": "openai-compatible",
        "managed": False,
    }
    assert models["fast-model"]["capabilities"]["models"] is True


def test_proxy_uses_explicit_fallback_chain_on_upstream_5xx(monkeypatch):
    _FakeAsyncClient.responses = {"fast": [_response(503)], "deep": [_response(200)]}
    with _client(monkeypatch, _config()) as client:
        _FakeAsyncClient.responses = {"fast": [_response(503)], "deep": [_response(200)]}
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["x-routed-backend"] == "deep"
    _assert_route_headers(response, engine="fast_local", backend="deep", fallback="true")
    assert [request["backend"] for request in _FakeAsyncClient.requests] == [
        "fast",
        "deep",
    ]


def test_proxy_backend_denylist_blocks_forwarding(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(
        monkeypatch,
        _config(
            log_path=log_path,
            backend_policy=ProxyBackendPolicyConfig(backend_denylist=("fast",)),
        ),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "backend_policy_rejected"
    assert "backend fast denied by backend policy" in response.json()["error"]["message"]
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert row["status"] == "backend_policy_rejected"
    assert row["backend"] == "fast"
    assert row["receipt_summary"].startswith("Selected fast_local")
    assert "route.simple" in row["reason_codes"]
    assert _FakeAsyncClient.requests == []


def test_proxy_backend_allowlist_blocks_non_allowed_backend(monkeypatch):
    with _client(
        monkeypatch,
        _config(
            backend_policy=ProxyBackendPolicyConfig(backend_allowlist=("deep",)),
        ),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "backend_policy_rejected"
    assert "backend fast not allowed by backend policy" in response.json()["error"][
        "message"
    ]
    assert _FakeAsyncClient.requests == []


def test_proxy_fallback_chain_does_not_escape_denied_backend(monkeypatch):
    with _client(
        monkeypatch,
        _config(
            backend_policy=ProxyBackendPolicyConfig(backend_denylist=("deep",)),
        ),
    ) as client:
        _FakeAsyncClient.responses = {"fast": [_response(503)]}
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 503
    assert response.headers["x-routed-backend"] == "fast"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert [request["backend"] for request in _FakeAsyncClient.requests] == ["fast"]


def test_proxy_does_not_fallback_when_chain_absent(monkeypatch):
    with _client(monkeypatch, _config(fallback=False)) as client:
        _FakeAsyncClient.responses = {"fast": [_response(503)]}
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 503
    assert response.headers["x-routed-backend"] == "fast"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert [request["backend"] for request in _FakeAsyncClient.requests] == ["fast"]


def test_proxy_timeout_returns_502_when_fallback_unavailable(monkeypatch):
    with _client(monkeypatch, _config(fallback=False)) as client:
        _FakeAsyncClient.responses = {"fast": [httpx.TimeoutException("slow")]}
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_request_failed"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert [request["backend"] for request in _FakeAsyncClient.requests] == ["fast"]


def test_proxy_auth_accepts_valid_token_and_rejects_missing_or_wrong_token(monkeypatch):
    with _client(monkeypatch, _config(api_key="proxy-secret")) as client:
        missing = client.get("/v1/models")
        wrong = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        ok = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer proxy-secret"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200


def test_proxy_streaming_preserves_sse_bytes(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        _FakeAsyncClient.responses = {"fast": [_stream_response(b"data: chunk\n\n")]}
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "stream": True,
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        ) as response:
            body = response.read()

    assert response.status_code == 200
    assert body == b"data: chunk\n\n"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert _FakeAsyncClient.requests[0]["stream"] is True


def test_proxy_streaming_touches_managed_runtime_after_cleanup(monkeypatch, tmp_path):
    fake_manager = _patch_runtime_manager(monkeypatch)
    with _client(
        monkeypatch,
        _config(fast_runtime=_managed_runtime(tmp_path)),
    ) as client:
        _FakeAsyncClient.responses = {"fast": [_stream_response(b"data: chunk\n\n")]}
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "stream": True,
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        ) as response:
            body = response.read()

    manager = fake_manager.instances[-1]
    assert response.status_code == 200
    assert body == b"data: chunk\n\n"
    assert manager.ensure_calls == ["fast"]
    assert manager.begin_calls == ["fast"]
    assert manager.end_calls == ["fast"]
    assert manager.touch_calls == []


def test_proxy_streaming_preserves_final_upstream_status(monkeypatch):
    with _client(monkeypatch, _config(fallback=False)) as client:
        _FakeAsyncClient.responses = {"fast": [httpx.Response(503, content=b"busy")]}
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "stream": True,
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        ) as response:
            body = response.read()

    assert response.status_code == 503
    assert body == b"busy"
    assert response.headers["x-routed-backend"] == "fast"
    _assert_route_headers(response, engine="fast_local", backend="fast")


def test_proxy_streaming_uses_explicit_fallback_on_upstream_5xx(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        _FakeAsyncClient.responses = {
            "fast": [httpx.Response(503, content=b"busy")],
            "deep": [_stream_response(b"data: fallback\n\n")],
        }
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "stream": True,
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        ) as response:
            body = response.read()

    assert response.status_code == 200
    assert body == b"data: fallback\n\n"
    assert response.headers["x-routed-backend"] == "deep"
    _assert_route_headers(response, engine="fast_local", backend="deep", fallback="true")
    assert [request["backend"] for request in _FakeAsyncClient.requests] == [
        "fast",
        "deep",
    ]
    assert _FakeStream.exits == 2


def test_proxy_streaming_open_timeout_returns_502_without_fallback(monkeypatch):
    with _client(monkeypatch, _config(fallback=False)) as client:
        _FakeAsyncClient.responses = {"fast": [httpx.TimeoutException("slow")]}
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "stream": True,
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_request_failed"
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert [request["backend"] for request in _FakeAsyncClient.requests] == ["fast"]


def test_proxy_streaming_body_disconnect_logs_interruption(
    monkeypatch,
    tmp_path,
):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(monkeypatch, _config(log_path=log_path)) as client:
        _FakeAsyncClient.responses = {
            "fast": [_stream_response(b"data: first\n\n", raise_after=RuntimeError("lost"))]
        }
        with pytest.raises(RuntimeError, match="lost"):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "model-router",
                    "stream": True,
                    "messages": [{"role": "user", "content": "rewrite this text"}],
                },
            ) as response:
                response.read()

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["status"] == "stream_interrupted"
    assert row["status_code"] == 200
    assert _FakeStream.exits == 1


def test_stream_generator_closes_context_when_client_disconnects():
    async def run() -> None:
        config = _config()
        stream_context = _FakeStream(_stream_response(b"data: first\n\n"))
        response = await stream_context.__aenter__()
        generator = _stream_response_bytes(
            stream_context,
            response,
            "req-1",
            "fast_local",
            config.backends["fast"],
            False,
            0.0,
            None,
            "rewrite this text",
            0.01,
            None,
            config,
            None,
        )

        assert await anext(generator) == b"data: first\n\n"
        await generator.aclose()

    _FakeStream.exits = 0
    asyncio.run(run())
    assert _FakeStream.exits == 1


def test_live_proxy_socket_disconnect_closes_upstream_and_keeps_logs_safe(
    tmp_path,
    caplog,
):
    caplog.set_level(logging.INFO, logger="model-router-proxy")
    state = _LiveStreamState()
    log_path = tmp_path / "routing-events.jsonl"
    prompt = "rewrite this text with raw_prompt_secret api_key=body-secret"
    payload = {
        "model": "model-router",
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }

    with _uvicorn_server(_live_streaming_upstream(state)) as upstream_url:
        proxy_config = _live_proxy_config(
            upstream_base_url=upstream_url,
            log_path=log_path,
        )
        with _uvicorn_server(create_app(proxy_config)) as proxy_url:
            response_bytes = _stream_and_disconnect(proxy_url, payload)

            assert b"HTTP/1.1 200 OK" in response_bytes
            assert b"data: first" in response_bytes
            assert state.request_seen.wait(1)
            assert state.first_chunk_sent.wait(1)
            assert state.stream_closed.wait(5)

            row = _wait_for_log_row(log_path)

    assert state.headers["authorization"] == "Bearer backend-secret"
    assert state.body["model"] == "fast-live-model"
    assert row["status"] == "stream_interrupted"
    assert row["backend"] == "fast"
    assert row["backend_model"] == "fast-live-model"
    assert row["status_code"] == 200
    assert "prompt" not in row
    assert "prompt_preview" not in row

    serialized_log = json.dumps(row)
    captured_logs = caplog.text
    for sensitive_value in (
        prompt,
        "raw_prompt_secret",
        "api_key=body-secret",
        "client-secret",
        "backend-secret",
        "messages",
    ):
        assert sensitive_value not in serialized_log
        assert sensitive_value not in captured_logs


def test_proxy_models_and_health_do_not_expose_secrets(monkeypatch):
    with _client(monkeypatch, _config(api_key="proxy-secret")) as client:
        headers = {"Authorization": "Bearer proxy-secret"}
        models = client.get("/v1/models", headers=headers)
        health = client.get("/health")

    assert models.json()["data"][0]["id"] == "model-router"
    serialized_health = json.dumps(health.json())
    assert "proxy-secret" not in serialized_health
    assert "deep-secret" not in serialized_health
    payload = health.json()
    assert not any(
        name.startswith("x-modelrouter") for name in health.headers.keys()
    )
    assert payload["backends"] == ["deep", "fast"]
    assert payload["status"] == "ok"
    assert payload["backend_health"]["deep"]["reachable"] is True
    assert payload["observability"]["enabled"] is False
    assert payload["verifier"]["mode"] == "off"


def test_proxy_health_reports_unreachable_backend(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        _FakeAsyncClient.responses = {"fast": [RuntimeError("offline")]}
        health = client.get("/health")

    payload = health.json()
    assert payload["status"] == "degraded"
    assert payload["backend_health"]["fast"]["reachable"] is False


def test_proxy_health_reports_missing_configured_model(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        _FakeAsyncClient.responses = {
            "fast": [
                httpx.Response(
                    200,
                    json={"data": [{"id": "other-model", "object": "model"}]},
                )
            ]
        }
        health = client.get("/health")

    payload = health.json()
    assert payload["status"] == "degraded"
    assert payload["backend_health"]["fast"]["reachable"] is True
    assert payload["backend_health"]["fast"]["ok"] is False
    assert "configured model 'fast-model' not listed" in payload["backend_health"][
        "fast"
    ]["detail"]


def test_proxy_health_treats_backend_4xx_as_degraded(monkeypatch):
    with _client(monkeypatch, _config()) as client:
        _FakeAsyncClient.responses = {
            "fast": [httpx.Response(401, json={"error": "nope"})]
        }
        health = client.get("/health")

    payload = health.json()
    assert payload["status"] == "degraded"
    assert payload["backend_health"]["fast"]["reachable"] is True
    assert payload["backend_health"]["fast"]["ok"] is False
    assert "HTTP 401" in payload["backend_health"]["fast"]["detail"]


def test_proxy_writes_privacy_safe_routing_event(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(monkeypatch, _config(log_path=log_path)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [
                    {"role": "user", "content": "rewrite this api_key=secret123"}
                ],
                "tools": [{"type": "function", "name": "private_tool"}],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    serialized = json.dumps(row)
    route_headers = _serialized_route_headers(response)
    assert response.status_code == 200
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert row["event_type"] == "routing_event"
    assert row["selected_engine"] == "fast_local"
    assert row["backend"] == "fast"
    assert row["backend_model"] == "fast-model"
    assert row["status"] == "forwarded"
    assert row["complexity_score"] >= 0
    assert "prompt_hash" in row
    assert "prompt" not in row
    assert "secret123" not in serialized
    assert "private_tool" not in serialized
    assert "secret123" not in route_headers
    assert "private_tool" not in route_headers
    assert "api_key" not in route_headers


def test_proxy_verifier_default_is_disabled(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(monkeypatch, _config(log_path=log_path)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert response.status_code == 200
    assert "verification_status" not in row
    assert [request["backend"] for request in _FakeAsyncClient.requests] == ["fast"]


def test_proxy_verifier_receipt_only_logs_qualification(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(
        monkeypatch,
        _config(
            log_path=log_path,
            verifier=ProxyVerifierConfig(mode="receipt-only"),
        ),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert response.status_code == 200
    assert row["verification_mode"] == "receipt-only"
    assert row["verification_status"] == "qualified"
    assert [request["backend"] for request in _FakeAsyncClient.requests] == ["fast"]


def test_proxy_verifier_sampled_calls_configured_backend(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(
        monkeypatch,
        _config(
            log_path=log_path,
            verifier=ProxyVerifierConfig(
                mode="sampled",
                backend="deep",
                sample_rate=1.0,
            ),
        ),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [
                    {"role": "user", "content": "rewrite this token=secret-value"}
                ],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    verifier_body = _FakeAsyncClient.requests[1]["body"]
    serialized_verifier_body = json.dumps(verifier_body)
    assert response.status_code == 200
    assert row["verification_mode"] == "sampled"
    assert row["verification_status"] == "passed"
    assert row["verification_backend"] == "deep"
    assert row["verification_status_code"] == 200
    assert [request["backend"] for request in _FakeAsyncClient.requests] == [
        "fast",
        "deep",
    ]
    assert "receipt" in serialized_verifier_body.lower()
    assert "secret-value" not in serialized_verifier_body
    assert "messages" not in json.dumps(row)


def test_proxy_verifier_fail_closed_has_clear_proxy_behavior(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(
        monkeypatch,
        _config(
            log_path=log_path,
            verifier=ProxyVerifierConfig(
                mode="sampled",
                backend="deep",
                sample_rate=1.0,
                failure_behavior="fail_closed",
            ),
        ),
    ) as client:
        _FakeAsyncClient.responses = {
            "fast": [_response(200)],
            "deep": [_response(503)],
        }
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "verification_failed"
    assert row["status"] == "verification_failed"
    assert row["verification_status"] == "failed"
    assert row["verification_backend"] == "deep"
    assert [request["backend"] for request in _FakeAsyncClient.requests] == [
        "fast",
        "deep",
    ]


def test_proxy_verifier_streaming_logs_skipped_without_buffering(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(
        monkeypatch,
        _config(
            log_path=log_path,
            verifier=ProxyVerifierConfig(
                mode="sampled",
                backend="deep",
                sample_rate=1.0,
            ),
        ),
    ) as client:
        _FakeAsyncClient.responses = {"fast": [_stream_response(b"data: chunk\n\n")]}
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "stream": True,
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        ) as response:
            body = response.read()

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert response.status_code == 200
    assert body == b"data: chunk\n\n"
    assert row["verification_status"] == "skipped_streaming"
    assert [request["backend"] for request in _FakeAsyncClient.requests] == ["fast"]


def test_proxy_logs_human_confirm_without_upstream_call(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(monkeypatch, _config(log_path=log_path)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "delete all files"}],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert response.status_code == 409
    _assert_route_headers(response, engine="human_confirm")
    assert row["status"] == "human_confirm"
    assert row["selected_engine"] == "human_confirm"
    assert _FakeAsyncClient.requests == []


def test_proxy_logs_upstream_failure(monkeypatch, tmp_path):
    log_path = tmp_path / "routing-events.jsonl"
    with _client(monkeypatch, _config(log_path=log_path)) as client:
        _FakeAsyncClient.responses = {
            "fast": [RuntimeError("down")],
            "deep": [RuntimeError("also down")],
        }
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-router",
                "messages": [{"role": "user", "content": "rewrite this text"}],
            },
        )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert response.status_code == 502
    _assert_route_headers(response, engine="fast_local", backend="fast")
    assert row["status"] == "upstream_request_failed"
    assert row["status_code"] == 502


def test_proxy_session_summary_is_privacy_safe_and_actionable(tmp_path):
    config = _config(api_key="client-secret", log_path=tmp_path / "routing-events.jsonl")
    config = replace(config, source_path=str(tmp_path / "routing_proxy.yaml"))
    stats = ProxySessionStats()
    stats.record(
        selected_engine="fast_local",
        status="forwarded",
        backend="fast",
    )
    stats.record(
        selected_engine="fast_local",
        status="stream_interrupted",
        backend="deep",
        fallback_used=True,
    )
    stats.record(
        selected_engine="human_confirm",
        status="human_confirm",
    )
    stats.record(
        selected_engine="fast_local",
        status="upstream_request_failed",
        backend="fast",
    )

    summary = _format_proxy_session_summary(stats, config)

    assert "ModelRouter session summary" in summary
    assert "Events: 4" in summary
    assert "Engines: fast_local=3, human_confirm=1" in summary
    assert "Backends: fast=2, deep=1" in summary
    assert "Statuses:" in summary
    assert "Fallbacks: 1" in summary
    assert "Interruptions: 1" in summary
    assert "Errors: 1" in summary
    assert "model-router telemetry summary" in summary
    assert str(tmp_path / "routing-events.jsonl") in summary
    assert str(tmp_path / "routing-feedback.jsonl") in summary
    for sensitive_value in (
        "client-secret",
        "backend-secret",
        "api_key",
        "messages",
        "raw prompt",
    ):
        assert sensitive_value not in summary


def test_core_import_path_does_not_import_proxy_dependencies():
    script = """
import sys
import model_router
router = model_router.ModelRouter.from_config(validate_availability=False)
router.route_fast('rewrite this text')
print(any(name.split('.')[0] in {'fastapi', 'httpx', 'uvicorn'} for name in sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "False"
