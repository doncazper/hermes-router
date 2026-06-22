import asyncio
from contextlib import contextmanager
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

from hermes.plugins.model_router.proxy import create_app, _stream_response_bytes
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyObservabilityConfig,
    ProxyServerConfig,
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

    async def post(self, path: str, *, content: bytes):
        self.requests.append(
            {
                "backend": self.backend_name,
                "path": path,
                "headers": self.headers,
                "body": json.loads(content.decode("utf-8")),
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
) -> RoutingProxyConfig:
    fallback_backends = {"fast": ("deep",)} if fallback else {}
    return RoutingProxyConfig(
        proxy=ProxyServerConfig(api_key=api_key),
        router_config=None,
        source_path="test.yaml",
        backends={
            "fast": ProxyBackendConfig(
                name="fast",
                base_url="http://fast.test/v1",
                model="fast-model",
                strip_tools=True,
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
    )


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
    assert response.headers["x-routed-engine"] == "fast_local"
    assert response.headers["x-routed-backend"] == "fast"
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "fast-model"


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
    assert response.headers["x-routed-backend"] == "fast"
    assert _FakeAsyncClient.requests[0]["path"] == "/responses"
    assert _FakeAsyncClient.requests[0]["stream"] is True
    assert _FakeAsyncClient.requests[0]["body"]["model"] == "fast-model"


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
    assert _FakeAsyncClient.requests == []


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
    assert [request["backend"] for request in _FakeAsyncClient.requests] == [
        "fast",
        "deep",
    ]


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
    assert _FakeAsyncClient.requests[0]["stream"] is True


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
    assert payload["backends"] == ["deep", "fast"]
    assert payload["status"] == "ok"
    assert payload["backend_health"]["deep"]["reachable"] is True
    assert payload["observability"]["enabled"] is False


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
    assert response.status_code == 200
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
    assert row["status"] == "upstream_request_failed"
    assert row["status_code"] == 502


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
