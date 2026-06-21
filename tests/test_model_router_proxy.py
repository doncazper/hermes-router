import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from hermes.plugins.model_router.proxy import create_app
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyObservabilityConfig,
    ProxyServerConfig,
    RoutingProxyConfig,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeStream:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.status_code = response.status_code
        self.headers = response.headers

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def aiter_bytes(self):
        yield self.response.content


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


def _response(status_code: int, payload: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload
        or {
            "id": f"status-{status_code}",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
    )


def _stream_response(content: bytes = b"data: hello\n\n") -> httpx.Response:
    return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})


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
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    return TestClient(create_app(config))


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
                "functions": [{"name": "legacy"}],
            },
        )

    fast_body = _FakeAsyncClient.requests[0]["body"]
    deep_body = _FakeAsyncClient.requests[1]["body"]
    assert "tools" not in fast_body
    assert "tool_choice" not in fast_body
    assert "functions" not in fast_body
    assert deep_body["tools"] == [{"type": "function"}]
    assert deep_body["tool_choice"] == "auto"
    assert deep_body["functions"] == [{"name": "legacy"}]


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
