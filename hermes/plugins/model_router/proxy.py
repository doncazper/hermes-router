"""Optional OpenAI-compatible routing proxy."""

import argparse
import asyncio
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import time
import uuid
from typing import Any

from hermes.plugins.model_router.policy import FAIL_CLOSED_ENGINE, ModelRouter
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)
from hermes.plugins.model_router.routing_log import (
    RoutingLogWriter,
    build_routing_event,
)


LOG = logging.getLogger("model-router-proxy")
ROUTER_VERSION = "unknown"
try:
    ROUTER_VERSION = version("hermes-router")
except PackageNotFoundError:  # pragma: no cover - editable metadata is present in tests.
    pass


class ProxyDependencyError(RuntimeError):
    """Raised when optional proxy dependencies are not installed."""


class UpstreamRequestError(RuntimeError):
    """Raised when an upstream request cannot be completed."""


def create_app(config: RoutingProxyConfig):
    """Create the FastAPI app for a loaded proxy config."""

    try:
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ImportError as exc:  # pragma: no cover - exercised through CLI path.
        raise ProxyDependencyError(
            "routing proxy dependencies are missing; install with "
            'python -m pip install "hermes-router[proxy]"'
        ) from exc

    router = ModelRouter.from_config(
        config.router_config,
        validate_availability=False,
    )
    event_writer = (
        RoutingLogWriter(
            config.observability.log_path,
            max_bytes=config.observability.max_bytes,
            backups=config.observability.backups,
        )
        if config.observability.enabled
        else None
    )
    clients: dict[str, httpx.AsyncClient] = {}

    async def _lifespan(_: FastAPI):
        for backend in config.backends.values():
            headers = _backend_headers(backend)
            clients[backend.name] = httpx.AsyncClient(
                base_url=backend.base_url,
                headers=headers,
                timeout=backend.timeout_seconds,
            )
        LOG.info(
            "routing proxy ready host=%s port=%s backends=%s",
            config.proxy.host,
            config.proxy.port,
            ",".join(sorted(config.backends)),
        )
        yield
        await asyncio.gather(*(client.aclose() for client in clients.values()))

    app = FastAPI(
        title="model-router-proxy",
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        auth_response = _authorize_request(request, config)
        if auth_response is not None:
            return auth_response

        request_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Request body must be valid JSON.",
                        "type": "invalid_request_error",
                    }
                },
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Request body must be a JSON object.",
                        "type": "invalid_request_error",
                    }
                },
            )

        messages = body.get("messages", [])
        prompt = _extract_recent_user_text(messages)
        route_started = time.perf_counter()
        engine = router.route_fast(prompt) if prompt else FAIL_CLOSED_ENGINE
        route_latency_ms = (time.perf_counter() - route_started) * 1000
        diagnostic_decision = None
        diagnostic_latency_ms = None
        if event_writer is not None:
            diagnostic_started = time.perf_counter()
            diagnostic_decision = router.route(prompt, include_alternatives=False)
            diagnostic_latency_ms = (time.perf_counter() - diagnostic_started) * 1000
        if engine == FAIL_CLOSED_ENGINE:
            _write_proxy_event(
                event_writer,
                request_id=request_id,
                prompt=prompt,
                selected_engine=engine,
                status="human_confirm",
                route_latency_ms=route_latency_ms,
                diagnostic_latency_ms=diagnostic_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                config=config,
                decision=diagnostic_decision,
                status_code=409,
            )
            return JSONResponse(
                status_code=409,
                headers={"X-Routed-Engine": engine, "X-Request-Id": request_id},
                content={
                    "error": {
                        "message": "Human confirmation is required before dispatch.",
                        "type": "human_confirmation_required",
                    },
                    "selected_engine": engine,
                },
            )

        backend = config.backend_for_engine(engine)
        if backend is None:
            _write_proxy_event(
                event_writer,
                request_id=request_id,
                prompt=prompt,
                selected_engine=engine,
                status="routing_backend_missing",
                route_latency_ms=route_latency_ms,
                diagnostic_latency_ms=diagnostic_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                config=config,
                decision=diagnostic_decision,
                status_code=502,
            )
            return JSONResponse(
                status_code=502,
                headers={"X-Routed-Engine": engine, "X-Request-Id": request_id},
                content={
                    "error": {
                        "message": f"No backend configured for selected engine {engine}.",
                        "type": "routing_backend_missing",
                    },
                    "selected_engine": engine,
                },
            )

        payload = _payload_for_backend(body, backend)
        if bool(body.get("stream", False)):
            try:
                (
                    stream_context,
                    upstream_response,
                    used_backend,
                    fallback_used,
                ) = await _open_stream_with_fallbacks(
                    clients,
                    backend,
                    config.fallback_chain_for_backend(backend.name),
                    payload,
                    body,
                )
            except UpstreamRequestError as exc:
                _write_proxy_event(
                    event_writer,
                    request_id=request_id,
                    prompt=prompt,
                    selected_engine=engine,
                    status="upstream_request_failed",
                    route_latency_ms=route_latency_ms,
                    diagnostic_latency_ms=diagnostic_latency_ms,
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    config=config,
                    decision=diagnostic_decision,
                    backend=backend.name,
                    backend_model=backend.model,
                    status_code=502,
                )
                return JSONResponse(
                    status_code=502,
                    headers={
                        "X-Routed-Engine": engine,
                        "X-Request-Id": request_id,
                    },
                    content={
                        "error": {
                            "message": f"Upstream backend request failed: {exc}",
                            "type": "upstream_request_failed",
                        },
                        "selected_engine": engine,
                    },
                )
            return StreamingResponse(
                _stream_response_bytes(
                    stream_context,
                    upstream_response,
                    request_id,
                    engine,
                    used_backend,
                    fallback_used,
                    started,
                    event_writer,
                    prompt,
                    route_latency_ms,
                    diagnostic_latency_ms,
                    config,
                    diagnostic_decision,
                ),
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get(
                    "content-type",
                    "text/event-stream",
                ),
                headers={
                    "X-Routed-Engine": engine,
                    "X-Routed-Backend": used_backend.name,
                    "X-Request-Id": request_id,
                },
            )

        upstream_started = time.perf_counter()
        try:
            response, used_backend, fallback_used = await _post_with_fallbacks(
                clients,
                backend,
                config.fallback_chain_for_backend(backend.name),
                payload,
                body,
            )
        except UpstreamRequestError as exc:
            _write_proxy_event(
                event_writer,
                request_id=request_id,
                prompt=prompt,
                selected_engine=engine,
                status="upstream_request_failed",
                route_latency_ms=route_latency_ms,
                diagnostic_latency_ms=diagnostic_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                config=config,
                decision=diagnostic_decision,
                backend=backend.name,
                backend_model=backend.model,
                status_code=502,
            )
            return JSONResponse(
                status_code=502,
                headers={"X-Routed-Engine": engine, "X-Request-Id": request_id},
                content={
                    "error": {
                        "message": f"Upstream backend request failed: {exc}",
                        "type": "upstream_request_failed",
                    },
                    "selected_engine": engine,
                },
            )
        upstream_latency_ms = (time.perf_counter() - upstream_started) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000
        _write_proxy_event(
            event_writer,
            request_id=request_id,
            prompt=prompt,
            selected_engine=engine,
            status="forwarded",
            route_latency_ms=route_latency_ms,
            diagnostic_latency_ms=diagnostic_latency_ms,
            upstream_latency_ms=upstream_latency_ms,
            total_latency_ms=elapsed_ms,
            config=config,
            decision=diagnostic_decision,
            backend=used_backend.name,
            backend_model=used_backend.model,
            fallback_used=fallback_used,
            status_code=response.status_code,
        )
        LOG.info(
            "request=%s engine=%s backend=%s status=%s fallback=%s latency_ms=%.2f",
            request_id,
            engine,
            used_backend.name,
            response.status_code,
            fallback_used,
            elapsed_ms,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
            headers={
                "X-Routed-Engine": engine,
                "X-Routed-Backend": used_backend.name,
                "X-Request-Id": request_id,
            },
        )

    @app.get("/v1/models")
    async def list_models(request: Request):
        auth_response = _authorize_request(request, config)
        if auth_response is not None:
            return auth_response
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "model-router",
                }
                for model_id in config.proxy.model_ids
            ],
        }

    @app.get("/health")
    async def health():
        backend_items = sorted(clients.items())
        backend_results = await asyncio.gather(
            *(
                _check_backend_health(
                    client,
                    expected_model=config.backends[name].model,
                    timeout_seconds=config.health.backend_timeout_seconds,
                )
                for name, client in backend_items
            )
        )
        backend_health = {
            name: result
            for (name, _client), result in zip(
                backend_items,
                backend_results,
                strict=True,
            )
        }
        all_ok = all(result["ok"] for result in backend_health.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "backends": sorted(config.backends),
            "backend_health": backend_health,
            "engine_backends": dict(sorted(config.engine_backends.items())),
            "observability": {
                "enabled": config.observability.enabled,
                "prompt_capture": config.observability.prompt_capture,
                "max_bytes": config.observability.max_bytes,
                "backups": config.observability.backups,
            },
            "router_config": config.router_config or "default",
            "proxy_config": config.source_path,
        }

    return app


def _authorize_request(request: Any, config: RoutingProxyConfig):
    expected = config.proxy.resolved_api_key
    if not expected:
        return None
    header = request.headers.get("authorization", "")
    if header == f"Bearer {expected}":
        return None
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Missing or invalid bearer token.",
                "type": "authentication_error",
            }
        },
    )


def _backend_headers(backend: ProxyBackendConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = backend.resolved_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _check_backend_health(
    client: Any,
    *,
    expected_model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        response = await client.get("/models", timeout=timeout_seconds)
    except Exception as exc:
        return {
            "reachable": False,
            "ok": False,
            "status_code": None,
            "detail": f"request failed: {exc.__class__.__name__}",
        }
    status_code = int(response.status_code)
    model_ok, model_detail = _backend_model_detail(expected_model, response.content)
    status_ok = 200 <= status_code < 300
    ok = status_ok and model_ok
    detail = f"HTTP {status_code}"
    if not status_ok:
        detail += "; backend status is not successful"
    if model_detail:
        detail += f"; {model_detail}"
    return {
        "reachable": True,
        "ok": ok,
        "status_code": status_code,
        "detail": detail,
    }


def _backend_model_detail(model: str, body: bytes) -> tuple[bool, str | None]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True, "model list unavailable"
    if not isinstance(payload, dict):
        return True, "model list unavailable"
    data = payload.get("data")
    if not isinstance(data, list):
        return True, "model list unavailable"
    model_ids = {
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model in model_ids:
        return True, f"configured model {model!r} listed"
    return False, f"configured model {model!r} not listed"


def _extract_recent_user_text(messages: Any, *, lookback: int = 3) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    seen = 0
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        parts.append(_message_content_text(message.get("content")))
        seen += 1
        if seen >= lookback:
            break
    return " ".join(reversed([part for part in parts if part]))


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return " ".join(text_parts)
    return ""


def _payload_for_backend(body: dict[str, Any], backend: ProxyBackendConfig) -> bytes:
    payload = dict(body)
    payload["model"] = backend.model
    if backend.strip_tools:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        payload.pop("functions", None)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def _post_with_fallbacks(
    clients: dict[str, Any],
    primary: ProxyBackendConfig,
    fallbacks: tuple[ProxyBackendConfig, ...],
    primary_payload: bytes,
    original_body: dict[str, Any],
) -> tuple[Any, ProxyBackendConfig, bool]:
    attempted = (primary, *fallbacks)
    last_error: Exception | None = None
    for index, backend in enumerate(attempted):
        payload = primary_payload if index == 0 else _payload_for_backend(original_body, backend)
        try:
            response = await clients[backend.name].post(
                "/chat/completions",
                content=payload,
            )
        except Exception as exc:
            last_error = exc
            continue
        if response.status_code < 500 or index == len(attempted) - 1:
            return response, backend, index > 0
    raise UpstreamRequestError(str(last_error) if last_error else "upstream failed")


async def _open_stream_with_fallbacks(
    clients: dict[str, Any],
    primary: ProxyBackendConfig,
    fallbacks: tuple[ProxyBackendConfig, ...],
    primary_payload: bytes,
    original_body: dict[str, Any],
) -> tuple[Any, Any, ProxyBackendConfig, bool]:
    attempted = (primary, *fallbacks)
    last_error: Exception | None = None
    for index, backend in enumerate(attempted):
        payload = primary_payload if index == 0 else _payload_for_backend(original_body, backend)
        try:
            stream_context = clients[backend.name].stream(
                "POST",
                "/chat/completions",
                content=payload,
            )
            response = await stream_context.__aenter__()
        except Exception as exc:
            last_error = exc
            continue
        if response.status_code >= 500 and index < len(attempted) - 1:
            await stream_context.__aexit__(None, None, None)
            continue
        return stream_context, response, backend, index > 0
    raise UpstreamRequestError(str(last_error) if last_error else "upstream failed")


async def _stream_response_bytes(
    stream_context: Any,
    response: Any,
    request_id: str,
    engine: str,
    backend: ProxyBackendConfig,
    fallback_used: bool,
    started: float,
    event_writer: RoutingLogWriter | None,
    prompt: str,
    route_latency_ms: float,
    diagnostic_latency_ms: float | None,
    config: RoutingProxyConfig,
    decision: Any,
):
    status = "forwarded"
    exc_info: tuple[type[BaseException] | None, BaseException | None, Any] = (
        None,
        None,
        None,
    )
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    except asyncio.CancelledError as exc:
        status = "stream_interrupted"
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    except Exception as exc:
        status = "stream_interrupted"
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            await stream_context.__aexit__(*exc_info)
        except Exception as exc:
            LOG.warning(
                "request=%s stream cleanup failed error=%s",
                request_id,
                exc.__class__.__name__,
            )
        _write_proxy_event(
            event_writer,
            request_id=request_id,
            prompt=prompt,
            selected_engine=engine,
            status=status,
            route_latency_ms=route_latency_ms,
            diagnostic_latency_ms=diagnostic_latency_ms,
            upstream_latency_ms=max(0.0, elapsed_ms - route_latency_ms),
            total_latency_ms=elapsed_ms,
            config=config,
            decision=decision,
            backend=backend.name,
            backend_model=backend.model,
            fallback_used=fallback_used,
            status_code=response.status_code,
        )
        LOG.info(
            "request=%s engine=%s backend=%s status=%s stream_status=%s fallback=%s latency_ms=%.2f",
            request_id,
            engine,
            backend.name,
            response.status_code,
            status,
            fallback_used,
            (time.perf_counter() - started) * 1000,
        )


def _write_proxy_event(
    writer: RoutingLogWriter | None,
    *,
    request_id: str,
    prompt: str,
    selected_engine: str,
    status: str,
    route_latency_ms: float,
    diagnostic_latency_ms: float | None,
    total_latency_ms: float,
    config: RoutingProxyConfig,
    decision: Any,
    upstream_latency_ms: float | None = None,
    backend: str | None = None,
    backend_model: str | None = None,
    fallback_used: bool = False,
    status_code: int | None = None,
) -> None:
    if writer is None:
        return
    writer.write(
        build_routing_event(
            request_id=request_id,
            route_api="route_fast",
            selected_engine=selected_engine,
            status=status,
            prompt=prompt,
            route_latency_ms=route_latency_ms,
            diagnostic_latency_ms=diagnostic_latency_ms,
            upstream_latency_ms=upstream_latency_ms,
            total_latency_ms=total_latency_ms,
            config_source=config.router_config or "default",
            router_version=ROUTER_VERSION,
            fallback_used=fallback_used,
            backend=backend,
            backend_model=backend_model,
            status_code=status_code,
            decision=decision,
            prompt_capture=config.observability.prompt_capture,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-router-proxy",
        description="Run the optional OpenAI-compatible model routing proxy.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to routing proxy YAML config. Uses packaged example when omitted.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override proxy bind host from config.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override proxy bind port from config.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="Uvicorn log level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_proxy_config(args.config)
        app = create_app(config)
        import uvicorn
    except (ProxyConfigError, ProxyDependencyError) as exc:
        parser.error(str(exc))

    host = args.host or config.proxy.host
    port = args.port or config.proxy.port
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
