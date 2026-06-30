"""Optional OpenAI-compatible routing proxy."""

import argparse
import asyncio
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import time
import uuid
from typing import Any, Callable

from hermes.plugins.model_router.policy import FAIL_CLOSED_ENGINE, ModelRouter
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyConfigError,
    RoutingProxyConfig,
    load_proxy_config,
)
from hermes.plugins.model_router.proxy_runtime import (
    ManagedRuntimeManager,
    RuntimeStartError,
)
from hermes.plugins.model_router.routing_log import (
    DEFAULT_FEEDBACK_PATH,
    RoutingLogWriter,
    build_routing_event,
    redact_text,
)
from hermes.plugins.model_router.receipts import decision_to_receipt


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


class UnsupportedUpstreamAPIError(RuntimeError):
    """Raised when a backend cannot support the requested upstream API."""


class ManualRoutingError(RuntimeError):
    """Raised when manual routing cannot select a safe backend/model."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        status: str,
        status_code: int,
        backend: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status = status
        self.status_code = status_code
        self.backend = backend
        self.model = model


@dataclass(frozen=True)
class ManualRoutingSelection:
    backend: ProxyBackendConfig
    model: str


@dataclass
class ProxySessionStats:
    """Privacy-safe per-process route counters for shutdown summaries."""

    total_events: int = 0
    engine_counts: Counter[str] = field(default_factory=Counter)
    backend_counts: Counter[str] = field(default_factory=Counter)
    status_counts: Counter[str] = field(default_factory=Counter)
    fallback_count: int = 0
    interruption_count: int = 0
    error_count: int = 0

    def record(
        self,
        *,
        selected_engine: str,
        status: str,
        backend: str | None = None,
        fallback_used: bool = False,
    ) -> None:
        self.total_events += 1
        self.engine_counts.update([selected_engine])
        self.status_counts.update([status])
        if backend:
            self.backend_counts.update([backend])
        if fallback_used:
            self.fallback_count += 1
        if status == "stream_interrupted":
            self.interruption_count += 1
        if status in {
            "backend_policy_rejected",
            "routing_backend_missing",
            "runtime_start_failed",
            "upstream_api_unsupported",
            "upstream_request_failed",
            "verification_failed",
        }:
            self.error_count += 1


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
    session_stats = ProxySessionStats()
    clients: dict[str, httpx.AsyncClient] = {}

    async def _runtime_readiness_probe(url: str, timeout: float) -> bool:
        async with httpx.AsyncClient(timeout=timeout) as readiness_client:
            response = await readiness_client.get(url)
        return 200 <= int(response.status_code) < 500

    runtime_manager = ManagedRuntimeManager(
        config.backends,
        readiness_probe=_runtime_readiness_probe,
    )

    async def _lifespan(_: FastAPI):
        idle_reaper: asyncio.Task | None = None
        for backend in config.backends.values():
            headers = _backend_headers(backend)
            clients[backend.name] = httpx.AsyncClient(
                base_url=backend.base_url,
                headers=headers,
                timeout=backend.timeout_seconds,
            )
        LOG.info(
            "routing proxy ready host=%s port=%s profile=%s backends=%s",
            config.proxy.host,
            config.proxy.port,
            config.proxy.routing_profile,
            ",".join(sorted(config.backends)),
        )
        if runtime_manager.has_managed_backends:
            idle_reaper = asyncio.create_task(runtime_manager.reap_idle_forever())
        try:
            yield
        finally:
            if idle_reaper is not None:
                idle_reaper.cancel()
                with suppress(asyncio.CancelledError):
                    await idle_reaper
            await runtime_manager.stop_all()
            close_results = await asyncio.gather(
                *(client.aclose() for client in clients.values()),
                return_exceptions=True,
            )
            for result in close_results:
                if isinstance(result, Exception):
                    LOG.warning(
                        "backend client cleanup failed error=%s",
                        result.__class__.__name__,
                    )
            _print_proxy_session_summary(session_stats, config)

    app = FastAPI(
        title="model-router-proxy",
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
    )

    async def _forward_routed_request(
        request: Request,
        *,
        upstream_path: str,
        prompt_from_body: Callable[[dict[str, Any]], str],
    ):
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

        route_api = "route_fast"
        prompt = ""
        diagnostic_decision = None
        diagnostic_latency_ms = None
        selected_model: str | None = None
        if config.proxy.routing_mode == "manual":
            route_api = "manual"
            route_started = time.perf_counter()
            engine = "manual"
            route_latency_ms = (time.perf_counter() - route_started) * 1000
            try:
                manual_selection = _manual_routing_selection(config, body)
            except ManualRoutingError as exc:
                _write_proxy_event(
                    event_writer,
                    request_id=request_id,
                    prompt="",
                    selected_engine=engine,
                    status=exc.status,
                    route_latency_ms=route_latency_ms,
                    diagnostic_latency_ms=None,
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    config=config,
                    decision=None,
                    backend=exc.backend,
                    backend_model=exc.model,
                    fallback_used=False,
                    status_code=exc.status_code,
                    stats=session_stats,
                    route_api=route_api,
                )
                return JSONResponse(
                    status_code=exc.status_code,
                    headers=_route_headers(
                        request_id=request_id,
                        engine=engine,
                        backend=exc.backend,
                        model=exc.model,
                        fallback_used=False,
                        profile=config.proxy.routing_profile,
                        routing_mode=config.proxy.routing_mode,
                        decision_layer_enabled=False,
                        route_api=route_api,
                    ),
                    content={
                        "error": {
                            "message": str(exc),
                            "type": exc.error_type,
                        },
                        "selected_engine": engine,
                    },
                )
            backend = manual_selection.backend
            selected_model = manual_selection.model
        else:
            prompt = prompt_from_body(body)
            route_hints = {"profile": config.proxy.routing_profile}
            route_started = time.perf_counter()
            engine = (
                router.route_fast(prompt, hints=route_hints)
                if prompt
                else FAIL_CLOSED_ENGINE
            )
            route_latency_ms = (time.perf_counter() - route_started) * 1000
            if event_writer is not None:
                diagnostic_started = time.perf_counter()
                diagnostic_decision = router.route(
                    prompt,
                    hints=route_hints,
                    include_alternatives=False,
                )
                diagnostic_latency_ms = (time.perf_counter() - diagnostic_started) * 1000

        def route_headers(
            *,
            engine: str,
            backend: str | None = None,
            model: str | None = None,
            fallback_used: bool | None = None,
        ) -> dict[str, str]:
            return _route_headers(
                request_id=request_id,
                engine=engine,
                backend=backend,
                model=model,
                fallback_used=fallback_used,
                profile=config.proxy.routing_profile,
                routing_mode=config.proxy.routing_mode,
                decision_layer_enabled=config.proxy.routing_mode == "decision",
                route_api=route_api,
            )

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
                stats=session_stats,
                route_api=route_api,
            )
            return JSONResponse(
                status_code=409,
                headers=route_headers(
                    engine=engine,
                    fallback_used=False,
                ),
                content={
                    "error": {
                        "message": "Human confirmation is required before dispatch.",
                        "type": "human_confirmation_required",
                    },
                    "selected_engine": engine,
                },
            )

        if config.proxy.routing_mode == "decision":
            backend_name = config.engine_backends.get(engine)
            backend_policy_reason = config.backend_policy_rejection_reason(backend_name)
            if backend_policy_reason is not None:
                _write_proxy_event(
                    event_writer,
                    request_id=request_id,
                    prompt=prompt,
                    selected_engine=engine,
                    status="backend_policy_rejected",
                    route_latency_ms=route_latency_ms,
                    diagnostic_latency_ms=diagnostic_latency_ms,
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    config=config,
                    decision=diagnostic_decision,
                    backend=backend_name,
                    status_code=502,
                    stats=session_stats,
                    route_api=route_api,
                )
                return JSONResponse(
                    status_code=502,
                    headers=route_headers(
                        engine=engine,
                        backend=backend_name,
                        fallback_used=False,
                    ),
                    content={
                        "error": {
                            "message": backend_policy_reason,
                            "type": "backend_policy_rejected",
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
                    stats=session_stats,
                    route_api=route_api,
                )
                return JSONResponse(
                    status_code=502,
                    headers=route_headers(
                        engine=engine,
                        fallback_used=False,
                    ),
                    content={
                        "error": {
                            "message": f"No backend configured for selected engine {engine}.",
                            "type": "routing_backend_missing",
                        },
                        "selected_engine": engine,
                    },
                )
            selected_model = backend.model

        payload = _payload_for_backend(body, backend, model=selected_model)
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
                    upstream_path,
                    payload,
                    body,
                    runtime_manager,
                )
            except RuntimeStartError as exc:
                _write_proxy_event(
                    event_writer,
                    request_id=request_id,
                    prompt=prompt,
                    selected_engine=engine,
                    status="runtime_start_failed",
                    route_latency_ms=route_latency_ms,
                    diagnostic_latency_ms=diagnostic_latency_ms,
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    config=config,
                    decision=diagnostic_decision,
                    backend=backend.name,
                    backend_model=selected_model,
                    status_code=502,
                    stats=session_stats,
                    route_api=route_api,
                )
                return JSONResponse(
                    status_code=502,
                    headers=route_headers(
                        engine=engine,
                        backend=backend.name,
                        model=selected_model,
                        fallback_used=False,
                    ),
                    content={
                        "error": {
                            "message": str(exc),
                            "type": "runtime_start_failed",
                        },
                        "selected_engine": engine,
                    },
                )
            except UnsupportedUpstreamAPIError as exc:
                _write_proxy_event(
                    event_writer,
                    request_id=request_id,
                    prompt=prompt,
                    selected_engine=engine,
                    status="upstream_api_unsupported",
                    route_latency_ms=route_latency_ms,
                    diagnostic_latency_ms=diagnostic_latency_ms,
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    config=config,
                    decision=diagnostic_decision,
                    backend=backend.name,
                    backend_model=selected_model,
                    status_code=502,
                    stats=session_stats,
                    route_api=route_api,
                )
                return JSONResponse(
                    status_code=502,
                    headers=route_headers(
                        engine=engine,
                        backend=backend.name,
                        model=selected_model,
                        fallback_used=False,
                    ),
                    content={
                        "error": {
                            "message": str(exc),
                            "type": "upstream_api_unsupported",
                        },
                        "selected_engine": engine,
                    },
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
                    backend_model=selected_model,
                    status_code=502,
                    stats=session_stats,
                    route_api=route_api,
                )
                return JSONResponse(
                    status_code=502,
                    headers=route_headers(
                        engine=engine,
                        backend=backend.name,
                        model=selected_model,
                        fallback_used=False,
                    ),
                    content={
                        "error": {
                            "message": f"Upstream backend request failed: {exc}",
                            "type": "upstream_request_failed",
                        },
                        "selected_engine": engine,
                    },
                )
            used_model = _model_for_used_backend(backend, selected_model, used_backend)
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
                    session_stats,
                    runtime_manager,
                    _streaming_verification_result(config, diagnostic_decision),
                    selected_model=used_model,
                    route_api=route_api,
                ),
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get(
                    "content-type",
                    "text/event-stream",
                ),
                headers=route_headers(
                    engine=engine,
                    backend=used_backend.name,
                    model=used_model,
                    fallback_used=fallback_used,
                ),
            )

        upstream_started = time.perf_counter()
        try:
            response, used_backend, fallback_used = await _post_with_fallbacks(
                clients,
                backend,
                config.fallback_chain_for_backend(backend.name),
                upstream_path,
                payload,
                body,
                runtime_manager,
            )
        except RuntimeStartError as exc:
            _write_proxy_event(
                event_writer,
                request_id=request_id,
                prompt=prompt,
                selected_engine=engine,
                status="runtime_start_failed",
                route_latency_ms=route_latency_ms,
                diagnostic_latency_ms=diagnostic_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                config=config,
                decision=diagnostic_decision,
                backend=backend.name,
                backend_model=selected_model,
                status_code=502,
                stats=session_stats,
                route_api=route_api,
            )
            return JSONResponse(
                status_code=502,
                headers=route_headers(
                    engine=engine,
                    backend=backend.name,
                    model=selected_model,
                    fallback_used=False,
                ),
                content={
                    "error": {
                        "message": str(exc),
                        "type": "runtime_start_failed",
                    },
                    "selected_engine": engine,
                },
            )
        except UnsupportedUpstreamAPIError as exc:
            _write_proxy_event(
                event_writer,
                request_id=request_id,
                prompt=prompt,
                selected_engine=engine,
                status="upstream_api_unsupported",
                route_latency_ms=route_latency_ms,
                diagnostic_latency_ms=diagnostic_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                config=config,
                decision=diagnostic_decision,
                backend=backend.name,
                backend_model=selected_model,
                status_code=502,
                stats=session_stats,
                route_api=route_api,
            )
            return JSONResponse(
                status_code=502,
                headers=route_headers(
                    engine=engine,
                    backend=backend.name,
                    model=selected_model,
                    fallback_used=False,
                ),
                content={
                    "error": {
                        "message": str(exc),
                        "type": "upstream_api_unsupported",
                    },
                    "selected_engine": engine,
                },
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
                backend_model=selected_model,
                status_code=502,
                stats=session_stats,
                route_api=route_api,
            )
            return JSONResponse(
                status_code=502,
                headers=route_headers(
                    engine=engine,
                    backend=backend.name,
                    model=selected_model,
                    fallback_used=False,
                ),
                content={
                    "error": {
                        "message": f"Upstream backend request failed: {exc}",
                        "type": "upstream_request_failed",
                    },
                    "selected_engine": engine,
                },
            )
        upstream_latency_ms = (time.perf_counter() - upstream_started) * 1000
        used_model = _model_for_used_backend(backend, selected_model, used_backend)
        verification = await _verify_response_if_configured(
            config,
            clients,
            runtime_manager,
            request_id=request_id,
            selected_engine=engine,
            decision=diagnostic_decision,
            upstream_response=response,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if (
            verification
            and verification.get("status") in {"failed", "error"}
            and config.verifier.failure_behavior == "fail_closed"
        ):
            _write_proxy_event(
                event_writer,
                request_id=request_id,
                prompt=prompt,
                selected_engine=engine,
                status="verification_failed",
                route_latency_ms=route_latency_ms,
                diagnostic_latency_ms=diagnostic_latency_ms,
                upstream_latency_ms=upstream_latency_ms,
                total_latency_ms=elapsed_ms,
                config=config,
                decision=diagnostic_decision,
                backend=used_backend.name,
                backend_model=used_model,
                fallback_used=fallback_used,
                status_code=502,
                stats=session_stats,
                verification=verification,
                route_api=route_api,
            )
            runtime_manager.touch(used_backend.name)
            return JSONResponse(
                status_code=502,
                headers=route_headers(
                    engine=engine,
                    backend=used_backend.name,
                    model=used_model,
                    fallback_used=fallback_used,
                ),
                content={
                    "error": {
                        "message": "Verifier rejected or failed the routed response.",
                        "type": "verification_failed",
                    },
                    "selected_engine": engine,
                },
            )
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
            backend_model=used_model,
            fallback_used=fallback_used,
            status_code=response.status_code,
            stats=session_stats,
            verification=verification,
            route_api=route_api,
        )
        runtime_manager.touch(used_backend.name)
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
            headers=route_headers(
                engine=engine,
                backend=used_backend.name,
                model=used_model,
                fallback_used=fallback_used,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _forward_routed_request(
            request,
            upstream_path="/chat/completions",
            prompt_from_body=lambda body: _extract_recent_user_text(
                body.get("messages", [])
            ),
        )

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await _forward_routed_request(
            request,
            upstream_path="/responses",
            prompt_from_body=lambda body: _extract_responses_input_text(
                body.get("input")
            ),
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
            "routing_profile": config.proxy.routing_profile,
            "routing_mode": config.proxy.routing_mode,
            "decision_layer_enabled": config.proxy.routing_mode == "decision",
            "default_backend": config.proxy.default_backend,
            "default_model": config.proxy.default_model,
            "respect_client_model": config.proxy.respect_client_model,
            "unknown_model_behavior": config.proxy.unknown_model_behavior,
            "safety_gate_mode": config.proxy.safety_gate_mode,
            "backend_policy": config.backend_policy.to_dict(),
            "verifier": config.verifier.to_dict(),
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


def _manual_routing_selection(
    config: RoutingProxyConfig,
    body: dict[str, Any],
) -> ManualRoutingSelection:
    backend_name = config.proxy.default_backend
    model = config.proxy.default_model
    if not backend_name or not model:
        raise ManualRoutingError(
            "Manual routing mode requires proxy.default_backend and proxy.default_model.",
            error_type="manual_config_invalid",
            status="manual_config_invalid",
            status_code=500,
            backend=backend_name,
            model=model,
        )
    backend_policy_reason = config.backend_policy_rejection_reason(backend_name)
    if backend_policy_reason is not None:
        raise ManualRoutingError(
            backend_policy_reason,
            error_type="backend_policy_rejected",
            status="backend_policy_rejected",
            status_code=502,
            backend=backend_name,
            model=model,
        )
    backend = config.backends.get(backend_name)
    if backend is None:
        raise ManualRoutingError(
            f"Manual routing backend {backend_name!r} is not configured.",
            error_type="manual_backend_missing",
            status="manual_backend_missing",
            status_code=502,
            backend=backend_name,
            model=model,
        )
    client_model = body.get("model")
    if (
        config.proxy.respect_client_model
        and isinstance(client_model, str)
        and client_model.strip()
    ):
        requested_model = client_model.strip()
        if _manual_model_allowed(
            backend=backend,
            default_model=config.proxy.default_model,
            requested_model=requested_model,
        ):
            model = requested_model
        elif config.proxy.unknown_model_behavior == "reject_404":
            raise ManualRoutingError(
                f"Model {requested_model!r} is not allowed by manual routing config.",
                error_type="unknown_model",
                status="unknown_model",
                status_code=404,
                backend=backend_name,
                model=model,
            )
    return ManualRoutingSelection(backend=backend, model=model)


def _manual_model_allowed(
    *,
    backend: ProxyBackendConfig,
    default_model: str | None,
    requested_model: str,
) -> bool:
    allowed_models = {
        item
        for item in (
            default_model,
            backend.model,
        )
        if isinstance(item, str) and item.strip()
    }
    return requested_model in allowed_models


def _model_for_used_backend(
    primary_backend: ProxyBackendConfig,
    primary_model: str | None,
    used_backend: ProxyBackendConfig,
) -> str:
    if used_backend.name == primary_backend.name and primary_model:
        return primary_model
    return used_backend.model


def _route_headers(
    *,
    request_id: str,
    engine: str,
    backend: str | None = None,
    model: str | None = None,
    fallback_used: bool | None = None,
    profile: str = "balanced",
    routing_mode: str = "decision",
    decision_layer_enabled: bool = True,
    route_api: str = "route_fast",
) -> dict[str, str]:
    headers = {
        "X-ModelRouter-Request-ID": _safe_header_value(request_id),
        "X-ModelRouter-Engine": _safe_header_value(engine),
        "X-ModelRouter-Mode": _safe_header_value(routing_mode),
        "X-ModelRouter-Decision-Layer": (
            "on" if decision_layer_enabled else "off"
        ),
        "X-ModelRouter-Profile": _safe_header_value(profile),
        "X-ModelRouter-Route-API": _safe_header_value(route_api),
        # Kept for compatibility with the earlier proxy header names.
        "X-Request-Id": _safe_header_value(request_id),
        "X-Routed-Engine": _safe_header_value(engine),
    }
    if backend:
        safe_backend = _safe_header_value(backend)
        headers["X-ModelRouter-Backend"] = safe_backend
        headers["X-Routed-Backend"] = safe_backend
    if model:
        headers["X-ModelRouter-Model"] = _safe_header_value(model)
    if fallback_used is not None:
        headers["X-ModelRouter-Fallback"] = "true" if fallback_used else "false"
    return headers


def _safe_header_value(value: str) -> str:
    return "".join(
        character if 32 <= ord(character) < 127 and character not in "\r\n" else "?"
        for character in value
    )[:128]


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


def _extract_responses_input_text(input_value: Any, *, lookback: int = 3) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, dict):
        return _responses_content_text(
            input_value.get("content", input_value.get("text", ""))
        )
    if not isinstance(input_value, list):
        return ""

    parts: list[str] = []
    seen = 0
    for item in reversed(input_value):
        text = ""
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            role = item.get("role")
            if role is not None and role != "user":
                continue
            text = _responses_content_text(
                item.get("content", item.get("text", ""))
            )
        if not text:
            continue
        parts.append(text)
        seen += 1
        if seen >= lookback:
            break
    return " ".join(reversed(parts))


def _responses_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return _responses_content_text(content.get("content"))
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = _responses_content_text(part)
            if text:
                parts.append(text)
        return " ".join(parts)
    return ""


def _payload_for_backend(
    body: dict[str, Any],
    backend: ProxyBackendConfig,
    *,
    model: str | None = None,
) -> bytes:
    payload = dict(body)
    payload["model"] = model or backend.model
    if backend.strip_tools:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        payload.pop("parallel_tool_calls", None)
        payload.pop("functions", None)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _streaming_verification_result(
    config: RoutingProxyConfig,
    decision: Any,
) -> dict[str, Any] | None:
    if config.verifier.mode == "off":
        return None
    base = _verification_base(config, decision)
    if base is None:
        return None
    return {**base, "status": "skipped_streaming"}


async def _verify_response_if_configured(
    config: RoutingProxyConfig,
    clients: dict[str, Any],
    runtime_manager: ManagedRuntimeManager,
    *,
    request_id: str,
    selected_engine: str,
    decision: Any,
    upstream_response: Any,
) -> dict[str, Any] | None:
    base = _verification_base(config, decision)
    if base is None:
        return None
    if upstream_response.status_code >= 500:
        return {**base, "status": "skipped_upstream_status"}
    if decision is None:
        return {**base, "status": "skipped_no_diagnostic_receipt"}
    if decision.requires_confirmation or selected_engine == FAIL_CLOSED_ENGINE:
        return {**base, "status": "skipped_human_confirm"}
    route_codes = set(base.get("route_codes", ()))
    configured_route_codes = set(config.verifier.route_codes)
    if configured_route_codes and route_codes.isdisjoint(configured_route_codes):
        return {**base, "status": "skipped_route"}
    if config.verifier.mode == "receipt-only":
        return {**base, "status": "qualified"}
    if config.verifier.mode == "sampled":
        if decision.risk_score >= 50:
            return {**base, "status": "skipped_risk"}
        if not _verification_sample_selected(request_id, config.verifier.sample_rate):
            return {**base, "status": "skipped_sample"}

    backend_name = config.verifier.backend
    backend_policy_reason = config.backend_policy_rejection_reason(backend_name)
    if backend_policy_reason is not None:
        return {**base, "status": "skipped_backend_policy", "error": backend_policy_reason}
    backend = config.backends.get(str(backend_name))
    if backend is None:
        return {**base, "status": "error", "error": "verifier backend missing"}

    started = time.perf_counter()
    try:
        _ensure_backend_supports_upstream_path(backend, "/chat/completions")
        await runtime_manager.ensure_running(backend)
        runtime_manager.begin_request(backend.name)
        try:
            response = await clients[backend.name].post(
                "/chat/completions",
                content=_verifier_payload(
                    config,
                    backend,
                    selected_engine=selected_engine,
                    decision=decision,
                    upstream_response=upstream_response,
                ),
                timeout=config.verifier.timeout_seconds,
            )
        finally:
            runtime_manager.end_request(backend.name)
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "backend": backend.name,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": exc.__class__.__name__,
        }

    status = "passed" if response.status_code < 500 else "failed"
    return {
        **base,
        "status": status,
        "backend": backend.name,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "status_code": int(response.status_code),
    }


def _verification_base(
    config: RoutingProxyConfig,
    decision: Any,
) -> dict[str, Any] | None:
    if config.verifier.mode == "off":
        return None
    reason_codes: tuple[str, ...] = ()
    if decision is not None:
        reason_codes = decision_to_receipt(decision).reason_codes
    return {
        "mode": config.verifier.mode,
        "status": "pending",
        "backend": config.verifier.backend,
        "route_codes": reason_codes,
    }


def _verification_sample_selected(request_id: str, sample_rate: float) -> bool:
    if sample_rate >= 1:
        return True
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return bucket < sample_rate


def _verifier_payload(
    config: RoutingProxyConfig,
    backend: ProxyBackendConfig,
    *,
    selected_engine: str,
    decision: Any,
    upstream_response: Any,
) -> bytes:
    receipt = decision_to_receipt(decision)
    response_preview = ""
    if config.verifier.include_response_preview:
        response_preview = redact_text(
            (upstream_response.text or "")[: config.verifier.max_response_preview_chars]
        )
    user_content = config.verifier.prompt_template.format(
        selected_engine=selected_engine,
        receipt_summary=receipt.summary,
        reason_codes=", ".join(receipt.reason_codes),
        policy_explanation=receipt.policy_explanation,
        fallback_explanation=receipt.fallback_explanation,
        safety_explanation=receipt.safety_explanation,
        privacy_explanation=receipt.privacy_explanation,
        response_preview=response_preview,
    )
    payload = {
        "model": backend.model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a configured ModelRouter verifier.",
            },
            {"role": "user", "content": user_content},
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _ensure_backend_supports_upstream_path(
    backend: ProxyBackendConfig,
    upstream_path: str,
) -> None:
    if (
        upstream_path == "/responses"
        and backend.runtime.enabled
        and backend.runtime.kind == "mlx-lm"
    ):
        raise UnsupportedUpstreamAPIError(
            "MLX-LM managed runtimes are chat/models-first; "
            "/v1/responses requires an upstream that supports Responses API."
        )


async def _post_with_fallbacks(
    clients: dict[str, Any],
    primary: ProxyBackendConfig,
    fallbacks: tuple[ProxyBackendConfig, ...],
    upstream_path: str,
    primary_payload: bytes,
    original_body: dict[str, Any],
    runtime_manager: ManagedRuntimeManager,
) -> tuple[Any, ProxyBackendConfig, bool]:
    attempted = (primary, *fallbacks)
    last_error: Exception | None = None
    for index, backend in enumerate(attempted):
        payload = (
            primary_payload
            if index == 0
            else _payload_for_backend(original_body, backend)
        )
        _ensure_backend_supports_upstream_path(backend, upstream_path)
        await runtime_manager.ensure_running(backend)
        runtime_manager.begin_request(backend.name)
        try:
            response = await clients[backend.name].post(
                upstream_path,
                content=payload,
            )
        except Exception as exc:
            last_error = exc
            runtime_manager.end_request(backend.name)
            continue
        runtime_manager.end_request(backend.name)
        if response.status_code < 500 or index == len(attempted) - 1:
            return response, backend, index > 0
    raise UpstreamRequestError(str(last_error) if last_error else "upstream failed")


async def _open_stream_with_fallbacks(
    clients: dict[str, Any],
    primary: ProxyBackendConfig,
    fallbacks: tuple[ProxyBackendConfig, ...],
    upstream_path: str,
    primary_payload: bytes,
    original_body: dict[str, Any],
    runtime_manager: ManagedRuntimeManager,
) -> tuple[Any, Any, ProxyBackendConfig, bool]:
    attempted = (primary, *fallbacks)
    last_error: Exception | None = None
    for index, backend in enumerate(attempted):
        payload = (
            primary_payload
            if index == 0
            else _payload_for_backend(original_body, backend)
        )
        _ensure_backend_supports_upstream_path(backend, upstream_path)
        await runtime_manager.ensure_running(backend)
        runtime_manager.begin_request(backend.name)
        try:
            stream_context = clients[backend.name].stream(
                "POST",
                upstream_path,
                content=payload,
            )
            response = await stream_context.__aenter__()
        except Exception as exc:
            last_error = exc
            runtime_manager.end_request(backend.name)
            continue
        if response.status_code >= 500 and index < len(attempted) - 1:
            try:
                await stream_context.__aexit__(None, None, None)
            finally:
                runtime_manager.end_request(backend.name)
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
    session_stats: ProxySessionStats | None = None,
    runtime_manager: ManagedRuntimeManager | None = None,
    verification: dict[str, Any] | None = None,
    selected_model: str | None = None,
    route_api: str = "route_fast",
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
            backend_model=selected_model or backend.model,
            fallback_used=fallback_used,
            status_code=response.status_code,
            stats=session_stats,
            verification=verification,
            route_api=route_api,
        )
        if runtime_manager is not None:
            runtime_manager.end_request(backend.name)
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
    stats: ProxySessionStats | None = None,
    verification: dict[str, Any] | None = None,
    route_api: str = "route_fast",
) -> None:
    if stats is not None:
        stats.record(
            selected_engine=selected_engine,
            status=status,
            backend=backend,
            fallback_used=fallback_used,
        )
    if writer is None:
        return
    writer.write(
        build_routing_event(
            request_id=request_id,
            route_api=route_api,
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
            verification=verification,
            routing_mode=config.proxy.routing_mode,
            decision_layer_enabled=config.proxy.routing_mode == "decision",
            selected_backend=backend,
            selected_model=backend_model,
        )
    )


def _format_proxy_session_summary(
    stats: ProxySessionStats,
    config: RoutingProxyConfig,
) -> str:
    lines = [
        "ModelRouter session summary",
        f"Events: {stats.total_events}",
        f"Engines: {_format_counter(stats.engine_counts)}",
        f"Backends: {_format_counter(stats.backend_counts)}",
        f"Statuses: {_format_counter(stats.status_counts)}",
        f"Fallbacks: {stats.fallback_count}",
        f"Interruptions: {stats.interruption_count}",
        f"Errors: {stats.error_count}",
    ]
    if not config.observability.enabled:
        lines.append("Telemetry: disabled; enable observability to persist events.")
    lines.extend(
        (
            "Review:",
            "  model-router telemetry summary "
            f"--events {config.observability.log_path} "
            f"--feedback {_feedback_path_for_config(config)}",
        )
    )
    return "\n".join(lines)


def _print_proxy_session_summary(
    stats: ProxySessionStats,
    config: RoutingProxyConfig,
) -> None:
    try:
        print(_format_proxy_session_summary(stats, config), flush=True)
    except Exception as exc:  # pragma: no cover - defensive only.
        LOG.warning("proxy session summary failed error=%s", exc.__class__.__name__)


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(
        f"{name}={count}"
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def _feedback_path_for_config(config: RoutingProxyConfig) -> str:
    if config.source_path.startswith("resource://"):
        return DEFAULT_FEEDBACK_PATH
    source_path = Path(config.source_path).expanduser()
    if source_path.name:
        return str(source_path.parent / "routing-feedback.jsonl")
    return DEFAULT_FEEDBACK_PATH


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
