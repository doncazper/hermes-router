#!/usr/bin/env python3
"""
Dual-model routing proxy for llama-server.

Classifies each prompt with hermes-router route_fast() and forwards to:
  fast  → configs/routing_proxy.yaml backends.fast  (Llama 3.2 3B)
  deep  → configs/routing_proxy.yaml backends.deep  (Qwopus 27B)

Config:  hermes-router/configs/routing_proxy.yaml
Usage:   .venv/bin/python scripts/routing_proxy.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_router import ModelRouter

# ── config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "routing_proxy.yaml"

def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

_cfg = _load_config()

PROXY_HOST: str = os.getenv("PROXY_HOST", _cfg["proxy"]["host"])
PROXY_PORT: int = int(os.getenv("PROXY_PORT", _cfg["proxy"]["port"]))

_fast_cfg = _cfg["backends"]["fast"]
_deep_cfg = _cfg["backends"]["deep"]
FAST_URL: str = os.getenv("FAST_MODEL_URL", _fast_cfg["url"])
DEEP_URL: str = os.getenv("DEEP_MODEL_URL", _deep_cfg["url"])
FAST_KEY: str = os.getenv("FAST_MODEL_API_KEY", _fast_cfg["api_key"])
DEEP_KEY: str = os.getenv("DEEP_MODEL_API_KEY", _deep_cfg["api_key"])

_routing = _cfg["routing"]
DEEP_ENGINES: frozenset[str] = frozenset(_routing["deep_engines"])
_CODING_OVERRIDE_TOKENS: frozenset[str] = frozenset(_routing["coding_override_tokens"])
_OVERRIDE_SCAN_CHARS: int = _routing.get("override_scan_chars", 300)
_CONTEXT_LOOKBACK: int = _routing.get("context_lookback_turns", 3)
_FALLBACK_TO_DEEP: bool = _routing.get("fallback_to_deep", True)

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("routing-proxy")

# ── router + clients ──────────────────────────────────────────────────────────

_model_router = ModelRouter.from_config(validate_availability=False)
_fast_client: httpx.AsyncClient
_deep_client: httpx.AsyncClient


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _fast_client, _deep_client
    _fast_client = httpx.AsyncClient(
        base_url=FAST_URL,
        headers={"Authorization": f"Bearer {FAST_KEY}"},
        timeout=300,
    )
    _deep_client = httpx.AsyncClient(
        base_url=DEEP_URL,
        headers={"Authorization": f"Bearer {DEEP_KEY}"},
        timeout=300,
    )
    log.info("Routing proxy  →  http://%s:%d/v1", PROXY_HOST, PROXY_PORT)
    log.info("  fast_local   →  %s", FAST_URL)
    log.info("  code_agent   →  %s", DEEP_URL)
    log.info("  fallback     →  %s", "deep on fast error" if _FALLBACK_TO_DEEP else "disabled")
    yield
    await asyncio.gather(_fast_client.aclose(), _deep_client.aclose())


app = FastAPI(title="hermes-routing-proxy", lifespan=lifespan, docs_url=None, redoc_url=None)

# ── routing logic ─────────────────────────────────────────────────────────────

def _has_coding_context(text: str) -> bool:
    # Cap scan to avoid splitting huge pasted files on every request.
    return bool(set(text[:_OVERRIDE_SCAN_CHARS].lower().split()) & _CODING_OVERRIDE_TOKENS)


def _extract_user_text(messages: list[dict]) -> str:
    """Concatenate the last N user turns for context-aware routing."""
    parts: list[str] = []
    seen = 0
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts.append(" ".join(p.get("text", "") for p in content if isinstance(p, dict)))
        else:
            parts.append(str(content))
        seen += 1
        if seen >= _CONTEXT_LOOKBACK:
            break
    return " ".join(reversed(parts))


def _pick_client(messages: list[dict]) -> tuple[str, httpx.AsyncClient, httpx.AsyncClient | None]:
    """Return (engine, primary_client, fallback_client_or_None)."""
    text = _extract_user_text(messages)
    engine = _model_router.route_fast(text) if text else "code_agent"
    if engine in DEEP_ENGINES or _has_coding_context(text):
        return engine, _deep_client, None
    fallback = _deep_client if _FALLBACK_TO_DEEP else None
    return engine, _fast_client, fallback

# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    raw = await request.body()
    body = await request.json()
    messages = body.get("messages", [])
    engine, client, fallback = _pick_client(messages)
    req_id = uuid.uuid4().hex[:8]
    log.info("[%s] %-20s → %s", req_id, engine, client.base_url)

    upstream_headers = {"Content-Type": "application/json"}

    if body.get("stream", False):
        async def _stream(c: httpx.AsyncClient):
            async with c.stream("POST", "/chat/completions", content=raw, headers=upstream_headers) as resp:
                if resp.status_code >= 400 and fallback and c is not fallback:
                    log.warning("[%s] fast returned %d, falling back to deep", req_id, resp.status_code)
                    async with fallback.stream("POST", "/chat/completions", content=raw, headers=upstream_headers) as fb_resp:
                        async for chunk in fb_resp.aiter_bytes():
                            yield chunk
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            _stream(client),
            media_type="text/event-stream",
            headers={"X-Routed-Engine": engine, "X-Request-Id": req_id},
        )

    resp = await client.post("/chat/completions", content=raw, headers=upstream_headers)
    if resp.status_code >= 400 and fallback and client is not fallback:
        log.warning("[%s] fast returned %d, falling back to deep", req_id, resp.status_code)
        resp = await fallback.post("/chat/completions", content=raw, headers=upstream_headers)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type="application/json",
        headers={"X-Routed-Engine": engine, "X-Request-Id": req_id},
    )


@app.get("/v1/models")
async def list_models() -> dict:
    async def _fetch(c: httpx.AsyncClient) -> list[dict]:
        try:
            r = await c.get("/models")
            return r.json().get("data", [])
        except Exception as exc:
            log.warning("models fetch failed for %s: %s", c.base_url, exc)
            return []

    results = await asyncio.gather(_fetch(_fast_client), _fetch(_deep_client))
    return {"object": "list", "data": [m for batch in results for m in batch]}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "fast": str(_fast_client.base_url),
        "deep": str(_deep_client.base_url),
        "fallback_enabled": _FALLBACK_TO_DEEP,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")
