import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse

import yaml

from hermes.plugins.model_router.proxy_dogfood import (
    DOGFOOD_CHECKS,
    DogfoodHTTPResponse,
    run_proxy_dogfood,
)


def _write_proxy_config(
    tmp_path: Path,
    *,
    backend_denylist: list[str] | None = None,
    fallback: bool = False,
) -> Path:
    config = {
        "proxy": {"host": "127.0.0.1", "port": 8082},
        "backends": {
            "fast": {
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "fast-model",
            },
            "deep": {
                "base_url": "http://127.0.0.1:1235/v1",
                "model": "deep-model",
            },
        },
        "engine_backends": {
            "fast_local": "fast",
            "balanced_local": "fast",
            "reasoning_local": "deep",
            "code_agent": "deep",
            "human_confirm": "fast",
        },
    }
    if backend_denylist:
        config["backend_policy"] = {
            "version": 1,
            "backend_denylist": backend_denylist,
        }
    if fallback:
        config["fallback_backends"] = {"fast": ["deep"]}
    path = tmp_path / "routing_proxy.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _response(
    status: int = 200,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
) -> DogfoodHTTPResponse:
    return DogfoodHTTPResponse(
        status_code=status,
        headers={key.lower(): value for key, value in (headers or {}).items()},
        body=json.dumps(payload or {"ok": True}).encode("utf-8"),
    )


def _fake_runner(*, policy_denied: bool = False, verifier_mode: str = "off"):
    calls: list[tuple[str, str]] = []

    def runner(
        method: str,
        url: str,
        body: bytes | None,
        _headers: dict[str, str],
        _timeout: float,
    ) -> DogfoodHTTPResponse:
        path = urlparse(url).path
        calls.append((method, path))
        if path == "/health":
            return _response(
                payload={
                    "status": "ok",
                    "backend_policy": {
                        "version": 1,
                        "backend_allowlist": [],
                        "backend_denylist": ["fast"] if policy_denied else [],
                    },
                    "verifier": {"mode": verifier_mode},
                }
            )
        if path == "/v1/models":
            return _response(payload={"data": [{"id": "model-router"}]})
        if path == "/v1/responses":
            if policy_denied:
                return _response(
                    502,
                    {"error": {"type": "backend_policy_rejected"}},
                )
            return _response(payload={"id": "resp-1", "output": []})
        if path == "/v1/chat/completions":
            text = (body or b"").decode("utf-8")
            if "Drop the production database." in text:
                return _response(
                    409,
                    {"error": {"type": "human_confirmation_required"}},
                )
            if policy_denied:
                return _response(
                    502,
                    {"error": {"type": "backend_policy_rejected"}},
                )
            return _response(
                payload={"choices": [{"message": {"content": "ok"}}]},
            )
        return _response(404, {"error": {"type": "not_found"}})

    return calls, runner


def test_proxy_dogfood_plan_does_not_call_live_proxy(tmp_path):
    config = _write_proxy_config(tmp_path)

    report = run_proxy_dogfood(
        config_path=config,
        execute=False,
        http_runner=lambda *_args: (_ for _ in ()).throw(AssertionError("called")),
    )

    payload = report.to_dict()
    assert report.ok is True
    assert report.executed is False
    assert report.planned == len(DOGFOOD_CHECKS)
    assert {check["name"] for check in payload["checks"]} == set(DOGFOOD_CHECKS)
    assert "Drop the production database." not in json.dumps(payload)
    assert "Rewrite this text." not in json.dumps(payload)


def test_proxy_dogfood_unavailable_proxy_skips_unless_required(tmp_path):
    config = _write_proxy_config(tmp_path)

    def unavailable(*_args):
        raise URLError("connection refused")

    skipped = run_proxy_dogfood(
        config_path=config,
        execute=True,
        http_runner=unavailable,
    )
    failed = run_proxy_dogfood(
        config_path=config,
        execute=True,
        require_running=True,
        http_runner=unavailable,
    )

    assert skipped.ok is True
    assert skipped.checks[0].name == "health"
    assert skipped.checks[0].status == "skipped"
    assert failed.ok is False
    assert failed.checks[0].status == "failed"


def test_proxy_dogfood_executes_sanitized_local_smoke_checks(tmp_path):
    config = _write_proxy_config(tmp_path)
    calls, runner = _fake_runner(verifier_mode="receipt-only")

    report = run_proxy_dogfood(
        config_path=config,
        execute=True,
        http_runner=runner,
    )

    checks = {check.name: check for check in report.checks}
    assert report.ok is True
    assert checks["health"].status == "passed"
    assert checks["models"].status == "passed"
    assert checks["chat_completions"].status == "passed"
    assert checks["chat_streaming"].status == "passed"
    assert checks["responses"].status == "passed"
    assert checks["human_confirm"].status == "passed"
    assert checks["backend_policy_rejection"].status == "skipped"
    assert checks["fallback"].status == "skipped"
    assert checks["verifier_modes"].status == "passed"
    assert ("GET", "/health") in calls
    assert "Drop the production database." not in json.dumps(report.to_dict())
    assert "Rewrite this text." not in json.dumps(report.to_dict())


def test_proxy_dogfood_backend_policy_rejection_is_explicit(tmp_path):
    config = _write_proxy_config(tmp_path, backend_denylist=["fast"])
    _calls, runner = _fake_runner(policy_denied=True)

    report = run_proxy_dogfood(
        config_path=config,
        execute=True,
        http_runner=runner,
    )

    checks = {check.name: check for check in report.checks}
    assert report.ok is True
    assert checks["chat_completions"].status == "skipped"
    assert checks["chat_streaming"].status == "skipped"
    assert checks["responses"].status == "skipped"
    assert checks["backend_policy_rejection"].status == "passed"
    assert checks["backend_policy_rejection"].status_code == 502
    assert checks["human_confirm"].status == "passed"
