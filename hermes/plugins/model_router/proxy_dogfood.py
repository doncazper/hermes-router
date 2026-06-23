"""Opt-in local proxy dogfood harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from hermes.plugins.model_router.proxy_config import RoutingProxyConfig, load_proxy_config


DOGFOOD_CHECKS = (
    "health",
    "models",
    "chat_completions",
    "chat_streaming",
    "responses",
    "human_confirm",
    "backend_policy_rejection",
    "fallback",
    "verifier_modes",
)


@dataclass(frozen=True)
class DogfoodHTTPResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes

    def json_payload(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True)
class DogfoodCheckResult:
    name: str
    status: str
    detail: str
    status_code: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"passed", "skipped", "planned"}

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class DogfoodReport:
    executed: bool
    endpoint: str
    config_path: str
    ok: bool
    checks: tuple[DogfoodCheckResult, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> int:
        return sum(1 for check in self.checks if check.status == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for check in self.checks if check.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for check in self.checks if check.status == "skipped")

    @property
    def planned(self) -> int:
        return sum(1 for check in self.checks if check.status == "planned")

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "endpoint": self.endpoint,
            "config_path": self.config_path,
            "ok": self.ok,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "planned": self.planned,
            "checks": [check.to_dict() for check in self.checks],
            "notes": list(self.notes),
        }


HTTPRunner = Callable[
    [str, str, bytes | None, dict[str, str], float],
    DogfoodHTTPResponse,
]


def run_proxy_dogfood(
    *,
    config_path: str | Path,
    endpoint: str | None = None,
    execute: bool = False,
    require_running: bool = False,
    timeout_seconds: float = 5.0,
    http_runner: HTTPRunner | None = None,
) -> DogfoodReport:
    config = load_proxy_config(config_path)
    resolved_endpoint = (endpoint or _endpoint_for_config(config)).rstrip("/")
    config_source = str(Path(config_path).expanduser())
    if not execute:
        return DogfoodReport(
            executed=False,
            endpoint=resolved_endpoint,
            config_path=config_source,
            ok=True,
            checks=tuple(
                DogfoodCheckResult(
                    name=name,
                    status="planned",
                    detail="Pass --execute to run this live local proxy check.",
                )
                for name in DOGFOOD_CHECKS
            ),
            notes=(
                "Plan only; no proxy, backend, verifier, or hosted calls were made.",
                "Live dogfood is local and opt-in with --execute.",
            ),
        )

    runner = http_runner or _urllib_runner
    headers = _proxy_headers(config)
    checks: list[DogfoodCheckResult] = []
    health_response = _safe_request(
        runner,
        "GET",
        _url(resolved_endpoint, "/health"),
        None,
        headers,
        timeout_seconds,
    )
    if isinstance(health_response, DogfoodCheckResult):
        status = "failed" if require_running else "skipped"
        checks.append(
            DogfoodCheckResult(
                "health",
                status,
                health_response.detail,
                health_response.status_code,
            )
        )
        checks.extend(
            DogfoodCheckResult(
                name,
                "skipped",
                "Proxy health check did not pass, so this live check was skipped.",
            )
            for name in DOGFOOD_CHECKS
            if name != "health"
        )
        return _report(
            executed=True,
            endpoint=resolved_endpoint,
            config_path=config_source,
            checks=checks,
        )

    health_payload = health_response.json_payload()
    checks.append(
        _http_status_check(
            "health",
            health_response,
            "Proxy health endpoint responded.",
        )
    )
    checks.append(
        _request_check(
            runner,
            "models",
            "GET",
            _url(resolved_endpoint, "/v1/models"),
            None,
            headers,
            timeout_seconds,
            success_detail="/v1/models responded with configured proxy model ids.",
        )
    )
    checks.append(
        _request_check(
            runner,
            "chat_completions",
            "POST",
            _url(resolved_endpoint, "/v1/chat/completions"),
            _chat_payload("Rewrite this text."),
            headers,
            timeout_seconds,
            success_detail="/v1/chat/completions routed a sanitized smoke prompt.",
            skip_error_types={
                "backend_policy_rejected": (
                    "Chat route was blocked by backend policy; "
                    "backend_policy_rejection covers this fail-closed path."
                ),
            },
        )
    )
    checks.append(
        _request_check(
            runner,
            "chat_streaming",
            "POST",
            _url(resolved_endpoint, "/v1/chat/completions"),
            _chat_payload("Rewrite this text.", stream=True),
            headers,
            timeout_seconds,
            success_detail="Streaming chat request reached the proxy.",
            skip_error_types={
                "backend_policy_rejected": (
                    "Streaming chat route was blocked by backend policy; "
                    "backend_policy_rejection covers this fail-closed path."
                ),
            },
        )
    )
    checks.append(
        _responses_check(
            runner,
            resolved_endpoint,
            headers,
            timeout_seconds,
        )
    )
    checks.append(
        _human_confirm_check(
            runner,
            resolved_endpoint,
            headers,
            timeout_seconds,
        )
    )
    checks.append(
        _backend_policy_check(
            runner,
            resolved_endpoint,
            headers,
            timeout_seconds,
            config,
            health_payload,
        )
    )
    checks.append(
        _fallback_check(
            runner,
            resolved_endpoint,
            headers,
            timeout_seconds,
            config,
        )
    )
    checks.append(_verifier_check(health_payload))
    return _report(
        executed=True,
        endpoint=resolved_endpoint,
        config_path=config_source,
        checks=checks,
    )


def _report(
    *,
    executed: bool,
    endpoint: str,
    config_path: str,
    checks: list[DogfoodCheckResult],
) -> DogfoodReport:
    failed = any(check.status == "failed" for check in checks)
    return DogfoodReport(
        executed=executed,
        endpoint=endpoint,
        config_path=config_path,
        ok=not failed,
        checks=tuple(checks),
        notes=(
            "Dogfood prompts are fixed sanitized smoke prompts and are not serialized.",
            "Skipped checks usually need a specific local runtime or proxy policy setup.",
        ),
    )


def _endpoint_for_config(config: RoutingProxyConfig) -> str:
    return f"http://{config.proxy.host}:{config.proxy.port}"


def _proxy_headers(config: RoutingProxyConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = config.proxy.resolved_api_key
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _url(endpoint: str, path: str) -> str:
    return urljoin(endpoint.rstrip("/") + "/", path.lstrip("/"))


def _safe_request(
    runner: HTTPRunner,
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout_seconds: float,
) -> DogfoodHTTPResponse | DogfoodCheckResult:
    try:
        return runner(method, url, body, headers, timeout_seconds)
    except HTTPError as exc:
        return DogfoodHTTPResponse(
            status_code=int(exc.code),
            headers=dict(exc.headers.items()),
            body=exc.read(),
        )
    except (OSError, TimeoutError, URLError) as exc:
        return DogfoodCheckResult(
            name="request",
            status="failed",
            detail=f"Request failed: {exc.__class__.__name__}",
        )


def _request_check(
    runner: HTTPRunner,
    name: str,
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout_seconds: float,
    *,
    success_detail: str,
    skip_error_types: dict[str, str] | None = None,
) -> DogfoodCheckResult:
    response = _safe_request(runner, method, url, body, headers, timeout_seconds)
    if isinstance(response, DogfoodCheckResult):
        return DogfoodCheckResult(name, "failed", response.detail, response.status_code)
    if skip_error_types and response.status_code >= 400:
        error_type = _error_type(response.json_payload())
        skip_detail = skip_error_types.get(error_type)
        if skip_detail:
            return DogfoodCheckResult(name, "skipped", skip_detail, response.status_code)
    return _http_status_check(name, response, success_detail)


def _http_status_check(
    name: str,
    response: DogfoodHTTPResponse,
    success_detail: str,
) -> DogfoodCheckResult:
    if 200 <= response.status_code < 300:
        return DogfoodCheckResult(name, "passed", success_detail, response.status_code)
    return DogfoodCheckResult(
        name,
        "failed",
        f"Unexpected HTTP {response.status_code}.",
        response.status_code,
    )


def _responses_check(
    runner: HTTPRunner,
    endpoint: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> DogfoodCheckResult:
    response = _safe_request(
        runner,
        "POST",
        _url(endpoint, "/v1/responses"),
        _responses_payload("Rewrite this text."),
        headers,
        timeout_seconds,
    )
    if isinstance(response, DogfoodCheckResult):
        return DogfoodCheckResult("responses", "failed", response.detail)
    payload = response.json_payload()
    error_type = _error_type(payload)
    if response.status_code == 502 and error_type == "upstream_api_unsupported":
        return DogfoodCheckResult(
            "responses",
            "skipped",
            "Configured backend does not support /v1/responses.",
            response.status_code,
        )
    if response.status_code == 502 and error_type == "backend_policy_rejected":
        return DogfoodCheckResult(
            "responses",
            "skipped",
            "Responses route was blocked by backend policy; "
            "backend_policy_rejection covers this fail-closed path.",
            response.status_code,
        )
    return _http_status_check(
        "responses",
        response,
        "/v1/responses reached the proxy.",
    )


def _human_confirm_check(
    runner: HTTPRunner,
    endpoint: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> DogfoodCheckResult:
    response = _safe_request(
        runner,
        "POST",
        _url(endpoint, "/v1/chat/completions"),
        _chat_payload("Drop the production database."),
        headers,
        timeout_seconds,
    )
    if isinstance(response, DogfoodCheckResult):
        return DogfoodCheckResult("human_confirm", "failed", response.detail)
    payload = response.json_payload()
    if response.status_code == 409 and _error_type(payload) == "human_confirmation_required":
        return DogfoodCheckResult(
            "human_confirm",
            "passed",
            "High-risk prompt failed closed to human confirmation.",
            response.status_code,
        )
    return DogfoodCheckResult(
        "human_confirm",
        "failed",
        "High-risk prompt did not return human_confirmation_required.",
        response.status_code,
    )


def _backend_policy_check(
    runner: HTTPRunner,
    endpoint: str,
    headers: dict[str, str],
    timeout_seconds: float,
    config: RoutingProxyConfig,
    health_payload: dict[str, Any],
) -> DogfoodCheckResult:
    policy = health_payload.get("backend_policy")
    if not isinstance(policy, dict):
        policy = config.backend_policy.to_dict()
    denied = set(policy.get("backend_denylist") or ())
    selected_backend = config.engine_backends.get("fast_local")
    if not denied or selected_backend not in denied:
        return DogfoodCheckResult(
            "backend_policy_rejection",
            "skipped",
            "No configured policy currently denies the fast local dogfood route.",
        )
    response = _safe_request(
        runner,
        "POST",
        _url(endpoint, "/v1/chat/completions"),
        _chat_payload("Rewrite this text."),
        headers,
        timeout_seconds,
    )
    if isinstance(response, DogfoodCheckResult):
        return DogfoodCheckResult("backend_policy_rejection", "failed", response.detail)
    payload = response.json_payload()
    if response.status_code == 502 and _error_type(payload) == "backend_policy_rejected":
        return DogfoodCheckResult(
            "backend_policy_rejection",
            "passed",
            "Denied backend was rejected before upstream forwarding.",
            response.status_code,
        )
    return DogfoodCheckResult(
        "backend_policy_rejection",
        "failed",
        "Denied backend did not return backend_policy_rejected.",
        response.status_code,
    )


def _fallback_check(
    runner: HTTPRunner,
    endpoint: str,
    headers: dict[str, str],
    timeout_seconds: float,
    config: RoutingProxyConfig,
) -> DogfoodCheckResult:
    if not config.fallback_backends:
        return DogfoodCheckResult(
            "fallback",
            "skipped",
            "No proxy fallback chain is configured.",
        )
    response = _safe_request(
        runner,
        "POST",
        _url(endpoint, "/v1/chat/completions"),
        _chat_payload("Rewrite this text."),
        headers,
        timeout_seconds,
    )
    if isinstance(response, DogfoodCheckResult):
        return DogfoodCheckResult("fallback", "failed", response.detail)
    fallback = response.headers.get("x-modelrouter-fallback", "").lower()
    if fallback == "true":
        return DogfoodCheckResult(
            "fallback",
            "passed",
            "Proxy reported fallback usage.",
            response.status_code,
        )
    return DogfoodCheckResult(
        "fallback",
        "skipped",
        "Fallback chain is configured but was not exercised by the healthy primary.",
        response.status_code,
    )


def _verifier_check(health_payload: dict[str, Any]) -> DogfoodCheckResult:
    verifier = health_payload.get("verifier")
    if not isinstance(verifier, dict):
        return DogfoodCheckResult(
            "verifier_modes",
            "skipped",
            "Health payload did not expose verifier metadata.",
        )
    mode = str(verifier.get("mode") or "off")
    if mode == "off":
        return DogfoodCheckResult(
            "verifier_modes",
            "skipped",
            "Verifier is disabled by default.",
        )
    return DogfoodCheckResult(
        "verifier_modes",
        "passed",
        f"Verifier mode is visible as {mode}.",
    )


def _chat_payload(prompt: str, *, stream: bool = False) -> bytes:
    return json.dumps(
        {
            "model": "model-router",
            "stream": stream,
            "messages": [{"role": "user", "content": prompt}],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _responses_payload(prompt: str) -> bytes:
    return json.dumps(
        {
            "model": "model-router",
            "input": prompt,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _error_type(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        return error["type"]
    return ""


def _urllib_runner(
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout_seconds: float,
) -> DogfoodHTTPResponse:
    request = Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return DogfoodHTTPResponse(
            status_code=int(response.status),
            headers={key.lower(): value for key, value in response.headers.items()},
            body=response.read(),
        )
