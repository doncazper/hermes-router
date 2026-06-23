"""Privacy-safe local backend benchmark helpers for setup-time advice."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    RoutingProxyConfig,
    load_proxy_config,
)
from hermes.plugins.model_router.routing_log import estimated_tokens


DEFAULT_BENCHMARK_PATH = "~/.model-router/benchmarks.json"
DEFAULT_CONFIG_DIR = "~/.model-router"
BENCHMARK_STORE_VERSION = 1
SYNTHETIC_BENCHMARK_PROMPT = (
    "Reply with one short sentence confirming this local benchmark is ready."
)
SYNTHETIC_PROMPT_HASH = hashlib.sha256(
    SYNTHETIC_BENCHMARK_PROMPT.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class BenchmarkTarget:
    backend: str
    route: str
    model: str
    base_url: str
    runtime_kind: str
    managed_runtime: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    route: str
    model: str
    base_url: str
    runtime_kind: str
    managed_runtime: bool
    status: str
    timestamp: str
    synthetic_prompt_hash: str = SYNTHETIC_PROMPT_HASH
    startup_time_ms: float | None = None
    first_token_latency_ms: float | None = None
    total_latency_ms: float | None = None
    tokens_per_second: float | None = None
    measured_tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class BenchmarkExecution:
    executed: bool
    output_path: str
    results: tuple[BenchmarkResult, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(result.status in {"planned", "completed"} for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "ok": self.ok,
            "output_path": self.output_path,
            "results": [result.to_dict() for result in self.results],
            "notes": list(self.notes),
        }


def default_benchmark_path(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> Path:
    return Path(config_dir).expanduser() / "benchmarks.json"


def plan_backend_benchmarks(
    config_path: str | Path,
    *,
    backends: Sequence[str] | None = None,
) -> tuple[BenchmarkTarget, ...]:
    config = load_proxy_config(config_path)
    selected = set(backends or ())
    route_by_backend = _route_by_backend(config)
    targets: list[BenchmarkTarget] = []
    for backend in config.backends.values():
        if selected and backend.name not in selected:
            continue
        if not _is_local_base_url(backend.base_url):
            continue
        route = route_by_backend.get(backend.name, backend.name)
        targets.append(_target_for_backend(backend, route))
    return tuple(targets)


def execute_benchmark_plan(
    targets: Sequence[BenchmarkTarget],
    *,
    output_path: str | Path = DEFAULT_BENCHMARK_PATH,
    execute: bool,
    confirmed: bool,
    timeout_seconds: float = 30.0,
    runner: Callable[[BenchmarkTarget, float], BenchmarkResult] | None = None,
) -> BenchmarkExecution:
    expanded_output = Path(output_path).expanduser()
    if not execute:
        return BenchmarkExecution(
            executed=False,
            output_path=str(expanded_output),
            results=tuple(_planned_result(target) for target in targets),
            notes=("Dry run only; pass --execute --yes to run local benchmark calls.",),
        )
    if not confirmed:
        return BenchmarkExecution(
            executed=False,
            output_path=str(expanded_output),
            results=tuple(
                _status_result(target, "confirmation_required") for target in targets
            ),
            notes=("Benchmark execution requires explicit confirmation.",),
        )

    runner = runner or run_backend_benchmark
    results = tuple(runner(target, timeout_seconds) for target in targets)
    _write_benchmark_store(expanded_output, results)
    return BenchmarkExecution(
        executed=True,
        output_path=str(expanded_output),
        results=results,
        notes=(
            "Stored privacy-safe benchmark metrics only; no prompt bodies or secrets.",
        ),
    )


def run_backend_benchmark(
    target: BenchmarkTarget,
    timeout_seconds: float = 30.0,
) -> BenchmarkResult:
    url = urljoin(target.base_url.rstrip("/") + "/", "chat/completions")
    payload = {
        "model": target.model,
        "messages": [{"role": "user", "content": SYNTHETIC_BENCHMARK_PROMPT}],
        "max_tokens": 24,
        "temperature": 0,
        "stream": False,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
        elapsed_ms = (perf_counter() - started) * 1000
        completion_text = _completion_text(body)
        tokens = _completion_tokens(body, completion_text)
        tokens_per_second = (
            round(tokens / (elapsed_ms / 1000), 3) if tokens and elapsed_ms > 0 else None
        )
        return BenchmarkResult(
            backend=target.backend,
            route=target.route,
            model=target.model,
            base_url=target.base_url,
            runtime_kind=target.runtime_kind,
            managed_runtime=target.managed_runtime,
            status="completed",
            timestamp=_now_iso(),
            total_latency_ms=round(elapsed_ms, 3),
            tokens_per_second=tokens_per_second,
            measured_tokens=tokens,
        )
    except HTTPError as exc:
        return _failed_result(target, f"HTTP {exc.code}")
    except (OSError, URLError, TimeoutError) as exc:
        return _failed_result(target, exc.__class__.__name__)


def load_benchmark_results(path: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    expanded = Path(path or DEFAULT_BENCHMARK_PATH).expanduser()
    if not expanded.exists():
        return ()
    try:
        payload = json.loads(expanded.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict))


def benchmark_summary(path: str | Path | None = None) -> dict[str, Any]:
    rows = load_benchmark_results(path)
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") not in {"completed", "planned"}]
    best = sorted(
        completed,
        key=lambda row: float(row.get("tokens_per_second") or 0),
        reverse=True,
    )[:5]
    return {
        "results": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "best": [
            {
                "backend": row.get("backend"),
                "route": row.get("route"),
                "model": row.get("model"),
                "tokens_per_second": row.get("tokens_per_second"),
                "total_latency_ms": row.get("total_latency_ms"),
            }
            for row in best
        ],
    }


def _route_by_backend(config: RoutingProxyConfig) -> dict[str, str]:
    routes: dict[str, str] = {}
    for engine, backend_name in config.engine_backends.items():
        routes.setdefault(backend_name, engine)
    return routes


def _target_for_backend(backend: ProxyBackendConfig, route: str) -> BenchmarkTarget:
    return BenchmarkTarget(
        backend=backend.name,
        route=route,
        model=backend.model,
        base_url=backend.base_url,
        runtime_kind=backend.runtime.kind,
        managed_runtime=backend.runtime.enabled,
    )


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"} or hostname.startswith("192.168.")


def _planned_result(target: BenchmarkTarget) -> BenchmarkResult:
    return _status_result(target, "planned")


def _status_result(target: BenchmarkTarget, status: str) -> BenchmarkResult:
    return BenchmarkResult(
        backend=target.backend,
        route=target.route,
        model=target.model,
        base_url=target.base_url,
        runtime_kind=target.runtime_kind,
        managed_runtime=target.managed_runtime,
        status=status,
        timestamp=_now_iso(),
    )


def _failed_result(target: BenchmarkTarget, error: str) -> BenchmarkResult:
    return BenchmarkResult(
        backend=target.backend,
        route=target.route,
        model=target.model,
        base_url=target.base_url,
        runtime_kind=target.runtime_kind,
        managed_runtime=target.managed_runtime,
        status="failed",
        timestamp=_now_iso(),
        error=error,
    )


def _write_benchmark_store(path: Path, results: tuple[BenchmarkResult, ...]) -> None:
    existing = list(load_benchmark_results(path))
    rows = [*existing, *(result.to_dict() for result in results)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": BENCHMARK_STORE_VERSION,
                "updated_at": _now_iso(),
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _completion_text(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    output_text = payload.get("output_text")
    return output_text if isinstance(output_text, str) else ""


def _completion_tokens(body: bytes, completion_text: str) -> int:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return estimated_tokens(completion_text)
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            value = usage.get("completion_tokens") or usage.get("output_tokens")
            if isinstance(value, int) and value > 0:
                return value
    return estimated_tokens(completion_text)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
