"""Privacy-safe ModelRouter eval execution helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from hermes.plugins.model_router.evals import (
    EVAL_FIXTURE_SCHEMA_VERSION,
    EVAL_SCORER_VERSION,
    EvalFixture,
    EvalFixtureError,
    load_builtin_eval_fixtures,
    score_eval_output,
)
from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    RoutingProxyConfig,
    load_proxy_config,
)


DEFAULT_EVAL_RESULTS_PATH = "~/.model-router/evals/results.jsonl"
EVAL_RESULT_VERSION = 1


@dataclass(frozen=True)
class EvalBackendRequest:
    backend: str
    selected_engine: str | None
    model: str
    base_url: str
    system_prompt: str | None
    prompt: str
    timeout_seconds: float
    api_key: str | None = None


@dataclass(frozen=True)
class EvalBackendResponse:
    status: str
    output: str = ""
    latency_ms: float | None = None
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    usage_total_tokens: int | None = None
    upstream_model: str | None = None
    timed_out: bool = False
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class EvalRunResult:
    run_id: str
    created_at: str
    backend: str
    model: str
    selected_engine: str | None
    fixture_id: str
    category: str
    score_percent: float
    weighted_score: float
    exit_status: str
    status: str
    latency_ms: float | None
    timeout: bool
    scorer_version: int
    fixture_version: int
    fixture_prompt_hash: str
    output_hash: str
    passed_checks: int
    total_checks: int
    checks: tuple[dict[str, Any], ...]
    failure_reasons: tuple[str, ...]
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    usage_total_tokens: int | None = None
    upstream_model: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        return self.exit_status == "passed"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": EVAL_RESULT_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "backend": self.backend,
            "model": self.model,
            "selected_engine": self.selected_engine,
            "fixture_id": self.fixture_id,
            "category": self.category,
            "score_percent": self.score_percent,
            "weighted_score": self.weighted_score,
            "exit_status": self.exit_status,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "timeout": self.timeout,
            "scorer_version": self.scorer_version,
            "fixture_version": self.fixture_version,
            "fixture_prompt_hash": self.fixture_prompt_hash,
            "output_hash": self.output_hash,
            "passed_checks": self.passed_checks,
            "total_checks": self.total_checks,
            "checks": list(self.checks),
            "failure_reasons": list(self.failure_reasons),
            "usage_prompt_tokens": self.usage_prompt_tokens,
            "usage_completion_tokens": self.usage_completion_tokens,
            "usage_total_tokens": self.usage_total_tokens,
            "upstream_model": self.upstream_model,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class EvalRunExecution:
    run_id: str
    created_at: str
    output_path: str
    backend: str
    model: str
    results: tuple[EvalRunResult, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(result.status == "completed" for result in self.results)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "ok": self.ok,
            "passed": self.passed,
            "output_path": self.output_path,
            "backend": self.backend,
            "model": self.model,
            "results": [result.to_dict() for result in self.results],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EvalRunReport:
    run_id: str | None
    result_path: str
    backend: str | None
    model: str | None
    selected_engine: str | None
    total: int
    completed: int
    failed: int
    passed: int
    timeouts: int
    unknown: int
    score_mean_percent: float | None
    weighted_score_mean: float | None
    latency_summary: Mapping[str, Any]
    usage_summary: Mapping[str, Any]
    by_category: Mapping[str, dict[str, Any]]
    top_failure_reasons: tuple[dict[str, Any], ...]
    suitability_notes: tuple[str, ...]
    privacy: Mapping[str, Any]
    results: tuple[dict[str, Any], ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EVAL_RESULT_VERSION,
            "run_id": self.run_id,
            "result_path": self.result_path,
            "backend": self.backend,
            "model": self.model,
            "selected_engine": self.selected_engine,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "passed": self.passed,
            "timeouts": self.timeouts,
            "unknown": self.unknown,
            "score_mean_percent": self.score_mean_percent,
            "weighted_score_mean": self.weighted_score_mean,
            "latency_summary": dict(self.latency_summary),
            "usage_summary": dict(self.usage_summary),
            "by_category": dict(self.by_category),
            "top_failure_reasons": list(self.top_failure_reasons),
            "suitability_notes": list(self.suitability_notes),
            "privacy": dict(self.privacy),
            "results": list(self.results),
            "notes": list(self.notes),
        }


def eval_evidence_for_model(
    model: str,
    *,
    result_path: str | Path = DEFAULT_EVAL_RESULTS_PATH,
    backend: str | None = None,
) -> dict[str, Any]:
    rows = load_eval_results(result_path)
    return eval_evidence_from_rows(model, rows, backend=backend)


def eval_evidence_from_rows(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    model_id = model.strip()
    if not model_id:
        return _not_evaluated_summary(model="", backend=backend)
    matching = tuple(
        row
        for row in rows
        if _row_matches_model(row, model_id)
        and (backend is None or row.get("backend") == backend)
    )
    if not matching:
        return _not_evaluated_summary(model=model_id, backend=backend)
    latest_run_id = _latest_run_id(matching)
    selected = tuple(
        row
        for row in matching
        if latest_run_id is not None and row.get("run_id") == latest_run_id
    )
    if not selected:
        return _not_evaluated_summary(model=model_id, backend=backend)
    completed = sum(1 for row in selected if row.get("status") == "completed")
    passed = sum(1 for row in selected if row.get("exit_status") == "passed")
    failed = sum(1 for row in selected if _row_failed(row))
    timeouts = sum(1 for row in selected if _row_timed_out(row))
    scores = _numeric_values(selected, "score_percent")
    weighted_scores = _numeric_values(selected, "weighted_score")
    stale, stale_reasons = _staleness(selected)
    return {
        "status": "stale" if stale else "evaluated",
        "model": model_id,
        "backend": _unique_or_mixed(selected, "backend") or backend,
        "selected_engine": _unique_or_mixed(selected, "selected_engine"),
        "latest_run_id": latest_run_id,
        "last_evaluated_at": _latest_created_at(selected),
        "fixture_count": len(selected),
        "completed": completed,
        "passed": passed,
        "failed": failed,
        "timeouts": timeouts,
        "score_mean_percent": round(sum(scores) / len(scores), 2)
        if scores
        else None,
        "weighted_score_mean": round(sum(weighted_scores) / len(weighted_scores), 4)
        if weighted_scores
        else None,
        "by_category": _category_summary(selected),
        "top_failure_reasons": list(_top_failure_reasons(selected)),
        "latency_summary": _latency_summary(selected),
        "usage_summary": _usage_summary(selected),
        "stale": stale,
        "stale_reasons": stale_reasons,
        "privacy": _privacy_summary(),
        "notes": [
            "Eval evidence is advisory and does not change routing automatically.",
            "Best means best on this fixture set/profile, not universal model quality.",
        ],
    }


def eval_fixture_summaries(
    *,
    category: str | None = None,
) -> tuple[dict[str, Any], ...]:
    fixtures = _fixtures_for_selector(category, all_fixtures=category is None)
    return tuple(_fixture_summary(fixture) for fixture in fixtures)


def execute_eval_run(
    *,
    config_path: str | Path,
    backend: str,
    model: str | None = None,
    fixture_selector: str | None = None,
    all_fixtures: bool = False,
    output_path: str | Path = DEFAULT_EVAL_RESULTS_PATH,
    timeout_seconds: float | None = None,
    run_id: str | None = None,
    confirm_large_run: bool = False,
    runner: Callable[[EvalBackendRequest], EvalBackendResponse] | None = None,
) -> EvalRunExecution:
    config = load_proxy_config(config_path)
    backend_config = _backend(config, backend)
    selected_model = model or backend_config.model
    selected_engine = _selected_engine_for_backend(config, backend)
    fixtures = _fixtures_for_selector(fixture_selector, all_fixtures=all_fixtures)
    request_timeout = timeout_seconds or backend_config.timeout_seconds
    if all_fixtures and not confirm_large_run:
        estimated_timeout = round(len(fixtures) * request_timeout, 2)
        raise EvalFixtureError(
            "running all built-in eval fixtures requires --confirm-large-run "
            f"({len(fixtures)} backend requests; timeout budget up to "
            f"{estimated_timeout}s). Evals are explicit operator actions and "
            "do not sweep discovered models automatically."
        )
    created_at = _now_iso()
    resolved_run_id = run_id or _new_run_id(created_at)
    invoke = runner or run_backend_eval_request
    results = tuple(
        _run_fixture(
            fixture,
            backend=backend_config,
            selected_engine=selected_engine,
            model=selected_model,
            run_id=resolved_run_id,
            created_at=created_at,
            timeout_seconds=request_timeout,
            runner=invoke,
        )
        for fixture in fixtures
    )
    expanded_output = Path(output_path).expanduser()
    _append_eval_results(expanded_output, results)
    notes = [
        "Stored eval scores, hashes, latency, status, and usage only.",
        "Raw prompts and raw model outputs were not retained.",
        "Eval runs are explicit and do not change routing automatically.",
        "Interpret scores as best on this fixture set/profile, not as a universal model ranking.",
    ]
    if all_fixtures:
        estimated_timeout = round(len(fixtures) * request_timeout, 2)
        notes.append(
            f"Confirmed broad fixture run: {len(fixtures)} backend requests; "
            f"timeout budget up to {estimated_timeout}s. Review provider or "
            "runtime cost before hosted runs."
        )
    return EvalRunExecution(
        run_id=resolved_run_id,
        created_at=created_at,
        output_path=str(expanded_output),
        backend=backend,
        model=selected_model,
        results=results,
        notes=tuple(notes),
    )


def run_backend_eval_request(request: EvalBackendRequest) -> EvalBackendResponse:
    url = urljoin(request.base_url.rstrip("/") + "/", "chat/completions")
    messages: list[dict[str, str]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})
    payload = {
        "model": request.model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if request.api_key:
        headers["Authorization"] = f"Bearer {request.api_key}"
    http_request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(http_request, timeout=request.timeout_seconds) as response:
            body = response.read()
        latency_ms = round((perf_counter() - started) * 1000, 3)
        parsed = _json_payload(body)
        usage = _usage_fields(parsed)
        return EvalBackendResponse(
            status="completed",
            output=_completion_text(parsed),
            latency_ms=latency_ms,
            upstream_model=_metadata_string(parsed, "model"),
            **usage,
        )
    except TimeoutError as exc:
        return _error_response("timeout", exc, timed_out=True)
    except HTTPError as exc:
        return EvalBackendResponse(
            status="failed",
            latency_ms=round((perf_counter() - started) * 1000, 3),
            error_type="HTTPError",
            error_message=f"HTTP {exc.code}",
        )
    except (OSError, URLError) as exc:
        return _error_response("failed", exc)


def load_eval_results(path: str | Path = DEFAULT_EVAL_RESULTS_PATH) -> tuple[dict[str, Any], ...]:
    expanded = Path(path).expanduser()
    if not expanded.exists():
        return ()
    rows: list[dict[str, Any]] = []
    try:
        for line in expanded.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(rows)


def eval_report(
    run_id: str,
    *,
    result_path: str | Path = DEFAULT_EVAL_RESULTS_PATH,
) -> EvalRunReport:
    rows = load_eval_results(result_path)
    selected_run_id = _latest_run_id(rows) if run_id == "latest" else run_id
    selected = tuple(
        row for row in rows if selected_run_id is not None and row.get("run_id") == selected_run_id
    )
    completed = sum(1 for row in selected if row.get("status") == "completed")
    passed = sum(1 for row in selected if row.get("exit_status") == "passed")
    failed = sum(1 for row in selected if _row_failed(row))
    timeouts = sum(1 for row in selected if _row_timed_out(row))
    unknown = max(0, len(selected) - passed - failed)
    scores = _numeric_values(selected, "score_percent")
    weighted_scores = _numeric_values(selected, "weighted_score")
    return EvalRunReport(
        run_id=selected_run_id,
        result_path=str(Path(result_path).expanduser()),
        backend=_unique_or_mixed(selected, "backend"),
        model=_unique_or_mixed(selected, "model"),
        selected_engine=_unique_or_mixed(selected, "selected_engine"),
        total=len(selected),
        completed=completed,
        failed=failed,
        passed=passed,
        timeouts=timeouts,
        unknown=unknown,
        score_mean_percent=round(sum(scores) / len(scores), 2) if scores else None,
        weighted_score_mean=(
            round(sum(weighted_scores) / len(weighted_scores), 4)
            if weighted_scores
            else None
        ),
        latency_summary=_latency_summary(selected),
        usage_summary=_usage_summary(selected),
        by_category=_category_summary(selected),
        top_failure_reasons=_top_failure_reasons(selected),
        suitability_notes=_suitability_notes(
            total=len(selected),
            passed=passed,
            failed=failed,
            timeouts=timeouts,
            score_mean_percent=round(sum(scores) / len(scores), 2) if scores else None,
        ),
        privacy=_privacy_summary(),
        results=tuple(_report_row(row) for row in selected),
        notes=(
            "Report excludes raw prompts, request bodies, secrets, and response text.",
            "Scores are local fixture evidence, not universal model rankings.",
        ),
    )


def _run_fixture(
    fixture: EvalFixture,
    *,
    backend: ProxyBackendConfig,
    selected_engine: str | None,
    model: str,
    run_id: str,
    created_at: str,
    timeout_seconds: float,
    runner: Callable[[EvalBackendRequest], EvalBackendResponse],
) -> EvalRunResult:
    started = perf_counter()
    request = EvalBackendRequest(
        backend=backend.name,
        selected_engine=selected_engine,
        model=model,
        base_url=backend.base_url,
        system_prompt=fixture.system_prompt,
        prompt=fixture.prompt,
        timeout_seconds=timeout_seconds,
        api_key=backend.resolved_api_key,
    )
    response = runner(request)
    latency_ms = response.latency_ms
    if latency_ms is None:
        latency_ms = round((perf_counter() - started) * 1000, 3)
    score = score_eval_output(
        fixture,
        response.output,
        status=response.status,
        timed_out=response.timed_out,
        error=response.error_message,
    )
    exit_status = "passed" if response.status == "completed" and score.passed else "failed"
    return EvalRunResult(
        run_id=run_id,
        created_at=created_at,
        backend=backend.name,
        model=model,
        selected_engine=selected_engine,
        fixture_id=fixture.id,
        category=fixture.category,
        score_percent=score.score_percent,
        weighted_score=score.weighted_score,
        exit_status=exit_status,
        status=response.status,
        latency_ms=latency_ms,
        timeout=response.timed_out,
        scorer_version=score.scorer_version,
        fixture_version=EVAL_FIXTURE_SCHEMA_VERSION,
        fixture_prompt_hash=fixture.prompt_hash,
        output_hash=score.output_hash,
        passed_checks=score.passed_checks,
        total_checks=score.total_checks,
        checks=tuple(check.to_dict() for check in score.checks),
        failure_reasons=score.failure_reasons,
        usage_prompt_tokens=response.usage_prompt_tokens,
        usage_completion_tokens=response.usage_completion_tokens,
        usage_total_tokens=response.usage_total_tokens,
        upstream_model=response.upstream_model,
        error_type=response.error_type,
        error_message=response.error_message,
    )


def _fixtures_for_selector(
    selector: str | None,
    *,
    all_fixtures: bool,
) -> tuple[EvalFixture, ...]:
    pack = load_builtin_eval_fixtures()
    if all_fixtures:
        return pack.fixtures
    if not selector:
        raise EvalFixtureError("eval run requires --fixture or --all-fixtures")
    by_id = {fixture.id: fixture for fixture in pack.fixtures}
    if selector in by_id:
        return (by_id[selector],)
    by_category = tuple(
        fixture for fixture in pack.fixtures if fixture.category == selector
    )
    if by_category:
        return by_category
    raise EvalFixtureError(f"unknown eval fixture or category: {selector}")


def _fixture_summary(fixture: EvalFixture) -> dict[str, Any]:
    return {
        "id": fixture.id,
        "name": fixture.name,
        "category": fixture.category,
        "task_profile": fixture.task_profile,
        "privacy_level": fixture.privacy_level,
        "prompt_hash": fixture.prompt_hash,
        "required_patterns": len(fixture.required_patterns),
        "forbidden_patterns": len(fixture.forbidden_patterns),
        "expected_json_keys": list(fixture.expected_json_keys),
        "expected_bullet_count": fixture.expected_bullet_count,
        "max_non_empty_lines": fixture.max_non_empty_lines,
        "delegation_dimensions": dict(fixture.delegation_dimensions),
        "notes": list(fixture.notes),
    }


def _backend(config: RoutingProxyConfig, name: str) -> ProxyBackendConfig:
    backend = config.backends.get(name)
    if backend is None:
        available = ", ".join(sorted(config.backends)) or "none"
        raise ValueError(f"unknown backend {name!r}; available backends: {available}")
    return backend


def _selected_engine_for_backend(
    config: RoutingProxyConfig,
    backend_name: str,
) -> str | None:
    for engine, name in config.engine_backends.items():
        if name == backend_name:
            return engine
    return None


def _append_eval_results(path: Path, results: Sequence[EvalRunResult]) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def _json_payload(body: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _completion_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    output_text = payload.get("output_text")
    return output_text if isinstance(output_text, str) else ""


def _usage_fields(payload: Mapping[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    fields: dict[str, int] = {}
    prompt_tokens = _first_int(usage, ("prompt_tokens", "input_tokens"))
    completion_tokens = _first_int(usage, ("completion_tokens", "output_tokens"))
    total_tokens = _first_int(usage, ("total_tokens",))
    if prompt_tokens is not None:
        fields["usage_prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        fields["usage_completion_tokens"] = completion_tokens
    if total_tokens is not None:
        fields["usage_total_tokens"] = total_tokens
    return fields


def _first_int(payload: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _metadata_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _error_response(
    status: str,
    exc: Exception,
    *,
    timed_out: bool = False,
) -> EvalBackendResponse:
    return EvalBackendResponse(
        status=status,
        timed_out=timed_out,
        error_type=exc.__class__.__name__,
        error_message=_sanitize_error_message(exc),
    )


def _sanitize_error_message(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return exc.__class__.__name__
    return text[:240]


def _latest_run_id(rows: Sequence[Mapping[str, Any]]) -> str | None:
    latest: tuple[str, str] | None = None
    for row in rows:
        run_id = row.get("run_id")
        created_at = row.get("created_at")
        if not isinstance(run_id, str) or not isinstance(created_at, str):
            continue
        candidate = (created_at, run_id)
        if latest is None or candidate > latest:
            latest = candidate
    return latest[1] if latest else None


def _latest_created_at(rows: Sequence[Mapping[str, Any]]) -> str | None:
    timestamps = sorted(
        value
        for row in rows
        if isinstance((value := row.get("created_at")), str) and value.strip()
    )
    return timestamps[-1] if timestamps else None


def _row_matches_model(row: Mapping[str, Any], model: str) -> bool:
    return any(
        value == model
        for value in (
            row.get("model"),
            row.get("upstream_model"),
            row.get("backend_model"),
            row.get("selected_model"),
        )
        if isinstance(value, str)
    )


def _not_evaluated_summary(
    *,
    model: str,
    backend: str | None,
) -> dict[str, Any]:
    return {
        "status": "not_evaluated",
        "model": model,
        "backend": backend,
        "selected_engine": None,
        "latest_run_id": None,
        "last_evaluated_at": None,
        "fixture_count": 0,
        "completed": 0,
        "passed": 0,
        "failed": 0,
        "timeouts": 0,
        "score_mean_percent": None,
        "weighted_score_mean": None,
        "by_category": {},
        "top_failure_reasons": [],
        "latency_summary": _latency_summary(()),
        "usage_summary": _usage_summary(()),
        "stale": True,
        "stale_reasons": ["No cached eval evidence for this model."],
        "privacy": _privacy_summary(),
        "notes": [
            "Model has not been evaluated on the cached fixture set.",
            "Missing eval evidence does not block routing.",
        ],
    }


def _staleness(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not rows:
        return True, ["No cached eval evidence for this model."]
    if not _latest_created_at(rows):
        reasons.append("Missing eval timestamp.")
    fixture_versions = {
        value
        for row in rows
        if isinstance((value := row.get("fixture_version")), int)
        and not isinstance(value, bool)
    }
    scorer_versions = {
        value
        for row in rows
        if isinstance((value := row.get("scorer_version")), int)
        and not isinstance(value, bool)
    }
    if fixture_versions != {EVAL_FIXTURE_SCHEMA_VERSION}:
        reasons.append(
            f"Fixture version mismatch; expected {EVAL_FIXTURE_SCHEMA_VERSION}."
        )
    if scorer_versions != {EVAL_SCORER_VERSION}:
        reasons.append(f"Scorer version mismatch; expected {EVAL_SCORER_VERSION}.")
    return bool(reasons), reasons


def _row_failed(row: Mapping[str, Any]) -> bool:
    if row.get("exit_status") == "failed":
        return True
    return row.get("status") in {"failed", "timeout", "error"}


def _row_timed_out(row: Mapping[str, Any]) -> bool:
    return row.get("timeout") is True or row.get("status") == "timeout"


def _numeric_values(
    rows: Sequence[Mapping[str, Any]],
    field_name: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value < 0:
            continue
        values.append(float(value))
    return values


def _unique_or_mixed(
    rows: Sequence[Mapping[str, Any]],
    field_name: str,
) -> str | None:
    values = sorted(
        {
            value.strip()
            for row in rows
            if isinstance((value := row.get(field_name)), str) and value.strip()
        }
    )
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return "mixed"


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(_numeric_values(rows, "latency_ms"))
    if not values:
        return {
            "count": 0,
            "missing": len(rows),
            "min_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "median_ms": None,
        }
    return {
        "count": len(values),
        "missing": max(0, len(rows) - len(values)),
        "min_ms": round(values[0], 3),
        "max_ms": round(values[-1], 3),
        "mean_ms": round(sum(values) / len(values), 3),
        "median_ms": _median(values),
    }


def _median(values: Sequence[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2:
        return round(values[midpoint], 3)
    return round((values[midpoint - 1] + values[midpoint]) / 2, 3)


def _usage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        "usage_prompt_tokens": 0,
        "usage_completion_tokens": 0,
        "usage_total_tokens": 0,
        "usage_cached_input_tokens": 0,
    }
    rows_with_usage = 0
    upstream_models: Counter[str] = Counter()
    for row in rows:
        usage = {
            field: _non_negative_int(row.get(field))
            for field in totals
        }
        if any(value > 0 for value in usage.values()):
            rows_with_usage += 1
            for field, value in usage.items():
                totals[field] += value
        upstream_model = row.get("upstream_model")
        if isinstance(upstream_model, str) and upstream_model.strip():
            upstream_models[upstream_model.strip()] += 1
    return {
        "rows_with_usage": rows_with_usage,
        "rows_missing_usage": max(0, len(rows) - rows_with_usage),
        **totals,
        "upstream_model_counts": dict(sorted(upstream_models.items())),
    }


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _category_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = row.get("category")
        if not isinstance(category, str) or not category:
            category = "unknown"
        group = grouped.setdefault(
            category,
            {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "timeouts": 0,
                "score_mean_percent": None,
                "weighted_score_mean": None,
            },
        )
        group["total"] += 1
        if row.get("exit_status") == "passed":
            group["passed"] += 1
        elif _row_failed(row):
            group["failed"] += 1
        if _row_timed_out(row):
            group["timeouts"] += 1
    for category, group in grouped.items():
        scores = [
            float(row["score_percent"])
            for row in rows
            if row.get("category") == category
            and isinstance(row.get("score_percent"), (int, float))
        ]
        weighted_scores = [
            float(row["weighted_score"])
            for row in rows
            if row.get("category") == category
            and isinstance(row.get("weighted_score"), (int, float))
        ]
        group["score_mean_percent"] = (
            round(sum(scores) / len(scores), 2) if scores else None
        )
        group["weighted_score_mean"] = (
            round(sum(weighted_scores) / len(weighted_scores), 4)
            if weighted_scores
            else None
        )
    return dict(sorted(grouped.items()))


def _top_failure_reasons(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 8,
) -> tuple[dict[str, Any], ...]:
    counts: Counter[str] = Counter()
    for row in rows:
        reasons = row.get("failure_reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                if isinstance(reason, str) and reason.strip():
                    counts[_sanitize_report_text(reason)] += 1
        elif _row_failed(row):
            error_type = row.get("error_type")
            if isinstance(error_type, str) and error_type.strip():
                counts[_sanitize_report_text(error_type)] += 1
            else:
                counts["Eval did not pass."] += 1
    return tuple(
        {"reason": reason, "count": count}
        for reason, count in counts.most_common(limit)
    )


def _sanitize_report_text(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()[:240]


def _suitability_notes(
    *,
    total: int,
    passed: int,
    failed: int,
    timeouts: int,
    score_mean_percent: float | None,
) -> tuple[str, ...]:
    notes = [
        "Interpret results as best on this fixture set/profile, not as a universal model ranking.",
    ]
    if total == 0:
        notes.append("No eval results found for this run id.")
        return tuple(notes)
    if passed == total:
        notes.append("All reported fixtures passed for this run.")
    if failed:
        notes.append(
            "Failed fixtures identify local suitability gaps to inspect before changing policy."
        )
    if timeouts:
        notes.append(
            "Timeouts suggest this backend may be unsuitable for latency-sensitive tasks in this fixture set."
        )
    if score_mean_percent is None:
        notes.append("Some rows lack score fields; old or partial rows were tolerated.")
    return tuple(notes)


def _privacy_summary() -> dict[str, Any]:
    return {
        "prompt_retention": "hash_only",
        "output_retention": "hash_only",
        "artifact_retention": "disabled_by_default",
        "raw_prompts_retained": False,
        "raw_outputs_retained": False,
        "secrets_retained": False,
        "report_excludes": [
            "raw prompts",
            "request bodies",
            "raw model outputs",
            "response text",
            "secrets",
        ],
    }


def _report_row(row: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "run_id",
        "created_at",
        "backend",
        "model",
        "selected_engine",
        "fixture_id",
        "category",
        "score_percent",
        "weighted_score",
        "exit_status",
        "status",
        "latency_ms",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "usage_total_tokens",
        "timeout",
        "error_type",
        "error_message",
        "scorer_version",
        "fixture_version",
        "fixture_prompt_hash",
        "output_hash",
        "passed_checks",
        "total_checks",
        "failure_reasons",
        "upstream_model",
    }
    return {key: value for key, value in row.items() if key in allowed}


def _new_run_id(created_at: str) -> str:
    safe = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace(".", "")
        .replace("Z", "Z")
    )
    return f"evalrun_{safe}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
