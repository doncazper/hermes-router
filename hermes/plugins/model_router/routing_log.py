"""Privacy-aware JSONL routing logs for hindsight evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from hermes.plugins.model_router.models import RoutingDecision


DEFAULT_LOG_PATH = "~/.model-router/routing-events.jsonl"
DEFAULT_FEEDBACK_PATH = "~/.model-router/routing-feedback.jsonl"
PROMPT_CAPTURE_OFF = "off"
PROMPT_CAPTURE_REDACTED = "redacted_preview"
PROMPT_CAPTURE_FULL = "full"
PROMPT_CAPTURE_MODES = (
    PROMPT_CAPTURE_OFF,
    PROMPT_CAPTURE_REDACTED,
    PROMPT_CAPTURE_FULL,
)

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)"
        r"\s*[:=]\s*([^\s,;]+)"
    ),
    re.compile(r"\b(?:sk|pk|hf|ghp|github_pat)_[A-Za-z0-9_=-]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}\b"),
)


@dataclass(frozen=True)
class RoutingEvent:
    event_type: str
    timestamp: str
    request_id: str
    route_api: str
    selected_engine: str
    status: str
    route_latency_ms: float
    diagnostic_latency_ms: float | None
    upstream_latency_ms: float | None
    total_latency_ms: float
    fallback_used: bool
    config_source: str
    router_version: str
    prompt_hash: str
    prompt_length: int
    estimated_tokens: int
    prompt_preview: str | None = None
    prompt: str | None = None
    backend: str | None = None
    backend_model: str | None = None
    status_code: int | None = None
    complexity_score: int | None = None
    risk_score: int | None = None
    confidence_score: int | None = None
    features: dict[str, Any] | None = None
    reasons: tuple[str, ...] = ()
    requirements: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class RoutingFeedback:
    event_type: str
    timestamp: str
    request_id: str
    expected_engine: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


class RoutingLogWriter:
    """Best-effort JSONL writer; logging failures never affect routing."""

    def __init__(
        self,
        path: str | Path = DEFAULT_LOG_PATH,
        *,
        max_bytes: int = 0,
        backups: int = 0,
    ) -> None:
        self.path = _expand_path(path)
        self.max_bytes = max(0, int(max_bytes))
        self.backups = max(0, int(backups))

    def write(self, payload: dict[str, Any] | RoutingEvent | RoutingFeedback) -> bool:
        try:
            data = payload.to_dict() if hasattr(payload, "to_dict") else dict(payload)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with self.path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(json.dumps(data, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            return True
        except Exception:
            return False

    def _rotate_if_needed(self) -> None:
        if self.max_bytes <= 0 or self.backups <= 0 or not self.path.exists():
            return
        if self.path.stat().st_size < self.max_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backups - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()


def estimated_tokens(prompt: str) -> int:
    text = prompt or ""
    return (len(text) + 3) // 4 if text else 0


def redact_text(text: str) -> str:
    redacted = text or ""
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_redaction_replacement, redacted)
    return redacted


def prompt_capture_mode(configured: str = PROMPT_CAPTURE_REDACTED) -> str:
    if _env_truthy("MODEL_ROUTER_LOG_PROMPTS"):
        return PROMPT_CAPTURE_FULL
    if configured not in PROMPT_CAPTURE_MODES:
        return PROMPT_CAPTURE_REDACTED
    return configured


def prompt_fields(
    prompt: str,
    *,
    capture: str = PROMPT_CAPTURE_REDACTED,
    preview_chars: int = 180,
) -> dict[str, Any]:
    mode = prompt_capture_mode(capture)
    fields: dict[str, Any] = {
        "prompt_hash": prompt_hash(prompt),
        "prompt_length": len(prompt or ""),
        "estimated_tokens": estimated_tokens(prompt),
    }
    if mode in {PROMPT_CAPTURE_REDACTED, PROMPT_CAPTURE_FULL}:
        fields["prompt_preview"] = redact_text(prompt or "")[:preview_chars]
    if mode == PROMPT_CAPTURE_FULL:
        fields["prompt"] = prompt or ""
    return fields


def build_routing_event(
    *,
    request_id: str,
    route_api: str,
    selected_engine: str,
    status: str,
    prompt: str,
    route_latency_ms: float,
    total_latency_ms: float,
    config_source: str,
    router_version: str,
    fallback_used: bool = False,
    upstream_latency_ms: float | None = None,
    diagnostic_latency_ms: float | None = None,
    backend: str | None = None,
    backend_model: str | None = None,
    status_code: int | None = None,
    decision: RoutingDecision | None = None,
    prompt_capture: str = PROMPT_CAPTURE_REDACTED,
) -> RoutingEvent:
    prompt_data = prompt_fields(prompt, capture=prompt_capture)
    return RoutingEvent(
        event_type="routing_event",
        timestamp=now_iso(),
        request_id=request_id,
        route_api=route_api,
        selected_engine=selected_engine,
        status=status,
        route_latency_ms=round(route_latency_ms, 4),
        diagnostic_latency_ms=(
            round(diagnostic_latency_ms, 4)
            if diagnostic_latency_ms is not None
            else None
        ),
        upstream_latency_ms=(
            round(upstream_latency_ms, 4) if upstream_latency_ms is not None else None
        ),
        total_latency_ms=round(total_latency_ms, 4),
        fallback_used=fallback_used,
        config_source=config_source,
        router_version=router_version,
        backend=backend,
        backend_model=backend_model,
        status_code=status_code,
        complexity_score=decision.complexity_score if decision else None,
        risk_score=decision.risk_score if decision else None,
        confidence_score=decision.confidence_score if decision else None,
        features=decision.features.to_dict() if decision else None,
        reasons=decision.reasons if decision else (),
        requirements=decision.requirements.to_dict() if decision else None,
        **prompt_data,
    )


def build_feedback(
    *,
    request_id: str,
    expected_engine: str,
    notes: str | None = None,
) -> RoutingFeedback:
    return RoutingFeedback(
        event_type="routing_feedback",
        timestamp=now_iso(),
        request_id=request_id,
        expected_engine=expected_engine,
        notes=notes,
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    expanded = _expand_path(path)
    if not expanded.exists():
        return []
    rows: list[dict[str, Any]] = []
    with expanded.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _redaction_replacement(match: re.Match[str]) -> str:
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}=[REDACTED]"
    text = match.group(0)
    if text.lower().startswith("bearer "):
        return "Bearer [REDACTED]"
    return "[REDACTED]"


def _expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
