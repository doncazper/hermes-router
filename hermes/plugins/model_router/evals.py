"""ModelRouter eval fixture schema and built-in fixture loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

from hermes.plugins.model_router.product import PRODUCT_DATA_PACKAGE


EVAL_FIXTURE_SCHEMA_VERSION = 1
DEFAULT_EVAL_FIXTURE_CATALOG = "eval_fixtures.yaml"
EVAL_CATEGORIES = (
    "structured_output",
    "no_reasoning_leakage",
    "mechanical_edit",
    "code_review_judgment",
    "risky_action_refusal",
    "verification_heavy_task",
    "slow_long_context_task",
)
DELEGATION_DIMENSIONS = (
    "mechanical_work_likely",
    "judgment_heavy_likely",
    "verification_heavy_likely",
    "repo_wide_likely",
    "risky_or_external_action",
    "ambiguity_sensitive",
)
PRIVACY_LEVELS = ("hash_only", "redacted_preview", "local_full")
EVAL_SCORER_VERSION = 1
SUCCESS_STATUSES = ("completed", "passed")
REASONING_LEAKAGE_PATTERNS = (
    r"(?i)chain[- ]of[- ]thought",
    r"(?i)scratchpad",
    r"(?i)hidden reasoning",
    r"(?i)let me think step by step",
    r"(?i)<analysis>",
    r"(?i)</analysis>",
)


class EvalFixtureError(ValueError):
    """Raised when eval fixture metadata is invalid."""


@dataclass(frozen=True)
class EvalFixture:
    id: str
    name: str
    category: str
    task_profile: str
    prompt: str
    required_patterns: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]
    expected_json_keys: tuple[str, ...]
    expected_bullet_count: int | None
    max_non_empty_lines: int | None
    weight: float
    privacy_level: str
    delegation_dimensions: Mapping[str, bool]
    notes: tuple[str, ...]
    system_prompt: str | None = None

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "task_profile": self.task_profile,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "required_patterns": list(self.required_patterns),
            "forbidden_patterns": list(self.forbidden_patterns),
            "expected_json_keys": list(self.expected_json_keys),
            "expected_bullet_count": self.expected_bullet_count,
            "max_non_empty_lines": self.max_non_empty_lines,
            "weight": self.weight,
            "privacy_level": self.privacy_level,
            "delegation_dimensions": dict(self.delegation_dimensions),
            "notes": list(self.notes),
            "prompt_hash": self.prompt_hash,
        }


@dataclass(frozen=True)
class EvalFixturePack:
    version: int
    fixture_pack_id: str
    fixture_pack_version: int
    fixtures: tuple[EvalFixture, ...] = field(default_factory=tuple)
    source: str = "unknown"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fixture_pack_id": self.fixture_pack_id,
            "fixture_pack_version": self.fixture_pack_version,
            "source": self.source,
            "fixtures": [fixture.to_dict() for fixture in self.fixtures],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EvalCheckResult:
    id: str
    passed: bool
    weight: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "passed": self.passed,
            "weight": self.weight,
            "message": self.message,
        }


@dataclass(frozen=True)
class EvalScoreResult:
    fixture_id: str
    score_percent: float
    passed_checks: int
    total_checks: int
    checks: tuple[EvalCheckResult, ...]
    failure_reasons: tuple[str, ...]
    weighted_score: float
    scorer_version: int
    status: str
    output_hash: str

    @property
    def passed(self) -> bool:
        return not self.failure_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "score_percent": self.score_percent,
            "passed": self.passed,
            "passed_checks": self.passed_checks,
            "total_checks": self.total_checks,
            "checks": [check.to_dict() for check in self.checks],
            "failure_reasons": list(self.failure_reasons),
            "weighted_score": self.weighted_score,
            "scorer_version": self.scorer_version,
            "status": self.status,
            "output_hash": self.output_hash,
        }


def load_builtin_eval_fixtures() -> EvalFixturePack:
    resource = resources.files(PRODUCT_DATA_PACKAGE).joinpath(
        DEFAULT_EVAL_FIXTURE_CATALOG,
    )
    return eval_fixture_pack_from_text(
        resource.read_text(encoding="utf-8"),
        source=f"packaged:{DEFAULT_EVAL_FIXTURE_CATALOG}",
    )


def score_eval_output(
    fixture: EvalFixture,
    output: str | None,
    *,
    status: str = "completed",
    timed_out: bool = False,
    error: str | None = None,
) -> EvalScoreResult:
    """Score one eval output with deterministic local checks only."""

    text = output or ""
    parsed_json = _parse_json_object(text)
    checks = _score_checks(
        fixture,
        text,
        status=status,
        timed_out=timed_out,
        error=error,
        parsed_json=parsed_json,
    )
    total_weight = sum(check.weight for check in checks)
    passed_weight = sum(check.weight for check in checks if check.passed)
    weighted_score = round(passed_weight / total_weight, 4) if total_weight else 0.0
    return EvalScoreResult(
        fixture_id=fixture.id,
        score_percent=round(weighted_score * 100, 2),
        passed_checks=sum(1 for check in checks if check.passed),
        total_checks=len(checks),
        checks=tuple(checks),
        failure_reasons=tuple(check.message for check in checks if not check.passed),
        weighted_score=weighted_score,
        scorer_version=EVAL_SCORER_VERSION,
        status="timeout" if timed_out else status,
        output_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def load_eval_fixture_pack(path: str | Path) -> EvalFixturePack:
    expanded = Path(path).expanduser()
    return eval_fixture_pack_from_text(
        expanded.read_text(encoding="utf-8"),
        source=str(expanded),
    )


def _score_checks(
    fixture: EvalFixture,
    text: str,
    *,
    status: str,
    timed_out: bool,
    error: str | None,
    parsed_json: Mapping[str, Any] | None,
) -> tuple[EvalCheckResult, ...]:
    checks: list[EvalCheckResult] = []
    normalized_status = status.strip().lower() if status else ""
    status_ok = not timed_out and normalized_status in SUCCESS_STATUSES and not error
    checks.append(
        EvalCheckResult(
            id="status_ok",
            passed=status_ok,
            weight=1.0,
            message=(
                "Eval execution completed."
                if status_ok
                else f"Eval execution did not complete cleanly: {normalized_status or 'unknown'}."
            ),
        )
    )
    output_present = bool(text.strip())
    checks.append(
        EvalCheckResult(
            id="non_empty_output",
            passed=output_present,
            weight=1.0,
            message="Output was non-empty." if output_present else "Output was empty.",
        )
    )
    for index, pattern in enumerate(fixture.required_patterns):
        passed = re.search(pattern, text) is not None
        checks.append(
            EvalCheckResult(
                id=f"required_pattern_{index}",
                passed=passed,
                weight=1.0,
                message=(
                    f"Required pattern {index} was present."
                    if passed
                    else f"Required pattern {index} was missing."
                ),
            )
        )
    for index, pattern in enumerate(fixture.forbidden_patterns):
        passed = re.search(pattern, text) is None
        checks.append(
            EvalCheckResult(
                id=f"forbidden_pattern_{index}",
                passed=passed,
                weight=1.0,
                message=(
                    f"Forbidden pattern {index} was absent."
                    if passed
                    else f"Forbidden pattern {index} was present."
                ),
            )
        )
    if fixture.expected_json_keys:
        valid_json = parsed_json is not None
        checks.append(
            EvalCheckResult(
                id="valid_json",
                passed=valid_json,
                weight=1.0,
                message=(
                    "Output parsed as a JSON object."
                    if valid_json
                    else "Output did not parse as a JSON object."
                ),
            )
        )
        actual_keys = set(parsed_json or {})
        expected_keys = set(fixture.expected_json_keys)
        exact_keys = valid_json and actual_keys == expected_keys
        checks.append(
            EvalCheckResult(
                id="exact_json_keys",
                passed=exact_keys,
                weight=1.0,
                message=(
                    "JSON keys matched exactly."
                    if exact_keys
                    else "JSON keys did not match expected keys exactly."
                ),
            )
        )
    if fixture.expected_bullet_count is not None:
        bullet_count = _bullet_count(text)
        expected = fixture.expected_bullet_count
        checks.append(
            EvalCheckResult(
                id="expected_bullet_count",
                passed=bullet_count == expected,
                weight=1.0,
                message=(
                    f"Output had expected bullet count {expected}."
                    if bullet_count == expected
                    else f"Expected {expected} bullets, found {bullet_count}."
                ),
            )
        )
    if fixture.max_non_empty_lines is not None:
        non_empty_lines = _non_empty_line_count(text)
        max_lines = fixture.max_non_empty_lines
        checks.append(
            EvalCheckResult(
                id="max_non_empty_lines",
                passed=non_empty_lines <= max_lines,
                weight=1.0,
                message=(
                    f"Output used {non_empty_lines} non-empty lines within limit {max_lines}."
                    if non_empty_lines <= max_lines
                    else f"Output used {non_empty_lines} non-empty lines over limit {max_lines}."
                ),
            )
        )
    leakage = [
        pattern for pattern in REASONING_LEAKAGE_PATTERNS if re.search(pattern, text)
    ]
    checks.append(
        EvalCheckResult(
            id="reasoning_leakage_absent",
            passed=not leakage,
            weight=1.0,
            message=(
                "No reasoning leakage markers were detected."
                if not leakage
                else "Reasoning leakage markers were detected."
            ),
        )
    )
    return tuple(checks)


def _parse_json_object(text: str) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _bullet_count(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)
    )


def _non_empty_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def eval_fixture_pack_from_text(text: str, *, source: str = "<memory>") -> EvalFixturePack:
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise EvalFixtureError(f"{source}: invalid YAML: {exc}") from exc
    return eval_fixture_pack_from_mapping(payload, source=source)


def eval_fixture_pack_from_mapping(
    payload: Mapping[str, Any],
    *,
    source: str = "<mapping>",
) -> EvalFixturePack:
    if not isinstance(payload, Mapping):
        raise EvalFixtureError(f"{source}: fixture pack must be a mapping")
    version = _positive_int(payload.get("version"), "version", source=source)
    if version != EVAL_FIXTURE_SCHEMA_VERSION:
        raise EvalFixtureError(
            f"{source}: unsupported eval fixture schema version {version}"
        )
    fixture_pack_id = _required_string(
        payload.get("fixture_pack_id"),
        "fixture_pack_id",
        source=source,
    )
    fixture_pack_version = _positive_int(
        payload.get("fixture_pack_version"),
        "fixture_pack_version",
        source=source,
    )
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, Sequence) or isinstance(raw_fixtures, (str, bytes)):
        raise EvalFixtureError(f"{source}: fixtures must be a list")
    fixtures = tuple(
        _fixture_from_mapping(item, source=f"{source}:fixtures[{index}]")
        for index, item in enumerate(raw_fixtures)
    )
    if not fixtures:
        raise EvalFixtureError(f"{source}: fixtures must not be empty")
    ids = [fixture.id for fixture in fixtures]
    duplicates = sorted({fixture_id for fixture_id in ids if ids.count(fixture_id) > 1})
    if duplicates:
        raise EvalFixtureError(
            f"{source}: duplicate fixture ids: {', '.join(duplicates)}"
        )
    return EvalFixturePack(
        version=version,
        fixture_pack_id=fixture_pack_id,
        fixture_pack_version=fixture_pack_version,
        fixtures=fixtures,
        source=source,
        notes=_string_tuple(payload.get("notes", ()), "notes", source=source),
    )


def _fixture_from_mapping(payload: Any, *, source: str) -> EvalFixture:
    if not isinstance(payload, Mapping):
        raise EvalFixtureError(f"{source}: fixture must be a mapping")
    fixture_id = _required_string(payload.get("id"), "id", source=source)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", fixture_id):
        raise EvalFixtureError(f"{source}: id must be lowercase slug text")
    category = _required_string(payload.get("category"), "category", source=source)
    if category not in EVAL_CATEGORIES:
        raise EvalFixtureError(f"{source}: unsupported category {category!r}")
    required_patterns = _string_tuple(
        payload.get("required_patterns"),
        "required_patterns",
        source=source,
    )
    forbidden_patterns = _string_tuple(
        payload.get("forbidden_patterns"),
        "forbidden_patterns",
        source=source,
    )
    _validate_regexes(required_patterns, "required_patterns", source=source)
    _validate_regexes(forbidden_patterns, "forbidden_patterns", source=source)
    privacy_level = _required_string(
        payload.get("privacy_level"),
        "privacy_level",
        source=source,
    )
    if privacy_level not in PRIVACY_LEVELS:
        raise EvalFixtureError(f"{source}: unsupported privacy_level {privacy_level!r}")
    return EvalFixture(
        id=fixture_id,
        name=_required_string(payload.get("name"), "name", source=source),
        category=category,
        task_profile=_required_string(
            payload.get("task_profile"),
            "task_profile",
            source=source,
        ),
        prompt=_required_string(payload.get("prompt"), "prompt", source=source),
        system_prompt=_optional_string(
            payload.get("system_prompt"),
            "system_prompt",
            source=source,
        ),
        required_patterns=required_patterns,
        forbidden_patterns=forbidden_patterns,
        expected_json_keys=_string_tuple(
            payload.get("expected_json_keys"),
            "expected_json_keys",
            source=source,
        ),
        expected_bullet_count=_optional_positive_int(
            payload.get("expected_bullet_count"),
            "expected_bullet_count",
            source=source,
        ),
        max_non_empty_lines=_optional_positive_int(
            payload.get("max_non_empty_lines"),
            "max_non_empty_lines",
            source=source,
        ),
        weight=_positive_float(payload.get("weight"), "weight", source=source),
        privacy_level=privacy_level,
        delegation_dimensions=_delegation_dimensions(
            payload.get("delegation_dimensions"),
            source=source,
        ),
        notes=_string_tuple(payload.get("notes"), "notes", source=source),
    )


def _required_string(value: Any, field_name: str, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalFixtureError(f"{source}: {field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, field_name: str, *, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvalFixtureError(f"{source}: {field_name} must be a string or null")
    return value.strip() or None


def _string_tuple(value: Any, field_name: str, *, source: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvalFixtureError(f"{source}: {field_name} must be a list of strings")
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise EvalFixtureError(
                f"{source}: {field_name}[{index}] must be a non-empty string"
            )
        strings.append(item.strip())
    return tuple(strings)


def _positive_int(value: Any, field_name: str, *, source: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise EvalFixtureError(f"{source}: {field_name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, field_name: str, *, source: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise EvalFixtureError(
            f"{source}: {field_name} must be a positive integer or null"
        )
    return value


def _positive_float(value: Any, field_name: str, *, source: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise EvalFixtureError(f"{source}: {field_name} must be a positive number")
    return float(value)


def _delegation_dimensions(value: Any, *, source: str) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise EvalFixtureError(f"{source}: delegation_dimensions must be a mapping")
    dimensions: dict[str, bool] = {}
    for dimension in DELEGATION_DIMENSIONS:
        raw = value.get(dimension)
        if not isinstance(raw, bool):
            raise EvalFixtureError(
                f"{source}: delegation_dimensions.{dimension} must be a boolean"
            )
        dimensions[dimension] = raw
    unknown = sorted(set(value) - set(DELEGATION_DIMENSIONS))
    if unknown:
        raise EvalFixtureError(
            f"{source}: unknown delegation dimensions: {', '.join(unknown)}"
        )
    return dimensions


def _validate_regexes(patterns: tuple[str, ...], field_name: str, *, source: str) -> None:
    for index, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise EvalFixtureError(
                f"{source}: {field_name}[{index}] is not valid regex: {exc}"
            ) from exc
