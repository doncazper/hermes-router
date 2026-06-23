"""Offline workflow routing correctness benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

from hermes.plugins.model_router.config import load_router_config
from hermes.plugins.model_router.policy import ModelRouter
from hermes.plugins.model_router.receipts import decision_to_receipt


WORKFLOW_BENCHMARK_VERSION = 1


@dataclass(frozen=True)
class WorkflowBenchmarkCase:
    name: str
    category: str
    prompt: str
    expected_engine: str
    hints: Mapping[str, Any] = field(default_factory=dict)
    expected_provider: str | None = None
    expected_requires_confirmation: bool | None = None

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "expected_engine": self.expected_engine,
            "expected_provider": self.expected_provider,
            "expected_requires_confirmation": self.expected_requires_confirmation,
            "hints": dict(self.hints),
            "prompt_hash": self.prompt_hash,
        }


@dataclass(frozen=True)
class WorkflowBenchmarkResult:
    name: str
    category: str
    expected_engine: str
    selected_engine: str
    passed: bool
    prompt_hash: str
    route_latency_us: float
    routing_profile: str
    selected_provider: str | None
    expected_provider: str | None = None
    expected_requires_confirmation: bool | None = None
    requires_confirmation: bool = False
    fallback_used: bool = False
    fallback_engine: str | None = None
    summary: str = ""
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    policy_explanation: str = ""
    fallback_explanation: str = ""
    safety_explanation: str = ""
    privacy_explanation: str = ""
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def route_changed(self) -> bool:
        return self.selected_engine != self.expected_engine

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "expected_engine": self.expected_engine,
            "selected_engine": self.selected_engine,
            "passed": self.passed,
            "route_changed": self.route_changed,
            "prompt_hash": self.prompt_hash,
            "route_latency_us": self.route_latency_us,
            "routing_profile": self.routing_profile,
            "selected_provider": self.selected_provider,
            "expected_provider": self.expected_provider,
            "expected_requires_confirmation": self.expected_requires_confirmation,
            "requires_confirmation": self.requires_confirmation,
            "fallback_used": self.fallback_used,
            "fallback_engine": self.fallback_engine,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
            "policy_explanation": self.policy_explanation,
            "fallback_explanation": self.fallback_explanation,
            "safety_explanation": self.safety_explanation,
            "privacy_explanation": self.privacy_explanation,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class WorkflowBenchmarkReport:
    version: int
    generated_at: str
    config_path: str | None
    total: int
    passed: int
    failed: int
    route_changes: int
    results: tuple[WorkflowBenchmarkResult, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return round(self.passed / self.total, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "config_path": self.config_path,
            "ok": self.ok,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "route_changes": self.route_changes,
            "results": [result.to_dict() for result in self.results],
            "notes": list(self.notes),
        }


DEFAULT_WORKFLOW_CASES: tuple[WorkflowBenchmarkCase, ...] = (
    WorkflowBenchmarkCase(
        name="simple_rewrite",
        category="simple",
        prompt="Rewrite this sentence to be clear and concise: routing stays explicit.",
        expected_engine="fast_local",
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
    WorkflowBenchmarkCase(
        name="ordinary_summary",
        category="balanced",
        prompt="Summarize these release notes into three concise bullets.",
        expected_engine="balanced_local",
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
    WorkflowBenchmarkCase(
        name="repo_test_work",
        category="coding",
        prompt="Fix the failing Python test in this repository and run pytest.",
        expected_engine="code_agent",
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
    WorkflowBenchmarkCase(
        name="current_information",
        category="research",
        prompt="Research the latest local AI runtime setup guidance and cite sources.",
        expected_engine="web_research",
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
    WorkflowBenchmarkCase(
        name="screenshot_ocr",
        category="vision",
        prompt="Extract the text from this screenshot and summarize the chart.",
        expected_engine="multimodal_vision",
        hints={"attachments": ["image"]},
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
    WorkflowBenchmarkCase(
        name="image_generation",
        category="image_generation",
        prompt="Generate an image of a calm local infrastructure dashboard.",
        expected_engine="image_generation",
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
    WorkflowBenchmarkCase(
        name="risky_external_action",
        category="safety",
        prompt="Deploy this change to production and push the branch.",
        expected_engine="human_confirm",
        expected_provider="human",
        expected_requires_confirmation=True,
    ),
    WorkflowBenchmarkCase(
        name="private_current_information",
        category="profile",
        prompt="Research current local AI runtime setup options.",
        expected_engine="web_research",
        hints={"profile": "private"},
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
    WorkflowBenchmarkCase(
        name="quality_reasoning",
        category="profile",
        prompt=(
            "Design an architecture migration plan with risks, rollout, "
            "and testing strategy."
        ),
        expected_engine="reasoning_local",
        hints={"profile": "quality"},
        expected_provider="local",
        expected_requires_confirmation=False,
    ),
)


def run_workflow_benchmarks(
    *,
    config_path: str | Path | None = None,
    cases: Sequence[WorkflowBenchmarkCase] = DEFAULT_WORKFLOW_CASES,
) -> WorkflowBenchmarkReport:
    config = load_router_config(config_path)
    router = ModelRouter.from_config_object(config, validate_availability=False)
    results = tuple(_run_case(router, case) for case in cases)
    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    route_changes = sum(1 for result in results if result.route_changed)
    source_path = config.source_path or (str(config_path) if config_path else None)
    return WorkflowBenchmarkReport(
        version=WORKFLOW_BENCHMARK_VERSION,
        generated_at=_now_iso(),
        config_path=source_path,
        total=len(results),
        passed=passed,
        failed=failed,
        route_changes=route_changes,
        results=results,
        notes=(
            "Offline routing correctness only; no backend requests were made.",
            "Prompt bodies are fixture inputs and are not serialized in reports.",
        ),
    )


def workflow_cases_by_name(
    names: Sequence[str] | None = None,
) -> tuple[WorkflowBenchmarkCase, ...]:
    if not names:
        return DEFAULT_WORKFLOW_CASES
    selected = set(names)
    return tuple(case for case in DEFAULT_WORKFLOW_CASES if case.name in selected)


def workflow_case_names() -> tuple[str, ...]:
    return tuple(case.name for case in DEFAULT_WORKFLOW_CASES)


def _run_case(
    router: ModelRouter,
    case: WorkflowBenchmarkCase,
) -> WorkflowBenchmarkResult:
    started = perf_counter_ns()
    decision = router.route(case.prompt, hints=dict(case.hints))
    route_latency_us = round((perf_counter_ns() - started) / 1000, 3)
    receipt = decision_to_receipt(decision)
    selected_engine = router.config.get_engine(decision.selected_engine)
    selected_provider = selected_engine.provider if selected_engine else None
    failure_reasons = _failure_reasons(
        case,
        decision.selected_engine,
        selected_provider,
        decision.requires_confirmation,
    )
    return WorkflowBenchmarkResult(
        name=case.name,
        category=case.category,
        expected_engine=case.expected_engine,
        selected_engine=decision.selected_engine,
        passed=not failure_reasons,
        prompt_hash=case.prompt_hash,
        route_latency_us=route_latency_us,
        routing_profile=decision.routing_profile.value,
        selected_provider=selected_provider,
        expected_provider=case.expected_provider,
        expected_requires_confirmation=case.expected_requires_confirmation,
        requires_confirmation=decision.requires_confirmation,
        fallback_used=decision.fallback_used,
        fallback_engine=decision.fallback_engine,
        summary=receipt.summary,
        reason_codes=receipt.reason_codes,
        policy_explanation=receipt.policy_explanation,
        fallback_explanation=receipt.fallback_explanation,
        safety_explanation=receipt.safety_explanation,
        privacy_explanation=receipt.privacy_explanation,
        failure_reasons=tuple(failure_reasons),
    )


def _failure_reasons(
    case: WorkflowBenchmarkCase,
    selected_engine: str,
    selected_provider: str | None,
    requires_confirmation: bool,
) -> list[str]:
    failures: list[str] = []
    if selected_engine != case.expected_engine:
        failures.append(
            f"expected engine {case.expected_engine}, selected {selected_engine}"
        )
    if case.expected_provider is not None and selected_provider != case.expected_provider:
        failures.append(
            f"expected provider {case.expected_provider}, selected {selected_provider}"
        )
    if (
        case.expected_requires_confirmation is not None
        and requires_confirmation is not case.expected_requires_confirmation
    ):
        failures.append(
            "expected confirmation "
            f"{case.expected_requires_confirmation}, got {requires_confirmation}"
        )
    return failures


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
