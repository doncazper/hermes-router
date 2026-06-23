import json
import subprocess
import sys
from pathlib import Path

import yaml

from hermes.plugins.model_router.config import REQUIRED_ENGINE_CATEGORIES
from hermes.plugins.model_router.workflow_benchmark import (
    WorkflowBenchmarkCase,
    run_workflow_benchmarks,
)


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _engine(
    name: str,
    *,
    enabled: bool = True,
    fallback: str | None = None,
) -> dict:
    provider = "human" if name == "human_confirm" else "local"
    return {
        "provider": provider,
        "model": f"{name}-model",
        "adapter": name,
        "strengths": [name],
        "max_context": 8192,
        "cost_tier": "none" if provider == "human" else "low",
        "latency_tier": "manual" if provider == "human" else "low",
        "enabled": enabled,
        "fallback": fallback,
        "supports_tools": name
        in {
            "code_agent",
            "web_research",
            "multimodal_vision",
            "image_generation",
            "human_confirm",
        },
        "modalities": ["image"] if name == "multimodal_vision" else [],
    }


def _config_path(tmp_path: Path, overrides: dict[str, dict] | None = None) -> Path:
    fallbacks = {
        "intent_router": "fast_local",
        "fast_local": "balanced_local",
        "balanced_local": "reasoning_local",
        "reasoning_local": "human_confirm",
        "code_agent": "reasoning_local",
        "web_research": "reasoning_local",
        "multimodal_vision": "reasoning_local",
        "image_generation": "human_confirm",
        "human_confirm": None,
    }
    engines = {
        name: _engine(name, fallback=fallbacks[name])
        for name in REQUIRED_ENGINE_CATEGORIES
    }
    for name, patch in (overrides or {}).items():
        engines[name] = engines.get(name, _engine(name, fallback="reasoning_local")) | patch
    path = tmp_path / "model_router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "routing_targets": {
                    "simple": "fast_local",
                    "balanced": "balanced_local",
                    "reasoning": "reasoning_local",
                    "coding": "code_agent",
                    "research": "web_research",
                    "vision": "multimodal_vision",
                    "image_generation": "image_generation",
                    "confirmation": "human_confirm",
                },
                "engines": engines,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_workflow_benchmark_default_cases_pass_without_serializing_prompts():
    report = run_workflow_benchmarks()

    assert report.ok is True
    assert report.total >= 9
    assert report.route_changes == 0
    payload = json.dumps(report.to_dict())
    assert "routing stays explicit" not in payload
    assert "Deploy this change" not in payload
    assert "prompt_hash" in payload
    private_case = next(
        result for result in report.results if result.name == "private_current_information"
    )
    assert private_case.selected_provider == "local"
    assert "policy.local_only" in private_case.reason_codes
    assert "hosted providers are excluded" in private_case.privacy_explanation
    safety_case = next(
        result for result in report.results if result.name == "risky_external_action"
    )
    assert safety_case.selected_engine == "human_confirm"
    assert safety_case.requires_confirmation is True


def test_workflow_benchmark_detects_route_changes():
    report = run_workflow_benchmarks(
        cases=(
            WorkflowBenchmarkCase(
                name="intentional_mismatch",
                category="test",
                prompt="Rewrite this text.",
                expected_engine="code_agent",
            ),
        )
    )

    assert report.ok is False
    assert report.failed == 1
    assert report.route_changes == 1
    assert "expected engine code_agent" in report.results[0].failure_reasons[0]


def test_workflow_benchmark_private_profile_excludes_hosted_provider(tmp_path):
    path = _config_path(
        tmp_path,
        {
            "hosted_reasoning": {
                "provider": "openai",
                "model": "hosted-reasoning",
                "adapter": "openai_chat",
                "strengths": ["reasoning"],
                "max_context": 128000,
                "cost_tier": "paid",
                "latency_tier": "medium",
                "enabled": True,
                "fallback": "reasoning_local",
                "supports_tools": True,
            }
        },
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["routing_targets"]["reasoning"] = "hosted_reasoning"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    report = run_workflow_benchmarks(
        config_path=path,
        cases=(
            WorkflowBenchmarkCase(
                name="private_hosted_reasoning",
                category="profile",
                prompt=(
                    "Design a multi-step architecture plan with risks, "
                    "rollout, and testing strategy."
                ),
                expected_engine="reasoning_local",
                hints={"profile": "private"},
                expected_provider="local",
                expected_requires_confirmation=False,
            ),
        ),
    )

    assert report.ok is True
    result = report.results[0]
    assert result.selected_engine == "reasoning_local"
    assert "policy.local_only" in result.reason_codes
    assert "hosted providers are excluded" in result.privacy_explanation


def test_workflow_benchmark_cli_json_is_privacy_safe():
    result = _run_cli("workflow-benchmark", "--json", "--case", "simple_rewrite")

    assert result.returncode == 0
    assert "routing stays explicit" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["total"] == 1
    assert payload["results"][0]["selected_engine"] == "fast_local"


def test_workflow_benchmark_cli_readable_report_is_release_friendly():
    result = _run_cli("workflow-benchmark", "--case", "risky_external_action")

    assert result.returncode == 0
    assert "Workflow Routing Benchmark" in result.stdout
    assert "risky_external_action: pass" in result.stdout
    assert "Deploy this change" not in result.stdout
