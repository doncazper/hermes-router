import json
import subprocess
import sys
from pathlib import Path

from hermes.plugins.model_router.dispatch import (
    build_dispatch_plan,
    dispatch_plan_to_json,
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


def test_dispatch_plan_allows_safe_dry_run_without_execution():
    plan = build_dispatch_plan("rewrite this text")

    assert plan.dry_run is True
    assert plan.can_dispatch is True
    assert plan.blocked is False
    assert plan.selected_engine == "fast_local"
    assert plan.adapter == "local_chat"
    assert plan.provider == "local"
    assert plan.receipt.selected_engine == "fast_local"
    assert any("dry-run" in reason for reason in plan.reasons)


def test_dispatch_plan_blocks_high_risk_confirmation_routes():
    plan = build_dispatch_plan("drop the production database")

    assert plan.dry_run is True
    assert plan.can_dispatch is False
    assert plan.blocked is True
    assert plan.requires_confirmation is True
    assert plan.selected_engine == "human_confirm"
    assert any("confirmation" in reason for reason in plan.reasons)


def test_dispatch_plan_json_does_not_include_raw_prompt():
    prompt = "rewrite this private sentence"
    plan = build_dispatch_plan(prompt)

    serialized = dispatch_plan_to_json(plan)
    payload = json.loads(serialized)

    assert prompt not in serialized
    assert "prompt" not in payload
    assert payload["receipt"]["selected_engine"] == "fast_local"


def test_dispatch_plan_cli_json_emits_parseable_dry_run_plan():
    result = _run_cli("dispatch-plan", "--json", "fix the repo and run tests")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["selected_engine"] == "code_agent"
    assert payload["can_dispatch"] is True
    assert payload["receipt"]["requires_code_execution"] is True


def test_dispatch_plan_cli_readable_output_names_adapter():
    result = _run_cli("dispatch-plan", "rewrite this text")

    assert result.returncode == 0
    assert "Dry run: true" in result.stdout
    assert "Selected engine: fast_local" in result.stdout
    assert "Adapter: local_chat" in result.stdout
