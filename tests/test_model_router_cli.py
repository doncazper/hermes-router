import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_readable_cli_emits_selected_engine_and_scores():
    result = _run_cli("decide", "rewrite this text")

    assert result.returncode == 0
    assert "Selected engine: fast_local" in result.stdout
    assert "Complexity:" in result.stdout
    assert "Risk:" in result.stdout


def test_json_cli_emits_parseable_receipt():
    result = _run_cli("decide", "--json", "fix the repo and run tests")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["selected_engine"] == "code_agent"
    assert payload["requires_code_execution"] is True
    assert payload["requires_tools"] is True
    assert "alternatives" in payload


def test_json_cli_accepts_force_engine_hint():
    result = _run_cli(
        "decide",
        "--json",
        "--force-engine",
        "reasoning_local",
        "rewrite this text",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["selected_engine"] == "reasoning_local"
    assert any("forced engine reasoning_local" in reason for reason in payload["reasons"])


def test_json_cli_accepts_attachment_hint():
    result = _run_cli(
        "decide",
        "--json",
        "--attachment",
        "image",
        "summarize this attachment",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["selected_engine"] == "multimodal_vision"
    assert payload["requirements"]["required_modalities"] == ["image"]


def test_invalid_config_path_emits_fail_closed_receipt():
    result = _run_cli(
        "decide",
        "--json",
        "--config",
        "configs/does-not-exist.yaml",
        "rewrite this text",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["selected_engine"] == "human_confirm"
    assert payload["requires_confirmation"] is True
    assert payload["config_valid"] is False


def test_validate_config_json_emits_availability_report():
    result = _run_cli("validate-config", "--json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["config_valid"] is True
    assert "engines" in payload
    assert "code_agent" in payload["engines"]


def test_readable_cli_emits_ranked_alternatives():
    result = _run_cli("decide", "rewrite this text")

    assert result.returncode == 0
    assert "Alternatives:" in result.stdout
