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
