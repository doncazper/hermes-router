import json
import shutil
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


def test_console_script_emits_parseable_receipt_when_installed():
    executable = shutil.which("hermes-router") or str(
        Path(sys.executable).with_name("hermes-router")
    )
    assert Path(executable).is_file()
    result = subprocess.run(
        [executable, "decide", "--json", "fix the repo and run tests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["selected_engine"] == "code_agent"


def test_feedback_cli_appends_jsonl_label(tmp_path):
    output = tmp_path / "feedback.jsonl"
    result = _run_cli(
        "feedback",
        "--output",
        str(output),
        "--notes",
        "should have used code",
        "req-123",
        "code_agent",
    )

    assert result.returncode == 0
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["event_type"] == "routing_feedback"
    assert row["request_id"] == "req-123"
    assert row["expected_engine"] == "code_agent"
    assert row["notes"] == "should have used code"


def test_init_cli_writes_configs(tmp_path):
    result = _run_cli(
        "init",
        "--preset",
        "lmstudio",
        "--yes",
        "--config-dir",
        str(tmp_path),
        "--proxy-port",
        "9090",
        "--json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert (tmp_path / "model_router.yaml").is_file()
    assert (tmp_path / "routing_proxy.yaml").is_file()
    assert payload["preset"] == "lmstudio"


def test_validate_proxy_config_cli(tmp_path):
    init = _run_cli(
        "init",
        "--preset",
        "lmstudio",
        "--yes",
        "--config-dir",
        str(tmp_path),
        "--json",
    )
    assert init.returncode == 0

    result = _run_cli(
        "validate-proxy-config",
        "--config",
        str(tmp_path / "routing_proxy.yaml"),
        "--json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["config_valid"] is True
    assert "fast" in payload["backends"]


def test_doctor_cli_emits_json_report_for_unreachable_backends(tmp_path):
    init = _run_cli(
        "init",
        "--preset",
        "lmstudio",
        "--yes",
        "--config-dir",
        str(tmp_path),
        "--json",
    )
    assert init.returncode == 0

    result = _run_cli(
        "doctor",
        "--config",
        str(tmp_path / "routing_proxy.yaml"),
        "--timeout",
        "0.01",
        "--json",
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["proxy_config_valid"] is True
    assert payload["router_config_valid"] is True
    assert payload["backends"]
