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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_readable_cli_emits_selected_engine_and_scores():
    result = _run_cli("decide", "rewrite this text")

    assert result.returncode == 0
    assert "Summary: Selected fast_local" in result.stdout
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
    assert "summary" in payload
    assert "route.coding" in payload["reason_codes"]


def test_explain_cli_emits_privacy_safe_receipt_summary():
    prompt = "fix the repo with token=secret-value and run tests"
    result = _run_cli("decide", "--explain", prompt)

    assert result.returncode == 0
    assert "Route Receipt" in result.stdout
    assert "Summary: Selected code_agent" in result.stdout
    assert "Reason codes:" in result.stdout
    assert "route.coding" in result.stdout
    assert "Wrong route:" in result.stdout
    assert "secret-value" not in result.stdout


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


def test_json_cli_accepts_routing_profile():
    result = _run_cli(
        "decide",
        "--json",
        "--profile",
        "private",
        "rewrite this text",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["routing_profile"] == "private"
    assert payload["requirements"]["allowed_providers"] == ["local", "human"]


def test_json_cli_accepts_provider_policy_hints():
    result = _run_cli(
        "decide",
        "--json",
        "--provider-deny",
        "openai",
        "--no-hosted",
        "research latest model routing approaches",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["requirements"]["provider_denylist"] == ["openai"]
    assert payload["requirements"]["allowed_providers"] == ["local", "human"]


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


def test_settings_cli_help_exposes_local_admin_command():
    result = _run_cli("settings", "--help")

    assert result.returncode == 0
    assert "--config-dir" in result.stdout
    assert "--no-open" in result.stdout
    assert "--port" in result.stdout


def test_install_cli_json_is_parseable_plan_only(tmp_path):
    config_dir = tmp_path / "install"
    result = _run_cli(
        "install",
        "--config-dir",
        str(config_dir),
        "--quick",
        "--json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["existing_config"] is False
    assert not config_dir.exists()
    command_ids = {command["id"] for command in payload["next_commands"]}
    assert "init" in command_ids
    assert "doctor" in command_ids
    assert "settings" in command_ids
    assert "proxy" in command_ids


def test_telemetry_summary_cli_groups_mismatches_without_prompt_text(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "secret-prompt",
                "prompt": "api_key=secret-value rewrite this text",
                "selected_engine": "balanced_local",
                "status": "forwarded",
                "route_latency_ms": 0.02,
            },
            {
                "event_type": "routing_event",
                "request_id": "private",
                "prompt_hash": "abc",
                "selected_engine": "balanced_local",
                "status": "forwarded",
            },
        ],
    )
    _write_jsonl(
        feedback,
        [
            {
                "event_type": "routing_feedback",
                "request_id": "secret-prompt",
                "expected_engine": "fast_local",
                "notes": "token=secret-value",
            }
        ],
    )

    result = _run_cli(
        "telemetry",
        "summary",
        "--events",
        str(events),
        "--feedback",
        str(feedback),
        "--json",
    )

    assert result.returncode == 0
    assert "secret-value" not in result.stdout
    assert "rewrite this text" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["routing_events"] == 2
    assert payload["replayed"] == 1
    assert payload["skipped_no_prompt"] == 1
    assert payload["labeled_replayable"] == 1
    assert payload["expected_mismatch_count"] == 1
    assert payload["mismatch_groups"] == {"fast_local->reasoning_local": 1}
    assert payload["skipped_no_prompt_request_ids"] == ["private"]


def test_telemetry_feedback_cli_hides_notes_by_default(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "req-1",
                "prompt": "rewrite this text",
                "selected_engine": "fast_local",
                "status": "forwarded",
            }
        ],
    )
    _write_jsonl(
        feedback,
        [
            {
                "event_type": "routing_feedback",
                "request_id": "req-1",
                "expected_engine": "balanced_local",
                "notes": "contains private note",
            }
        ],
    )

    result = _run_cli(
        "telemetry",
        "feedback",
        "--events",
        str(events),
        "--feedback",
        str(feedback),
        "--json",
    )

    assert result.returncode == 0
    assert "contains private note" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["feedback_labels"] == 1
    assert payload["labels"][0]["request_id"] == "req-1"
    assert payload["labels"][0]["replayable"] is True
    assert "notes" not in payload["labels"][0]


def test_telemetry_review_cli_hides_prompts_and_notes_by_default(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "already-labeled",
                "prompt": "token=secret-value fix this bug",
                "prompt_preview": "token=secret-value fix this bug",
                "selected_engine": "balanced_local",
                "status": "forwarded",
            },
            {
                "event_type": "routing_event",
                "request_id": "needs-review",
                "prompt": "api_key=secret-value rewrite this",
                "prompt_preview": "api_key=secret-value rewrite this",
                "selected_engine": "fast_local",
                "routing_profile": "balanced",
                "status": "forwarded",
                "backend": "fast",
                "receipt_summary": "Selected fast_local under the balanced profile.",
                "reason_codes": ["profile.balanced", "route.simple"],
            },
            {
                "event_type": "routing_event",
                "request_id": "private-no-prompt",
                "prompt_hash": "abc",
                "selected_engine": "code_agent",
                "status": "forwarded",
            },
        ],
    )
    _write_jsonl(
        feedback,
        [
            {
                "event_type": "routing_feedback",
                "request_id": "already-labeled",
                "expected_engine": "code_agent",
                "notes": "contains token=secret-value",
            }
        ],
    )

    result = _run_cli(
        "telemetry",
        "review",
        "--events",
        str(events),
        "--feedback",
        str(feedback),
        "--json",
    )

    assert result.returncode == 0
    assert "secret-value" not in result.stdout
    assert "rewrite this" not in result.stdout
    assert "contains token" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["reviewable"] == 2
    assert payload["skipped_labeled"] == 1
    assert payload["skipped_private"] == 1
    assert payload["items"][0]["request_id"] == "private-no-prompt"
    assert payload["items"][0]["replayable"] is False
    assert payload["items"][1]["request_id"] == "needs-review"
    assert payload["items"][1]["reason_codes"] == ["profile.balanced", "route.simple"]
    assert "model-router feedback needs-review" in payload["items"][1][
        "suggested_feedback_command"
    ]


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


def test_init_cli_accepts_auto_models_for_llamacpp(tmp_path):
    model_dir = tmp_path / "models" / "Qwen3-4B-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3-4B-Q4_K_M.gguf").write_text("placeholder", encoding="utf-8")

    result = _run_cli(
        "init",
        "--preset",
        "llamacpp",
        "--auto-models",
        "--model-dir",
        str(model_dir),
        "--yes",
        "--config-dir",
        str(tmp_path / "config"),
        "--json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert any("Auto-selected" in message for message in payload["messages"])


def test_setup_install_prereqs_cli_defaults_to_dry_run():
    result = _run_cli(
        "setup",
        "install-prereqs",
        "--preset",
        "mlx-lm",
        "--json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["executed"] is False
    assert any(step["command"][-1] == "mlx-lm" for step in payload["steps"])


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


def test_dogfood_proxy_cli_defaults_to_plan_only(tmp_path):
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
        "dogfood",
        "proxy",
        "--config",
        str(tmp_path / "routing_proxy.yaml"),
        "--json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["executed"] is False
    assert payload["ok"] is True
    assert payload["planned"] >= 8
    assert "Live dogfood is local and opt-in" in " ".join(payload["notes"])


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
