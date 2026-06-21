import json
import subprocess
import sys

from hermes.plugins.model_router import ModelRouter
from hermes.plugins.model_router.routing_log import (
    PROMPT_CAPTURE_FULL,
    PROMPT_CAPTURE_REDACTED,
    RoutingLogWriter,
    build_routing_event,
    prompt_fields,
    read_jsonl,
    redact_text,
)


def test_redaction_removes_obvious_secrets():
    text = "api_key=abc123 password:supersecret Bearer tokenvalue123456"
    redacted = redact_text(text)

    assert "abc123" not in redacted
    assert "supersecret" not in redacted
    assert "tokenvalue123456" not in redacted
    assert "[REDACTED]" in redacted


def test_prompt_fields_default_to_hash_and_redacted_preview():
    fields = prompt_fields(
        "rewrite this api_key=abc123",
        capture=PROMPT_CAPTURE_REDACTED,
    )

    assert "prompt_hash" in fields
    assert fields["prompt_length"] > 0
    assert fields["estimated_tokens"] > 0
    assert "prompt" not in fields
    assert "abc123" not in fields["prompt_preview"]


def test_full_prompt_capture_requires_explicit_mode(monkeypatch):
    fields = prompt_fields("rewrite this", capture=PROMPT_CAPTURE_REDACTED)
    assert "prompt" not in fields

    full_fields = prompt_fields("rewrite this", capture=PROMPT_CAPTURE_FULL)
    assert full_fields["prompt"] == "rewrite this"

    monkeypatch.setenv("MODEL_ROUTER_LOG_PROMPTS", "1")
    env_fields = prompt_fields("rewrite this", capture=PROMPT_CAPTURE_REDACTED)
    assert env_fields["prompt"] == "rewrite this"


def test_routing_log_writer_is_best_effort(tmp_path):
    writer = RoutingLogWriter(tmp_path)

    assert writer.write({"event_type": "routing_event"}) is False


def test_routing_log_writer_rotates_when_size_limit_is_exceeded(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("x" * 20, encoding="utf-8")
    writer = RoutingLogWriter(path, max_bytes=10, backups=2)

    assert writer.write({"event_type": "routing_event", "request_id": "new"})

    assert path.exists()
    assert (tmp_path / "events.jsonl.1").exists()
    assert "request_id" in path.read_text(encoding="utf-8")


def test_routing_log_writer_rotation_respects_backup_limit(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = RoutingLogWriter(path, max_bytes=80, backups=3)

    for index in range(20):
        assert writer.write(
            {
                "event_type": "routing_event",
                "request_id": f"req-{index}",
                "payload": "x" * 80,
            }
        )

    assert path.exists()
    assert (tmp_path / "events.jsonl.1").exists()
    assert (tmp_path / "events.jsonl.2").exists()
    assert (tmp_path / "events.jsonl.3").exists()
    assert not (tmp_path / "events.jsonl.4").exists()


def test_routing_event_schema_is_json_safe(tmp_path):
    router = ModelRouter.from_config(validate_availability=False)
    decision = router.route("fix the repo and run tests", include_alternatives=False)
    event = build_routing_event(
        request_id="req-1",
        route_api="route_fast",
        selected_engine=decision.selected_engine,
        status="forwarded",
        prompt="fix the repo and run tests",
        route_latency_ms=0.01,
        diagnostic_latency_ms=0.2,
        upstream_latency_ms=4.0,
        total_latency_ms=4.3,
        config_source="default",
        router_version="test",
        backend="code",
        backend_model="code-model",
        status_code=200,
        decision=decision,
    )
    path = tmp_path / "events.jsonl"
    writer = RoutingLogWriter(path)

    assert writer.write(event) is True
    rows = read_jsonl(path)

    assert rows[0]["event_type"] == "routing_event"
    assert rows[0]["selected_engine"] == "code_agent"
    assert rows[0]["complexity_score"] == decision.complexity_score
    assert rows[0]["features"]["coding_intent"] is True
    assert json.dumps(rows[0])


def test_core_route_fast_does_not_import_routing_log():
    script = """
import sys
import model_router
router = model_router.ModelRouter.from_config(validate_availability=False)
router.route_fast('rewrite this text')
print('hermes.plugins.model_router.routing_log' in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "False"
