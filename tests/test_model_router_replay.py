import json
from pathlib import Path

from scripts.replay_routing_log import replay_events
from hermes.plugins.model_router.telemetry import feedback_summary

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_replay_routing_log_reports_changes_and_feedback_mismatches(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "ok",
                "prompt": "rewrite this text",
                "selected_engine": "fast_local",
                "route_latency_ms": 0.02,
            },
            {
                "event_type": "routing_event",
                "request_id": "changed",
                "prompt": "fix the repo and run tests",
                "selected_engine": "balanced_local",
                "route_latency_ms": 0.02,
            },
            {
                "event_type": "routing_event",
                "request_id": "private",
                "selected_engine": "balanced_local",
                "prompt_hash": "abc",
            },
        ],
    )
    _write_jsonl(
        feedback,
        [
            {
                "event_type": "routing_feedback",
                "request_id": "changed",
                "expected_engine": "reasoning_local",
            }
        ],
    )

    summary = replay_events(
        events_path=events,
        feedback_path=feedback,
        config_path=None,
    )

    assert summary["replayed"] == 2
    assert summary["skipped_no_prompt"] == 1
    assert summary["labeled_replayable"] == 1
    assert summary["unlabeled_replayable"] == 1
    assert summary["unlabeled_replayable_request_ids"] == ["ok"]
    assert summary["skipped_no_prompt_request_ids"] == ["private"]
    assert summary["route_change_count"] == 1
    assert summary["expected_mismatch_count"] == 1
    assert summary["confusion_matrix"] == {"reasoning_local->code_agent": 1}
    assert summary["mismatch_groups"] == {"reasoning_local->code_agent": 1}


def test_replay_fixture_corpus_has_no_expected_mismatches():
    fixture_dir = ROOT / "tests" / "fixtures" / "routing_corpus"

    fixture_pairs = (
        ("v0_5_proxy_events.jsonl", "v0_5_feedback.jsonl", 4),
        (
            "short_prompt_calibration_events.jsonl",
            "short_prompt_calibration_feedback.jsonl",
            7,
        ),
    )

    for events_name, feedback_name, expected_count in fixture_pairs:
        summary = replay_events(
            events_path=fixture_dir / events_name,
            feedback_path=fixture_dir / feedback_name,
            config_path=None,
        )

        assert summary["replayed"] == expected_count
        assert summary["expected_mismatch_count"] == 0


def test_feedback_summary_joins_events_without_prompt_text(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "labeled",
                "prompt": "api_key=secret-value rewrite this text",
                "selected_engine": "fast_local",
                "status": "forwarded",
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
                "request_id": "labeled",
                "expected_engine": "balanced_local",
                "notes": "contains private context",
            },
            {
                "event_type": "routing_feedback",
                "request_id": "missing",
                "expected_engine": "code_agent",
                "notes": "missing event",
            },
        ],
    )

    summary = feedback_summary(
        feedback_path=feedback,
        events_path=events,
        include_notes=False,
    )

    assert summary["feedback_labels"] == 2
    assert summary["expected_engine_counts"] == {
        "balanced_local": 1,
        "code_agent": 1,
    }
    assert "notes" not in summary["labels"][0]
    assert summary["labels"][0]["request_id"] == "labeled"
    assert summary["labels"][0]["event_found"] is True
    assert summary["labels"][0]["replayable"] is True
    assert summary["labels"][1]["request_id"] == "missing"
    assert summary["labels"][1]["event_found"] is False
