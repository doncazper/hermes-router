import json
from pathlib import Path

from scripts.replay_routing_log import replay_events

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
    assert summary["route_change_count"] == 1
    assert summary["expected_mismatch_count"] == 1
    assert summary["confusion_matrix"] == {"reasoning_local->code_agent": 1}


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
