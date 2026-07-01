import json
from pathlib import Path
import subprocess
import sys

from scripts.replay_routing_log import replay_events
from hermes.plugins.model_router.telemetry import feedback_summary, review_queue

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_pricing_catalog(path):
    path.write_text(
        """catalog_version: 3
updated_at: "2026-06-30T00:00:00Z"
entries:
  - provider: test
    model: actual-fast
    input_per_1m: 2
    output_per_1m: 4
    cached_input_per_1m: 0.5
    currency: USD
    effective_date: "2026-06-30"
    source: test-fixture
    notes: test only
""",
        encoding="utf-8",
    )


def _write_placeholder_pricing_catalog(path):
    path.write_text(
        """catalog_version: 5
updated_at: "2026-06-30T00:00:00Z"
entries:
  - provider: example
    model: placeholder-model
    input_per_1m: 1
    output_per_1m: 3
    cached_input_per_1m: 0.25
    currency: USD
    effective_date: "2026-06-30"
    source: example-placeholder-not-current-pricing
    notes: Non-authoritative placeholder for operator override shape.
""",
        encoding="utf-8",
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
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
                "outcome_label": "failed_verification",
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
    assert summary["outcome_label_counts"] == {"failed_verification": 1}
    assert summary["usage_events"] == 0
    assert summary["usage_by_backend"] == {}


def test_replay_routing_log_summarizes_usage_without_prompt_text(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(feedback, [])
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "usage-1",
                "prompt": "rewrite token=secret-value",
                "selected_engine": "fast_local",
                "backend": "fast",
                "backend_model": "configured-fast",
                "upstream_model": "actual-fast",
                "status": "forwarded",
                "usage_prompt_tokens": 10,
                "usage_completion_tokens": 5,
                "usage_total_tokens": 15,
                "usage_cached_input_tokens": 3,
            },
            {
                "event_type": "routing_event",
                "request_id": "usage-2",
                "prompt_hash": "private",
                "selected_engine": "code_agent",
                "backend": "code",
                "backend_model": "code-model",
                "status": "forwarded",
                "usage_prompt_tokens": 20,
                "usage_completion_tokens": 8,
                "usage_total_tokens": 28,
            },
            {
                "event_type": "routing_event",
                "request_id": "old-row",
                "prompt": "summarize this",
                "selected_engine": "balanced_local",
                "backend": "fast",
                "status": "forwarded",
            },
        ],
    )

    summary = replay_events(
        events_path=events,
        feedback_path=feedback,
        config_path=None,
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["usage_events"] == 2
    assert summary["usage_prompt_tokens"] == 30
    assert summary["usage_completion_tokens"] == 13
    assert summary["usage_total_tokens"] == 43
    assert summary["usage_cached_input_tokens"] == 3
    assert summary["usage_by_selected_engine"]["fast_local"][
        "usage_total_tokens"
    ] == 15
    assert summary["usage_by_backend"]["code"]["usage_prompt_tokens"] == 20
    assert summary["usage_by_model"]["actual-fast"]["usage_total_tokens"] == 15
    assert summary["usage_by_model"]["code-model"]["usage_total_tokens"] == 28
    assert summary["upstream_model_counts"] == {"actual-fast": 1}
    coverage = summary["catalog_coverage"]
    assert coverage["total_routing_rows"] == 3
    assert coverage["total_rows_with_usage"] == 2
    assert coverage["rows_without_enough_usage_data"] == 1
    assert coverage["rows_missing_provider_model_catalog_match"] == 2
    assert coverage["cost_confidence"] == "no_catalog_match"
    assert "secret-value" not in serialized


def test_replay_routing_log_estimates_cost_from_local_catalog(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    pricing = tmp_path / "pricing_catalog.yaml"
    _write_jsonl(feedback, [])
    _write_pricing_catalog(pricing)
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "priced",
                "prompt": "rewrite token=secret-value",
                "selected_engine": "fast_local",
                "backend": "fast",
                "backend_model": "configured-fast",
                "upstream_model": "actual-fast",
                "status": "forwarded",
                "usage_prompt_tokens": 10,
                "usage_completion_tokens": 5,
                "usage_total_tokens": 15,
                "usage_cached_input_tokens": 3,
            },
            {
                "event_type": "routing_event",
                "request_id": "missing-price",
                "prompt": "summarize this",
                "selected_engine": "balanced_local",
                "backend": "balanced",
                "backend_model": "unknown-model",
                "status": "forwarded",
                "usage_prompt_tokens": 20,
                "usage_completion_tokens": 8,
                "usage_total_tokens": 28,
            },
        ],
    )

    summary = replay_events(
        events_path=events,
        feedback_path=feedback,
        config_path=None,
        pricing_catalog_path=pricing,
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["pricing_catalog_version"] == 3
    assert summary["pricing_match_counts"] == {"matched": 1, "missing_price": 1}
    assert summary["estimated_cost_events"] == 1
    assert summary["estimated_input_cost"] == 0.000014
    assert summary["estimated_output_cost"] == 0.00002
    assert summary["estimated_cached_input_cost"] == 0.0000015
    assert summary["estimated_total_cost"] == 0.0000355
    assert summary["estimated_cost_currency"] == "USD"
    assert summary["usage_by_model"]["actual-fast"]["estimated_total_cost"] == 0.0000355
    coverage = summary["catalog_coverage"]
    assert coverage["active_catalog_version"] == 3
    assert coverage["total_rows_with_usage"] == 2
    assert coverage["rows_with_catalog_match"] == 1
    assert coverage["rows_missing_provider_model_catalog_match"] == 1
    assert coverage["rows_using_placeholder_pricing"] == 0
    assert coverage["rows_with_estimated_cost"] == 1
    assert coverage["cost_confidence"] == "partial_catalog_match"
    assert "secret-value" not in serialized


def test_replay_routing_log_reports_placeholder_pricing_coverage(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    pricing = tmp_path / "pricing_catalog.yaml"
    _write_jsonl(feedback, [])
    _write_placeholder_pricing_catalog(pricing)
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "placeholder",
                "prompt": "rewrite token=secret-value",
                "selected_engine": "fast_local",
                "upstream_model": "placeholder-model",
                "usage_prompt_tokens": 10,
                "usage_completion_tokens": 5,
                "usage_total_tokens": 15,
            }
        ],
    )

    summary = replay_events(
        events_path=events,
        feedback_path=feedback,
        config_path=None,
        pricing_catalog_path=pricing,
    )
    review = review_queue(
        events_path=events,
        feedback_path=feedback,
        pricing_catalog_path=pricing,
    )
    serialized = json.dumps({"summary": summary, "review": review}, sort_keys=True)

    assert summary["pricing_match_counts"] == {"matched": 1}
    assert summary["catalog_coverage"]["rows_using_placeholder_pricing"] == 1
    assert summary["catalog_coverage"]["cost_confidence"] == "placeholder_pricing"
    assert review["catalog_coverage"]["rows_using_placeholder_pricing"] == 1
    assert review["items"][0]["cost"]["pricing_is_placeholder"] is True
    assert "secret-value" not in serialized


def test_review_queue_includes_usage_without_prompt_or_response_text(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(feedback, [])
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "review-usage",
                "prompt": "api_key=secret-value fix this",
                "selected_engine": "code_agent",
                "backend": "code",
                "backend_model": "configured-code",
                "upstream_model": "actual-code",
                "status": "forwarded",
                "receipt_summary": "Selected code_agent.",
                "reason_codes": ["route.coding"],
                "usage_prompt_tokens": 40,
                "usage_completion_tokens": 12,
                "usage_total_tokens": 52,
                "usage_cached_input_tokens": 7,
            }
        ],
    )

    summary = review_queue(events_path=events, feedback_path=feedback)
    item = summary["items"][0]
    serialized = json.dumps(summary, sort_keys=True)

    assert item["usage"]["usage_prompt_tokens"] == 40
    assert item["usage"]["usage_completion_tokens"] == 12
    assert item["usage"]["usage_total_tokens"] == 52
    assert item["usage"]["usage_cached_input_tokens"] == 7
    assert item["upstream_model"] == "actual-code"
    assert "api_key" not in serialized
    assert "secret-value" not in serialized
    assert "fix this" not in serialized


def test_review_queue_includes_cost_without_prompt_or_response_text(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    pricing = tmp_path / "pricing_catalog.yaml"
    _write_jsonl(feedback, [])
    _write_pricing_catalog(pricing)
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "review-priced",
                "prompt": "api_key=secret-value rewrite this",
                "selected_engine": "fast_local",
                "backend_model": "configured-fast",
                "upstream_model": "actual-fast",
                "usage_prompt_tokens": 10,
                "usage_completion_tokens": 5,
                "usage_total_tokens": 15,
                "usage_cached_input_tokens": 3,
            }
        ],
    )

    summary = review_queue(
        events_path=events,
        feedback_path=feedback,
        pricing_catalog_path=pricing,
    )
    item = summary["items"][0]
    serialized = json.dumps(summary, sort_keys=True)

    assert item["cost"]["pricing_match_status"] == "matched"
    assert item["cost"]["estimated_total_cost"] == 0.0000355
    assert summary["catalog_coverage"]["rows_with_catalog_match"] == 1
    assert summary["catalog_coverage"]["rows_with_estimated_cost"] == 1
    assert "secret-value" not in serialized
    assert "rewrite this" not in serialized


def test_telemetry_cli_summary_and_review_show_usage_without_prompt_text(tmp_path):
    events = tmp_path / "events.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(feedback, [])
    pricing = tmp_path / "pricing_catalog.yaml"
    _write_pricing_catalog(pricing)
    _write_jsonl(
        events,
        [
            {
                "event_type": "routing_event",
                "request_id": "cli-usage",
                "prompt": "api_key=secret-value rewrite this",
                "selected_engine": "fast_local",
                "backend": "fast",
                "backend_model": "configured-fast",
                "upstream_model": "actual-fast",
                "status": "forwarded",
                "usage_prompt_tokens": 14,
                "usage_completion_tokens": 6,
                "usage_total_tokens": 20,
            }
        ],
    )

    summary = _run_cli(
        "telemetry",
        "summary",
        "--events",
        str(events),
        "--feedback",
        str(feedback),
        "--pricing-catalog",
        str(pricing),
    )
    review = _run_cli(
        "telemetry",
        "review",
        "--events",
        str(events),
        "--feedback",
        str(feedback),
        "--pricing-catalog",
        str(pricing),
    )

    assert summary.returncode == 0
    assert "Usage events: 1" in summary.stdout
    assert "Usage tokens: prompt=14, completion=6, total=20" in summary.stdout
    assert "Usage by backend:" in summary.stdout
    assert "fast: prompt=14, completion=6, total=20" in summary.stdout
    assert "Estimated cost events: 1" in summary.stdout
    assert "Estimated cost: 0.000052 USD events=1" in summary.stdout
    assert "Catalog coverage:" in summary.stdout
    assert "usage_rows=1" in summary.stdout
    assert "matched=1" in summary.stdout
    assert "confidence=catalog_matched" in summary.stdout
    assert "secret-value" not in summary.stdout
    assert "rewrite this" not in summary.stdout
    assert review.returncode == 0
    assert "usage: prompt=14, completion=6, total=20" in review.stdout
    assert "cost: 0.000052 USD" in review.stdout
    assert "Catalog coverage:" in review.stdout
    assert "usage_rows=1" in review.stdout
    assert "confidence=catalog_matched" in review.stdout
    assert "secret-value" not in review.stdout
    assert "rewrite this" not in review.stdout


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
                "outcome_label": "wrong_route",
                "notes": "contains private context",
            },
            {
                "event_type": "routing_feedback",
                "request_id": "missing",
                "expected_engine": "code_agent",
                "outcome_label": "automatic_success",
                "notes": "missing event",
            },
            {
                "event_type": "routing_feedback",
                "request_id": "private",
                "expected_engine": "balanced_local",
            },
        ],
    )

    summary = feedback_summary(
        feedback_path=feedback,
        events_path=events,
        include_notes=False,
    )

    assert summary["feedback_labels"] == 3
    assert summary["expected_engine_counts"] == {
        "balanced_local": 2,
        "code_agent": 1,
    }
    assert summary["outcome_label_counts"] == {"wrong_route": 1}
    assert "notes" not in summary["labels"][0]
    assert summary["labels"][0]["request_id"] == "labeled"
    assert summary["labels"][0]["outcome_label"] == "wrong_route"
    assert summary["labels"][0]["event_found"] is True
    assert summary["labels"][0]["replayable"] is True
    assert summary["labels"][1]["request_id"] == "missing"
    assert "outcome_label" not in summary["labels"][1]
    assert summary["labels"][1]["event_found"] is False
    assert summary["labels"][2]["request_id"] == "private"
    assert "outcome_label" not in summary["labels"][2]
