"""Telemetry inspection helpers for routing dogfood loops."""

from __future__ import annotations

from collections import Counter
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from hermes.plugins.model_router.policy import ModelRouter
from hermes.plugins.model_router.routing_log import DEFAULT_FEEDBACK_PATH, read_jsonl


def replay_events(
    *,
    events_path: str | Path,
    feedback_path: str | Path | None,
    config_path: str | Path | None,
    max_examples: int = 10,
) -> dict[str, Any]:
    events = read_jsonl(events_path)
    feedback_rows = read_jsonl(feedback_path) if feedback_path else []
    feedback = _feedback_records_by_request(feedback_rows)
    router = ModelRouter.from_config(config_path, validate_availability=False)

    routing_events = _routing_events(events)
    event_by_request = _events_by_request(routing_events)
    event_request_ids = set(event_by_request)
    feedback_request_ids = set(feedback)

    route_changes: list[dict[str, str]] = []
    expected_mismatches: list[dict[str, str]] = []
    confusion: Counter[str] = Counter()
    mismatch_groups: Counter[str] = Counter()
    selected_engine_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    replay_latencies_ms: list[float] = []
    historical_latencies_ms: list[float] = []
    unlabeled_replayable_ids: list[str] = []
    skipped_no_prompt_ids: list[str] = []
    replayed = 0
    skipped_no_prompt = 0
    labeled_replayable = 0

    for event in routing_events:
        request_id = str(event.get("request_id", ""))
        historical_engine = str(event.get("selected_engine", ""))
        status = str(event.get("status", "unknown"))
        if historical_engine:
            selected_engine_counts[historical_engine] += 1
        status_counts[status] += 1

        prompt = event.get("prompt")
        if not isinstance(prompt, str):
            skipped_no_prompt += 1
            if request_id:
                skipped_no_prompt_ids.append(request_id)
            continue

        if request_id and request_id not in feedback:
            unlabeled_replayable_ids.append(request_id)

        started = perf_counter()
        current_engine = router.route_fast(prompt)
        replay_latency_ms = (perf_counter() - started) * 1000
        replay_latencies_ms.append(replay_latency_ms)
        replayed += 1

        historical_latency = event.get("route_latency_ms")
        if isinstance(historical_latency, (int, float)):
            historical_latencies_ms.append(float(historical_latency))

        if current_engine != historical_engine:
            route_changes.append(
                {
                    "request_id": request_id,
                    "historical_engine": historical_engine,
                    "current_engine": current_engine,
                }
            )

        feedback_row = feedback.get(request_id)
        if feedback_row:
            labeled_replayable += 1
            expected_engine = str(feedback_row.get("expected_engine", ""))
            confusion[f"{expected_engine}->{current_engine}"] += 1
            if current_engine != expected_engine:
                mismatch_key = f"{expected_engine}->{current_engine}"
                mismatch_groups[mismatch_key] += 1
                expected_mismatches.append(
                    {
                        "request_id": request_id,
                        "expected_engine": expected_engine,
                        "current_engine": current_engine,
                    }
                )

    feedback_without_event_ids = sorted(feedback_request_ids - event_request_ids)
    feedback_for_private_event_ids = sorted(
        request_id
        for request_id in feedback_request_ids & event_request_ids
        if not isinstance(event_by_request[request_id].get("prompt"), str)
    )

    historical_mean = _mean(historical_latencies_ms)
    replay_mean = _mean(replay_latencies_ms)
    return {
        "events": len(events),
        "routing_events": len(routing_events),
        "feedback_labels": len(feedback),
        "replayed": replayed,
        "skipped_no_prompt": skipped_no_prompt,
        "labeled_replayable": labeled_replayable,
        "unlabeled_replayable": len(unlabeled_replayable_ids),
        "unlabeled_replayable_request_ids": _limit(
            sorted(unlabeled_replayable_ids),
            max_examples,
        ),
        "skipped_no_prompt_request_ids": _limit(
            sorted(skipped_no_prompt_ids),
            max_examples,
        ),
        "feedback_without_event_count": len(feedback_without_event_ids),
        "feedback_without_event_request_ids": _limit(
            feedback_without_event_ids,
            max_examples,
        ),
        "feedback_for_private_event_count": len(feedback_for_private_event_ids),
        "feedback_for_private_event_request_ids": _limit(
            feedback_for_private_event_ids,
            max_examples,
        ),
        "route_changes": route_changes,
        "route_change_count": len(route_changes),
        "expected_mismatches": expected_mismatches,
        "expected_mismatch_count": len(expected_mismatches),
        "mismatch_groups": dict(sorted(mismatch_groups.items())),
        "confusion_matrix": dict(sorted(confusion.items())),
        "selected_engine_counts": dict(sorted(selected_engine_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "historical_route_latency_mean_ms": historical_mean,
        "replay_route_latency_mean_ms": replay_mean,
        "route_latency_delta_mean_ms": (
            round(replay_mean - historical_mean, 6)
            if historical_mean is not None and replay_mean is not None
            else None
        ),
    }


def feedback_summary(
    *,
    feedback_path: str | Path,
    events_path: str | Path | None = None,
    include_notes: bool = False,
    max_rows: int = 50,
) -> dict[str, Any]:
    feedback = _feedback_records_by_request(read_jsonl(feedback_path))
    events = _routing_events(read_jsonl(events_path)) if events_path else []
    event_by_request = _events_by_request(events)

    labels: list[dict[str, Any]] = []
    expected_engine_counts: Counter[str] = Counter()
    for request_id, row in sorted(feedback.items()):
        expected_engine = str(row.get("expected_engine", ""))
        expected_engine_counts[expected_engine] += 1
        event = event_by_request.get(request_id)
        label: dict[str, Any] = {
            "request_id": request_id,
            "expected_engine": expected_engine,
            "timestamp": row.get("timestamp"),
            "event_found": event is not None,
            "replayable": bool(event and isinstance(event.get("prompt"), str)),
        }
        if event is not None:
            label["historical_engine"] = event.get("selected_engine")
            label["status"] = event.get("status")
        if include_notes and row.get("notes") is not None:
            label["notes"] = row.get("notes")
        labels.append(label)

    return {
        "feedback_labels": len(feedback),
        "expected_engine_counts": dict(sorted(expected_engine_counts.items())),
        "labels": labels[: max(0, max_rows)],
        "truncated": len(labels) > max_rows,
    }


def review_queue(
    *,
    events_path: str | Path,
    feedback_path: str | Path | None,
    max_rows: int = 20,
) -> dict[str, Any]:
    """Build a privacy-safe wrong-route review queue.

    The queue intentionally omits raw prompts, prompt previews, feedback notes,
    request bodies, and secrets. It is a local triage view over event metadata.
    """

    events = _routing_events(read_jsonl(events_path))
    feedback = (
        _feedback_records_by_request(read_jsonl(feedback_path))
        if feedback_path
        else {}
    )
    rows: list[dict[str, Any]] = []
    skipped_labeled = 0
    skipped_private = 0
    for event in reversed(events):
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        if request_id in feedback:
            skipped_labeled += 1
            continue
        replayable = isinstance(event.get("prompt"), str)
        if not replayable:
            skipped_private += 1
        selected_engine = str(event.get("selected_engine") or "")
        row = {
            "request_id": request_id,
            "selected_engine": selected_engine,
            "status": event.get("status"),
            "backend": event.get("backend"),
            "routing_profile": event.get("routing_profile"),
            "receipt_summary": event.get("receipt_summary"),
            "reason_codes": _string_list(event.get("reason_codes")),
            "replayable": replayable,
            "suggested_feedback_command": (
                "model-router feedback "
                f"{request_id} <expected_engine> "
                f"--output {feedback_path or DEFAULT_FEEDBACK_PATH}"
            ),
        }
        rows.append(row)
        if len(rows) >= max_rows:
            break
    return {
        "reviewable": len(rows),
        "items": rows,
        "truncated": len(rows) >= max_rows,
        "skipped_labeled": skipped_labeled,
        "skipped_private": skipped_private,
        "privacy": (
            "Prompts, prompt previews, request bodies, feedback notes, and "
            "secrets are hidden by default."
        ),
    }


def _feedback_records_by_request(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    feedback: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("event_type") != "routing_feedback":
            continue
        request_id = row.get("request_id")
        expected_engine = row.get("expected_engine")
        if isinstance(request_id, str) and isinstance(expected_engine, str):
            feedback[request_id] = row
    return feedback


def _routing_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("event_type") == "routing_event"]


def _events_by_request(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = row.get("request_id")
        if isinstance(request_id, str):
            indexed[request_id] = row
    return indexed


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 6)


def _limit(values: list[str], max_examples: int) -> list[str]:
    if max_examples <= 0:
        return []
    return values[:max_examples]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []
