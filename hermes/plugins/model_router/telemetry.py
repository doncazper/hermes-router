"""Telemetry inspection helpers for routing dogfood loops."""

from __future__ import annotations

from collections import Counter
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from hermes.plugins.model_router.policy import ModelRouter
from hermes.plugins.model_router.pricing_catalog import (
    PRICING_AMBIGUOUS_MODEL,
    PricingCatalog,
    PRICING_MATCHED,
    PRICING_MISSING_MODEL,
    PRICING_MISSING_PRICE,
    estimate_usage_cost,
    load_pricing_catalog,
)
from hermes.plugins.model_router.routing_log import (
    DEFAULT_FEEDBACK_PATH,
    OUTCOME_LABEL_SET,
    read_jsonl,
    redact_text,
)


USAGE_TOKEN_FIELDS = (
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "usage_total_tokens",
    "usage_cached_input_tokens",
)


def replay_events(
    *,
    events_path: str | Path,
    feedback_path: str | Path | None,
    config_path: str | Path | None,
    pricing_catalog_path: str | Path | None = None,
    max_examples: int = 10,
) -> dict[str, Any]:
    events = read_jsonl(events_path)
    feedback_rows = read_jsonl(feedback_path) if feedback_path else []
    feedback = _feedback_records_by_request(feedback_rows)
    router = ModelRouter.from_config(config_path, validate_availability=False)
    pricing_catalog = load_pricing_catalog(pricing_catalog_path)

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
    usage_summary = usage_telemetry_summary(
        routing_events,
        pricing_catalog=pricing_catalog,
    )
    outcome_label_counts = _outcome_label_counts(feedback.values())

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
        "outcome_label_counts": outcome_label_counts,
        **usage_summary,
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
    outcome_label_counts: Counter[str] = Counter()
    for request_id, row in sorted(feedback.items()):
        expected_engine = str(row.get("expected_engine", ""))
        expected_engine_counts[expected_engine] += 1
        outcome_label = _outcome_label(row.get("outcome_label"))
        if outcome_label:
            outcome_label_counts[outcome_label] += 1
        event = event_by_request.get(request_id)
        label: dict[str, Any] = {
            "request_id": request_id,
            "expected_engine": expected_engine,
            "timestamp": row.get("timestamp"),
            "event_found": event is not None,
            "replayable": bool(event and isinstance(event.get("prompt"), str)),
        }
        if outcome_label:
            label["outcome_label"] = outcome_label
        if event is not None:
            label["historical_engine"] = event.get("selected_engine")
            label["status"] = event.get("status")
        if include_notes and row.get("notes") is not None:
            label["notes"] = row.get("notes")
        labels.append(label)

    return {
        "feedback_labels": len(feedback),
        "expected_engine_counts": dict(sorted(expected_engine_counts.items())),
        "outcome_label_counts": dict(sorted(outcome_label_counts.items())),
        "labels": labels[: max(0, max_rows)],
        "truncated": len(labels) > max_rows,
    }


def review_queue(
    *,
    events_path: str | Path,
    feedback_path: str | Path | None,
    pricing_catalog_path: str | Path | None = None,
    max_rows: int = 20,
) -> dict[str, Any]:
    """Build a privacy-safe wrong-route review queue.

    The queue intentionally omits raw prompts, prompt previews, feedback notes,
    request bodies, and secrets. It is a local triage view over event metadata.
    """

    events = _routing_events(read_jsonl(events_path))
    pricing_catalog = load_pricing_catalog(pricing_catalog_path)
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
            "backend_model": _safe_group_key(event.get("backend_model")),
            "upstream_model": _safe_group_key(event.get("upstream_model")),
            "routing_profile": event.get("routing_profile"),
            "receipt_summary": event.get("receipt_summary"),
            "reason_codes": _string_list(event.get("reason_codes")),
            "usage": event_usage_summary(event),
            "cost": event_cost_summary(event, pricing_catalog),
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
    usage_summary = usage_telemetry_summary(
        events,
        pricing_catalog=pricing_catalog,
    )
    return {
        "reviewable": len(rows),
        "items": rows,
        "truncated": len(rows) >= max_rows,
        "skipped_labeled": skipped_labeled,
        "skipped_private": skipped_private,
        "catalog_coverage": usage_summary["catalog_coverage"],
        "catalog_coverage_gaps": usage_summary["catalog_coverage_gaps"],
        "privacy": (
            "Prompts, prompt previews, request bodies, feedback notes, and "
            "secrets are hidden by default."
        ),
    }


def usage_telemetry_summary(
    events: list[dict[str, Any]],
    *,
    pricing_catalog: PricingCatalog | None = None,
) -> dict[str, Any]:
    catalog = pricing_catalog or load_pricing_catalog()
    totals = _empty_usage_totals()
    cost_totals = _empty_cost_totals()
    by_engine: dict[str, dict[str, int]] = {}
    by_backend: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    upstream_model_counts: Counter[str] = Counter()
    pricing_match_counts: Counter[str] = Counter()
    usage_events = 0
    missing_catalog_match_rows = 0
    placeholder_pricing_rows = 0
    insufficient_usage_rows = 0
    catalog_gap_groups: dict[tuple[str, ...], dict[str, Any]] = {}

    for event in events:
        usage = event_usage_summary(event)
        has_usage = any(usage[field] > 0 for field in USAGE_TOKEN_FIELDS)
        if not has_usage:
            insufficient_usage_rows += 1
            continue
        usage_events += 1
        cost = event_cost_summary(event, catalog)
        pricing_status = str(cost.get("pricing_match_status") or "unknown")
        pricing_match_counts[pricing_status] += 1
        if pricing_status in {
            PRICING_AMBIGUOUS_MODEL,
            PRICING_MISSING_MODEL,
            PRICING_MISSING_PRICE,
        }:
            missing_catalog_match_rows += 1
            _merge_catalog_gap_group(catalog_gap_groups, event, usage, cost, pricing_status)
        if _pricing_is_placeholder(cost):
            placeholder_pricing_rows += 1
        _merge_usage_totals(totals, usage)
        _merge_cost_totals(cost_totals, cost)

        engine = _safe_group_key(event.get("selected_engine"))
        if engine:
            group = _group_totals(by_engine, engine)
            _merge_usage_totals(group, usage)
            _merge_cost_totals(group, cost)

        backend = _safe_group_key(event.get("backend") or event.get("selected_backend"))
        if backend:
            group = _group_totals(by_backend, backend)
            _merge_usage_totals(group, usage)
            _merge_cost_totals(group, cost)

        model = _safe_model_key(event)
        if model:
            group = _group_totals(by_model, model)
            _merge_usage_totals(group, usage)
            _merge_cost_totals(group, cost)

        upstream_model = _safe_group_key(event.get("upstream_model"))
        if upstream_model:
            upstream_model_counts[upstream_model] += 1

    return {
        "usage_events": usage_events,
        "usage_prompt_tokens": totals["usage_prompt_tokens"],
        "usage_completion_tokens": totals["usage_completion_tokens"],
        "usage_total_tokens": totals["usage_total_tokens"],
        "usage_cached_input_tokens": totals["usage_cached_input_tokens"],
        "usage_by_selected_engine": _sorted_usage_groups(by_engine),
        "usage_by_backend": _sorted_usage_groups(by_backend),
        "usage_by_model": _sorted_usage_groups(by_model),
        "upstream_model_counts": dict(sorted(upstream_model_counts.items())),
        "pricing_catalog_version": catalog.catalog_version,
        "pricing_catalog_source": catalog.source,
        "pricing_match_counts": dict(sorted(pricing_match_counts.items())),
        "catalog_coverage": _catalog_coverage_summary(
            total_routing_rows=len(events),
            usage_rows=usage_events,
            matched_rows=pricing_match_counts.get(PRICING_MATCHED, 0),
            missing_catalog_match_rows=missing_catalog_match_rows,
            placeholder_pricing_rows=placeholder_pricing_rows,
            estimated_cost_rows=cost_totals["estimated_cost_events"],
            insufficient_usage_rows=insufficient_usage_rows,
            catalog=catalog,
        ),
        "catalog_coverage_gaps": _sorted_catalog_gap_groups(catalog_gap_groups),
        **cost_totals,
    }


def event_usage_summary(event: dict[str, Any]) -> dict[str, Any]:
    usage = _empty_usage_totals()
    for field in USAGE_TOKEN_FIELDS:
        usage[field] = _non_negative_int(event.get(field))
    upstream_model = _safe_group_key(event.get("upstream_model"))
    if upstream_model:
        usage["upstream_model"] = upstream_model
    backend_model = _safe_group_key(event.get("backend_model"))
    if backend_model:
        usage["backend_model"] = backend_model
    return usage


def event_cost_summary(
    event: dict[str, Any],
    pricing_catalog: PricingCatalog | None = None,
) -> dict[str, Any]:
    catalog = pricing_catalog or load_pricing_catalog()
    usage = event_usage_summary(event)
    provider = _safe_group_key(event.get("provider") or event.get("backend_provider"))
    model_candidates = tuple(
        model
        for model in (
            _safe_group_key(event.get("upstream_model")),
            _safe_group_key(event.get("backend_model")),
            _safe_group_key(event.get("selected_model")),
        )
        if model
    )
    return estimate_usage_cost(
        usage,
        catalog,
        provider=provider or None,
        model_candidates=model_candidates,
    )


def _feedback_records_by_request(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    feedback: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("event_type") != "routing_feedback":
            continue
        request_id = row.get("request_id")
        expected_engine = row.get("expected_engine")
        if isinstance(request_id, str) and isinstance(expected_engine, str):
            normalized = dict(row)
            outcome_label = _outcome_label(row.get("outcome_label"))
            if outcome_label:
                normalized["outcome_label"] = outcome_label
            else:
                normalized.pop("outcome_label", None)
            feedback[request_id] = normalized
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


def _empty_usage_totals() -> dict[str, int]:
    return {field: 0 for field in USAGE_TOKEN_FIELDS}


def _empty_cost_totals() -> dict[str, Any]:
    return {
        "estimated_cost_events": 0,
        "estimated_input_cost": 0.0,
        "estimated_output_cost": 0.0,
        "estimated_cached_input_cost": 0.0,
        "estimated_total_cost": 0.0,
        "estimated_cost_currency": None,
    }


def _merge_usage_totals(target: dict[str, int], usage: dict[str, Any]) -> None:
    target["events"] = target.get("events", 0) + 1
    for field in USAGE_TOKEN_FIELDS:
        target[field] = target.get(field, 0) + _non_negative_int(usage.get(field))


def _group_totals(groups: dict[str, dict[str, int]], key: str) -> dict[str, int]:
    if key not in groups:
        groups[key] = {"events": 0, **_empty_usage_totals(), **_empty_cost_totals()}
    return groups[key]


def _merge_catalog_gap_group(
    groups: dict[tuple[str, ...], dict[str, Any]],
    event: dict[str, Any],
    usage: dict[str, Any],
    cost: dict[str, Any],
    pricing_status: str,
) -> None:
    provider = _safe_group_key(event.get("provider") or event.get("backend_provider"))
    backend = _safe_group_key(event.get("backend") or event.get("selected_backend"))
    backend_model = _safe_group_key(event.get("backend_model"))
    upstream_model = _safe_group_key(event.get("upstream_model"))
    selected_engine = _safe_group_key(event.get("selected_engine"))
    model = _safe_group_key(cost.get("pricing_model")) or _safe_model_key(event)
    key = (
        pricing_status,
        provider or "unknown",
        model or "unknown",
        backend or "unknown",
        backend_model or "unknown",
        upstream_model or "unknown",
        selected_engine or "unknown",
    )
    if key not in groups:
        groups[key] = {
            "pricing_match_status": key[0],
            "provider": key[1],
            "model": key[2],
            "backend": key[3],
            "backend_model": key[4],
            "upstream_model": key[5],
            "selected_engine": key[6],
            **_empty_usage_totals(),
            "events": 0,
        }
    _merge_usage_totals(groups[key], usage)


def _merge_cost_totals(target: dict[str, Any], cost: dict[str, Any]) -> None:
    if cost.get("pricing_match_status") != "matched":
        return
    target["estimated_cost_events"] = int(target.get("estimated_cost_events", 0)) + 1
    for field in (
        "estimated_input_cost",
        "estimated_output_cost",
        "estimated_cached_input_cost",
        "estimated_total_cost",
    ):
        target[field] = _round_cost(
            float(target.get(field, 0.0) or 0.0)
            + float(cost.get(field, 0.0) or 0.0)
        )
    target["estimated_cost_currency"] = _merged_currency(
        target.get("estimated_cost_currency"),
        cost.get("estimated_cost_currency"),
    )


def _sorted_usage_groups(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        key: groups[key]
        for key in sorted(
            groups,
            key=lambda item: (
                -groups[item].get("usage_total_tokens", 0),
                -groups[item].get("usage_prompt_tokens", 0),
                item,
            ),
        )
    }


def _sorted_catalog_gap_groups(
    groups: dict[tuple[str, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        groups[key]
        for key in sorted(
            groups,
            key=lambda item: (
                -groups[item].get("events", 0),
                -groups[item].get("usage_total_tokens", 0),
                groups[item].get("pricing_match_status", ""),
                groups[item].get("provider", ""),
                groups[item].get("model", ""),
                groups[item].get("backend", ""),
            ),
        )
    ]


def _safe_model_key(event: dict[str, Any]) -> str:
    return (
        _safe_group_key(event.get("upstream_model"))
        or _safe_group_key(event.get("backend_model"))
        or _safe_group_key(event.get("selected_model"))
    )


def _safe_group_key(value: Any, *, max_chars: int = 160) -> str:
    if not isinstance(value, str):
        return ""
    text = redact_text(value).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _round_cost(value: float) -> float:
    return round(value, 8)


def _merged_currency(current: Any, incoming: Any) -> str | None:
    if not isinstance(incoming, str) or not incoming:
        return current if isinstance(current, str) else None
    if not isinstance(current, str) or not current:
        return incoming
    if current == incoming:
        return current
    return "mixed"


def _catalog_coverage_summary(
    *,
    total_routing_rows: int,
    usage_rows: int,
    matched_rows: int,
    missing_catalog_match_rows: int,
    placeholder_pricing_rows: int,
    estimated_cost_rows: int,
    insufficient_usage_rows: int,
    catalog: PricingCatalog,
) -> dict[str, Any]:
    return {
        "total_routing_rows": total_routing_rows,
        "total_rows_with_usage": usage_rows,
        "rows_with_catalog_match": matched_rows,
        "rows_missing_provider_model_catalog_match": missing_catalog_match_rows,
        "rows_using_placeholder_pricing": placeholder_pricing_rows,
        "rows_with_estimated_cost": estimated_cost_rows,
        "rows_without_enough_usage_data": insufficient_usage_rows,
        "active_catalog_version": catalog.catalog_version,
        "active_catalog_source": catalog.source,
        "cost_confidence": _cost_confidence_label(
            usage_rows=usage_rows,
            matched_rows=matched_rows,
            missing_catalog_match_rows=missing_catalog_match_rows,
            placeholder_pricing_rows=placeholder_pricing_rows,
        ),
    }


def _cost_confidence_label(
    *,
    usage_rows: int,
    matched_rows: int,
    missing_catalog_match_rows: int,
    placeholder_pricing_rows: int,
) -> str:
    if usage_rows <= 0:
        return "no_usage"
    if matched_rows <= 0:
        return "no_catalog_match"
    if missing_catalog_match_rows > 0:
        return "partial_catalog_match"
    if placeholder_pricing_rows > 0:
        return "placeholder_pricing"
    return "catalog_matched"


def _pricing_is_placeholder(cost: dict[str, Any]) -> bool:
    value = cost.get("pricing_is_placeholder")
    if isinstance(value, bool):
        return value
    source = cost.get("pricing_source")
    if isinstance(source, str):
        lowered = source.lower()
        return any(
            signal in lowered
            for signal in (
                "example",
                "placeholder",
                "non-authoritative",
                "not-current-pricing",
            )
        )
    return False


def _outcome_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip()
    if label in OUTCOME_LABEL_SET:
        return label
    return None


def _outcome_label_counts(rows: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = _outcome_label(row.get("outcome_label"))
        if label:
            counts[label] += 1
    return dict(sorted(counts.items()))
