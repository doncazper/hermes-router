import json

import pytest

from hermes.plugins.model_router import cli as model_router_cli
from hermes.plugins.model_router.eval_runner import (
    EvalBackendResponse,
    eval_comparison_summaries_from_rows,
    eval_evidence_for_model,
    eval_evidence_from_rows,
    eval_fixture_summaries,
    eval_report,
    execute_eval_comparison,
    execute_eval_run,
    load_eval_results,
    parse_eval_candidate,
)
from hermes.plugins.model_router.evals import (
    DELEGATION_DIMENSIONS,
    EVAL_CATEGORIES,
    EVAL_FIXTURE_SCHEMA_VERSION,
    EVAL_SCORER_VERSION,
    EvalFixtureError,
    eval_fixture_pack_from_mapping,
    eval_fixture_pack_from_text,
    load_builtin_eval_fixtures,
    score_eval_output,
)
from hermes.plugins.model_router.product import initialize_product_config


def _builtin_fixture(fixture_id):
    pack = load_builtin_eval_fixtures()
    return next(fixture for fixture in pack.fixtures if fixture.id == fixture_id)


def _check(result, check_id):
    return next(check for check in result.checks if check.id == check_id)


def _passing_output_for_fixture(fixture_id):
    if fixture_id == "strict_json_routing_control_decision":
        return '{"route":"balanced","risk":"low","needs_confirmation":false}'
    if fixture_id == "structured_output_schema_following":
        return '{"status":"needs_review","blockers":[],"next_steps":["rerun","review"]}'
    return "- Search for matching files.\n- Run tests and verify behavior.\n- Report risk."


def _write_eval_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _eval_evidence_row(**overrides):
    row = {
        "version": 1,
        "run_id": "evalrun_evidence",
        "created_at": "2026-06-30T12:00:00.000Z",
        "backend": "fast",
        "model": "mock-model",
        "selected_engine": "fast_local",
        "fixture_id": "strict_json_routing_control_decision",
        "category": "structured_output",
        "score_percent": 100.0,
        "weighted_score": 1.0,
        "exit_status": "passed",
        "status": "completed",
        "latency_ms": 10.0,
        "timeout": False,
        "scorer_version": EVAL_SCORER_VERSION,
        "fixture_version": EVAL_FIXTURE_SCHEMA_VERSION,
        "failure_reasons": [],
        "usage_prompt_tokens": 8,
        "usage_completion_tokens": 4,
        "usage_total_tokens": 12,
    }
    row.update(overrides)
    return row


def test_builtin_eval_fixtures_load_and_are_json_safe():
    pack = load_builtin_eval_fixtures()

    assert pack.version == 1
    assert pack.fixture_pack_id == "modelrouter_builtin_suitability"
    assert pack.fixture_pack_version == 1
    assert len(pack.fixtures) == 8
    assert len({fixture.id for fixture in pack.fixtures}) == len(pack.fixtures)
    assert {fixture.category for fixture in pack.fixtures} <= set(EVAL_CATEGORIES)
    for fixture in pack.fixtures:
        assert fixture.prompt_hash
        assert fixture.privacy_level == "hash_only"
        assert set(fixture.delegation_dimensions) == set(DELEGATION_DIMENSIONS)
        assert all(isinstance(value, bool) for value in fixture.delegation_dimensions.values())
        assert "sk-" not in fixture.prompt.lower()
    json.dumps(pack.to_dict())


def test_builtin_eval_fixtures_cover_initial_categories():
    pack = load_builtin_eval_fixtures()
    fixture_ids = {fixture.id for fixture in pack.fixtures}
    categories = {fixture.category for fixture in pack.fixtures}

    assert {
        "strict_json_routing_control_decision",
        "reasoning_leakage_guard",
        "mechanical_bulk_edit_suitability",
        "code_review_judgment",
        "risky_action_refusal",
        "verification_heavy_task",
        "long_context_slow_test_suite_proxy",
        "structured_output_schema_following",
    } <= fixture_ids
    assert {
        "structured_output",
        "no_reasoning_leakage",
        "mechanical_edit",
        "code_review_judgment",
        "risky_action_refusal",
        "verification_heavy_task",
        "slow_long_context_task",
    } <= categories


def test_eval_fixture_schema_rejects_missing_required_field():
    payload = {
        "version": 1,
        "fixture_pack_id": "bad_pack",
        "fixture_pack_version": 1,
        "fixtures": [
            {
                "id": "missing_prompt",
                "name": "Missing prompt",
                "category": "structured_output",
                "task_profile": "schema_following",
                "required_patterns": [],
                "forbidden_patterns": [],
                "expected_json_keys": [],
                "expected_bullet_count": None,
                "max_non_empty_lines": None,
                "weight": 1.0,
                "privacy_level": "hash_only",
                "delegation_dimensions": {
                    dimension: False for dimension in DELEGATION_DIMENSIONS
                },
                "notes": ["invalid"],
            }
        ],
    }

    with pytest.raises(EvalFixtureError, match="prompt"):
        eval_fixture_pack_from_mapping(payload, source="test")


def test_eval_fixture_schema_rejects_invalid_regex():
    text = """
version: 1
fixture_pack_id: bad_pack
fixture_pack_version: 1
fixtures:
  - id: bad_regex
    name: Bad regex
    category: structured_output
    task_profile: schema_following
    prompt: Return JSON.
    required_patterns:
      - '['
    forbidden_patterns: []
    expected_json_keys: []
    expected_bullet_count:
    max_non_empty_lines:
    weight: 1.0
    privacy_level: hash_only
    delegation_dimensions:
      mechanical_work_likely: false
      judgment_heavy_likely: false
      verification_heavy_likely: false
      repo_wide_likely: false
      risky_or_external_action: false
      ambiguity_sensitive: false
    notes:
      - invalid
"""

    with pytest.raises(EvalFixtureError, match="required_patterns"):
        eval_fixture_pack_from_text(text, source="bad.yaml")


def test_eval_fixture_schema_rejects_unknown_delegation_dimension():
    payload = {
        "version": 1,
        "fixture_pack_id": "bad_pack",
        "fixture_pack_version": 1,
        "fixtures": [
            {
                "id": "extra_dimension",
                "name": "Extra dimension",
                "category": "structured_output",
                "task_profile": "schema_following",
                "prompt": "Return JSON.",
                "required_patterns": [],
                "forbidden_patterns": [],
                "expected_json_keys": [],
                "expected_bullet_count": None,
                "max_non_empty_lines": None,
                "weight": 1.0,
                "privacy_level": "hash_only",
                "delegation_dimensions": {
                    **{dimension: False for dimension in DELEGATION_DIMENSIONS},
                    "surprise": True,
                },
                "notes": ["invalid"],
            }
        ],
    }

    with pytest.raises(EvalFixtureError, match="unknown delegation dimensions"):
        eval_fixture_pack_from_mapping(payload, source="test")


def test_eval_fixture_summaries_do_not_include_prompt_bodies():
    summaries = eval_fixture_summaries(category="structured_output")

    assert len(summaries) == 2
    serialized = json.dumps(summaries)
    assert "prompt_hash" in serialized
    assert "Return only JSON" not in serialized
    assert "prompt" not in summaries[0]


def test_eval_scoring_passes_strict_json_fixture_without_raw_output():
    fixture = _builtin_fixture("strict_json_routing_control_decision")
    result = score_eval_output(
        fixture,
        '{"route":"balanced","risk":"low","needs_confirmation":false}',
    )

    assert result.fixture_id == fixture.id
    assert result.passed is True
    assert result.score_percent == 100.0
    assert result.weighted_score == 1.0
    assert result.passed_checks == result.total_checks
    assert _check(result, "valid_json").passed is True
    assert _check(result, "exact_json_keys").passed is True
    payload = result.to_dict()
    json.dumps(payload)
    serialized = json.dumps(payload)
    assert fixture.prompt not in serialized
    assert '"route":"balanced"' not in serialized


def test_eval_scoring_missing_output_fails_non_empty_check():
    fixture = _builtin_fixture("reasoning_leakage_guard")
    result = score_eval_output(fixture, "")

    assert result.passed is False
    assert _check(result, "non_empty_output").passed is False
    assert "Output was empty." in result.failure_reasons


def test_eval_scoring_malformed_json_fails_valid_json_check():
    fixture = _builtin_fixture("strict_json_routing_control_decision")
    result = score_eval_output(
        fixture,
        'route: balanced, risk: low, needs_confirmation: false',
    )

    assert result.passed is False
    assert _check(result, "valid_json").passed is False
    assert _check(result, "exact_json_keys").passed is False
    assert "Output did not parse as a JSON object." in result.failure_reasons


def test_eval_scoring_extra_json_keys_fail_exact_key_check():
    fixture = _builtin_fixture("strict_json_routing_control_decision")
    result = score_eval_output(
        fixture,
        (
            '{"route":"balanced","risk":"low","needs_confirmation":false,'
            '"extra":true}'
        ),
    )

    assert result.passed is False
    assert _check(result, "valid_json").passed is True
    assert _check(result, "exact_json_keys").passed is False
    assert (
        "JSON keys did not match expected keys exactly."
        in result.failure_reasons
    )


def test_eval_scoring_forbidden_reasoning_tags_fail():
    fixture = _builtin_fixture("reasoning_leakage_guard")
    result = score_eval_output(
        fixture,
        "<analysis>scratchpad</analysis>\nCheck policy evidence first.",
    )

    assert result.passed is False
    assert _check(result, "forbidden_pattern_1").passed is False
    assert _check(result, "reasoning_leakage_absent").passed is False
    assert "Reasoning leakage markers were detected." in result.failure_reasons


def test_eval_scoring_bullet_count_and_max_lines():
    fixture = _builtin_fixture("mechanical_bulk_edit_suitability")
    result = score_eval_output(
        fixture,
        "\n".join(
            [
                "- Search for every old key reference.",
                "- Update matching config names.",
                "- Run tests and verify behavior.",
                "- Write an extra note.",
            ]
        ),
    )

    assert result.passed is False
    assert _check(result, "expected_bullet_count").passed is False
    assert _check(result, "max_non_empty_lines").passed is False
    assert "Expected 3 bullets, found 4." in result.failure_reasons


def test_eval_scoring_required_pattern_missing_fails():
    fixture = _builtin_fixture("code_review_judgment")
    result = score_eval_output(
        fixture,
        "- This risk needs careful review.\n- Add more tests before merging.",
    )

    assert result.passed is False
    assert _check(result, "required_pattern_0").passed is False
    assert _check(result, "required_pattern_1").passed is True


def test_eval_scoring_timeout_or_failure_status_fails_status_check():
    fixture = _builtin_fixture("structured_output_schema_following")
    result = score_eval_output(
        fixture,
        '{"status":"needs_review","blockers":[],"next_steps":["rerun","review"]}',
        status="error",
        timed_out=True,
        error="timeout",
    )

    assert result.passed is False
    assert result.status == "timeout"
    assert _check(result, "status_ok").passed is False
    assert any(
        reason.startswith("Eval execution did not complete cleanly")
        for reason in result.failure_reasons
    )


def test_eval_runner_runs_one_fixture_with_mocked_backend_and_writes_jsonl(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "evals" / "results.jsonl"
    calls = []

    def runner(request):
        calls.append(request)
        return EvalBackendResponse(
            status="completed",
            output=_passing_output_for_fixture("strict_json_routing_control_decision"),
            latency_ms=12.5,
            usage_prompt_tokens=10,
            usage_completion_tokens=6,
            usage_total_tokens=16,
            upstream_model="mock-upstream",
        )

    execution = execute_eval_run(
        config_path=tmp_path / "routing_proxy.yaml",
        backend="fast",
        model="mock-model",
        fixture_selector="strict_json_routing_control_decision",
        output_path=output,
        run_id="evalrun_test_one",
        runner=runner,
    )

    assert execution.ok is True
    assert execution.passed is True
    assert len(calls) == 1
    assert calls[0].backend == "fast"
    assert calls[0].model == "mock-model"
    assert calls[0].selected_engine == "fast_local"
    rows = load_eval_results(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "evalrun_test_one"
    assert row["fixture_id"] == "strict_json_routing_control_decision"
    assert row["score_percent"] == 100.0
    assert row["usage_prompt_tokens"] == 10
    assert row["usage_completion_tokens"] == 6
    assert row["usage_total_tokens"] == 16
    serialized = output.read_text(encoding="utf-8")
    assert calls[0].prompt not in serialized
    assert '"route":"balanced"' not in serialized
    assert "secret-value" not in serialized


def test_eval_runner_runs_category_and_all_with_mocked_backend(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "results.jsonl"

    def runner(request):
        fixture_id = "structured_output_schema_following"
        if "route, risk, and needs_confirmation" in request.prompt:
            fixture_id = "strict_json_routing_control_decision"
        return EvalBackendResponse(
            status="completed",
            output=_passing_output_for_fixture(fixture_id),
            latency_ms=1.0,
        )

    category_run = execute_eval_run(
        config_path=tmp_path / "routing_proxy.yaml",
        backend="fast",
        fixture_selector="structured_output",
        output_path=output,
        run_id="evalrun_category",
        runner=runner,
    )
    all_run = execute_eval_run(
        config_path=tmp_path / "routing_proxy.yaml",
        backend="fast",
        all_fixtures=True,
        output_path=output,
        run_id="evalrun_all",
        confirm_large_run=True,
        runner=runner,
    )

    assert len(category_run.results) == 2
    assert {result.category for result in category_run.results} == {"structured_output"}
    assert len(all_run.results) == 8
    rows = load_eval_results(output)
    assert len(rows) == 10


def test_eval_runner_blocks_full_fixture_sweep_without_confirmation(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "results.jsonl"

    with pytest.raises(EvalFixtureError, match="confirm-large-run"):
        execute_eval_run(
            config_path=tmp_path / "routing_proxy.yaml",
            backend="fast",
            all_fixtures=True,
            output_path=output,
            run_id="evalrun_blocked_all",
            runner=lambda _request: EvalBackendResponse(
                status="completed",
                output="- one\n- two\n- three",
            ),
        )

    assert not output.exists()


def test_eval_candidate_parser_requires_explicit_backend_and_model():
    candidate = parse_eval_candidate("fast:qwen2.5-coder:7b")

    assert candidate.backend == "fast"
    assert candidate.model == "qwen2.5-coder:7b"
    assert candidate.candidate_id == "fast:qwen2.5-coder:7b"

    with pytest.raises(EvalFixtureError, match="backend:model"):
        parse_eval_candidate("fast")

    with pytest.raises(EvalFixtureError, match="backend:model"):
        parse_eval_candidate(":missing-backend")


def test_eval_compare_runs_explicit_candidates_and_summarizes_winner(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "comparison-results.jsonl"
    calls = []

    def runner(request):
        calls.append(request)
        if request.model == "good-model":
            return EvalBackendResponse(
                status="completed",
                output=_passing_output_for_fixture(
                    "strict_json_routing_control_decision"
                ),
                latency_ms=5.0,
                usage_prompt_tokens=9,
                usage_completion_tokens=4,
                usage_total_tokens=13,
            )
        return EvalBackendResponse(
            status="completed",
            output="not json",
            latency_ms=12.0,
        )

    execution = execute_eval_comparison(
        config_path=tmp_path / "routing_proxy.yaml",
        candidates=("fast:good-model", "balanced:weak-model"),
        fixture_selector="strict_json_routing_control_decision",
        output_path=output,
        comparison_id="evalcompare_test",
        runner=runner,
    )
    payload = execution.to_dict()
    serialized = json.dumps(payload)

    assert execution.ok is True
    assert execution.passed is False
    assert execution.request_count == 2
    assert len(calls) == 2
    assert calls[0].backend == "fast"
    assert calls[1].backend == "balanced"
    assert payload["winner"]["candidate"] == "fast:good-model"
    assert payload["winner"]["label"] == "best on this fixture set/profile"
    assert payload["candidate_summaries"][0]["usage_summary"]["usage_total_tokens"] == 13
    assert payload["candidate_summaries"][1]["usage_summary"]["rows_missing_usage"] == 1
    rows = load_eval_results(output)
    assert len(rows) == 2
    assert {row["comparison_id"] for row in rows} == {"evalcompare_test"}
    assert {row["candidate"] for row in rows} == {
        "fast:good-model",
        "balanced:weak-model",
    }
    assert calls[0].prompt not in serialized
    assert '"route":"balanced"' not in serialized
    assert "not json" not in serialized


def test_eval_compare_reports_no_winner_for_all_failed_tie(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    execution = execute_eval_comparison(
        config_path=tmp_path / "routing_proxy.yaml",
        candidates=("fast:model-a", "balanced:model-b"),
        fixture_selector="strict_json_routing_control_decision",
        output_path=tmp_path / "failed-tie.jsonl",
        comparison_id="evalcompare_failed_tie",
        runner=lambda _request: EvalBackendResponse(
            status="completed",
            output="not json",
            latency_ms=10.0,
        ),
    )

    assert execution.passed is False
    assert execution.winner is None
    assert execution.to_dict()["winner"] is None


def test_eval_compare_persisted_error_messages_redact_prompt_and_output(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "redacted-errors.jsonl"

    def runner(request):
        model_output = _passing_output_for_fixture(
            "strict_json_routing_control_decision"
        )
        return EvalBackendResponse(
            status="failed",
            output=model_output,
            latency_ms=1.0,
            error_type="ProviderError",
            error_message=(
                f"provider echoed prompt: {request.prompt} "
                f"and output: {model_output}"
            ),
        )

    execute_eval_comparison(
        config_path=tmp_path / "routing_proxy.yaml",
        candidates=("fast:model-a", "balanced:model-b"),
        fixture_selector="strict_json_routing_control_decision",
        output_path=output,
        comparison_id="evalcompare_redaction",
        runner=runner,
    )
    serialized = output.read_text(encoding="utf-8")

    assert "provider echoed prompt" in serialized
    assert "[redacted]" in serialized
    assert "Return only JSON" not in serialized
    assert '"route":"balanced"' not in serialized


def test_eval_compare_runs_category_across_explicit_candidates(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "category-comparison.jsonl"

    def runner(request):
        fixture_id = "structured_output_schema_following"
        if "route, risk, and needs_confirmation" in request.prompt:
            fixture_id = "strict_json_routing_control_decision"
        return EvalBackendResponse(
            status="completed",
            output=_passing_output_for_fixture(fixture_id),
            latency_ms=2.0,
        )

    execution = execute_eval_comparison(
        config_path=tmp_path / "routing_proxy.yaml",
        candidates=("fast:model-a", "balanced:model-b"),
        fixture_selector="structured_output",
        output_path=output,
        comparison_id="evalcompare_category",
        runner=runner,
    )
    rows = load_eval_results(output)

    assert execution.fixture_count == 2
    assert execution.request_count == 4
    assert len(rows) == 4
    assert {row["category"] for row in rows} == {"structured_output"}
    assert all(
        summary.usage_summary["rows_missing_usage"] == 2
        for summary in execution.candidate_summaries
    )


def test_eval_compare_blocks_broad_comparison_without_confirmation(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "blocked-comparison.jsonl"

    with pytest.raises(EvalFixtureError, match="confirm-large-run"):
        execute_eval_comparison(
            config_path=tmp_path / "routing_proxy.yaml",
            candidates=("fast:model-a", "balanced:model-b"),
            all_fixtures=True,
            output_path=output,
            comparison_id="evalcompare_blocked",
            runner=lambda _request: EvalBackendResponse(
                status="completed",
                output="- one\n- two\n- three",
            ),
        )

    assert not output.exists()


def test_eval_compare_rejects_duplicate_candidates(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    with pytest.raises(EvalFixtureError, match="duplicate"):
        execute_eval_comparison(
            config_path=tmp_path / "routing_proxy.yaml",
            candidates=("fast:model-a", "fast:model-a"),
            fixture_selector="strict_json_routing_control_decision",
            output_path=tmp_path / "duplicate-comparison.jsonl",
        )


def test_eval_compare_blocks_high_request_count_without_confirmation(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "blocked-threshold-comparison.jsonl"
    called = False

    def runner(_request):
        nonlocal called
        called = True
        return EvalBackendResponse(status="completed", output="{}")

    with pytest.raises(EvalFixtureError, match="10 backend requests"):
        execute_eval_comparison(
            config_path=tmp_path / "routing_proxy.yaml",
            candidates=(
                "fast:model-a",
                "balanced:model-b",
                "reasoning:model-c",
                "code:model-d",
                "fast:model-e",
            ),
            fixture_selector="structured_output",
            output_path=output,
            comparison_id="evalcompare_threshold_blocked",
            runner=runner,
        )

    assert called is False
    assert not output.exists()


def test_eval_comparison_summaries_mark_mixed_versions_stale():
    rows = [
        _eval_evidence_row(
            comparison_id="evalcompare_mixed_versions",
            candidate="fast:model-a",
            model="model-a",
            fixture_id="strict_json_routing_control_decision",
            fixture_version=EVAL_FIXTURE_SCHEMA_VERSION,
            scorer_version=EVAL_SCORER_VERSION,
        ),
        _eval_evidence_row(
            comparison_id="evalcompare_mixed_versions",
            candidate="balanced:model-b",
            backend="balanced",
            model="model-b",
            fixture_id="strict_json_routing_control_decision",
        )
        | {
            "fixture_version": None,
            "scorer_version": None,
        },
    ]
    rows[1] = {key: value for key, value in rows[1].items() if value is not None}

    summary = eval_comparison_summaries_from_rows(rows)[0]

    assert summary["status"] == "stale"
    assert summary["winner"] is None
    assert any("Fixture version mismatch" in reason for reason in summary["stale_reasons"])
    assert any("Scorer version mismatch" in reason for reason in summary["stale_reasons"])


def test_eval_comparison_summaries_detect_incomplete_candidate_coverage():
    rows = [
        _eval_evidence_row(
            comparison_id="evalcompare_incomplete",
            candidate="fast:model-a",
            model="model-a",
            fixture_id="strict_json_routing_control_decision",
            category="structured_output",
        ),
        _eval_evidence_row(
            comparison_id="evalcompare_incomplete",
            candidate="balanced:model-b",
            backend="balanced",
            model="model-b",
            fixture_id="strict_json_routing_control_decision",
            category="structured_output",
        ),
        _eval_evidence_row(
            comparison_id="evalcompare_incomplete",
            candidate="fast:model-a",
            model="model-a",
            fixture_id="structured_output_schema_following",
            category="structured_output",
        ),
    ]

    summary = eval_comparison_summaries_from_rows(rows)[0]

    assert summary["status"] == "stale"
    assert summary["winner"] is None
    assert any("coverage mismatch" in reason for reason in summary["stale_reasons"])


def test_eval_comparison_summaries_support_old_run_id_shape_and_filter_non_finite():
    rows = [
        _eval_evidence_row(
            run_id="evalcompare_legacy_01_fast_model_a",
            candidate=None,
            comparison_id=None,
            model="model-a",
            score_percent=float("nan"),
            weighted_score=float("inf"),
            latency_ms=float("inf"),
        ),
        _eval_evidence_row(
            run_id="evalcompare_legacy_02_balanced_model_b",
            candidate=None,
            comparison_id=None,
            backend="balanced",
            model="model-b",
            score_percent=80.0,
            weighted_score=0.8,
            latency_ms=4.0,
        ),
    ]
    rows = [
        {key: value for key, value in row.items() if value is not None}
        for row in rows
    ]

    summary = eval_comparison_summaries_from_rows(rows)[0]
    serialized = json.dumps(summary, allow_nan=False)

    assert summary["comparison_id"] == "evalcompare_legacy"
    assert summary["candidate_count"] == 2
    model_a = next(
        candidate
        for candidate in summary["candidates"]
        if candidate["model"] == "model-a"
    )
    assert model_a["score_mean_percent"] is None
    assert model_a["latency_summary"]["mean_ms"] is None
    assert "Infinity" not in serialized
    assert "NaN" not in serialized


def test_eval_report_latest_is_privacy_safe(tmp_path):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "results.jsonl"

    execute_eval_run(
        config_path=tmp_path / "routing_proxy.yaml",
        backend="fast",
        fixture_selector="strict_json_routing_control_decision",
        output_path=output,
        run_id="evalrun_report",
        runner=lambda _request: EvalBackendResponse(
            status="completed",
            output=_passing_output_for_fixture("strict_json_routing_control_decision"),
            latency_ms=2.0,
        ),
    )

    report = eval_report("latest", result_path=output)
    payload = report.to_dict()
    serialized = json.dumps(payload)

    assert payload["run_id"] == "evalrun_report"
    assert payload["backend"] == "fast"
    assert payload["model"] == "lmstudio-fast-model"
    assert payload["selected_engine"] == "fast_local"
    assert payload["total"] == 1
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["timeouts"] == 0
    assert payload["score_mean_percent"] == 100.0
    assert payload["weighted_score_mean"] == 1.0
    assert payload["latency_summary"]["mean_ms"] == 2.0
    assert payload["usage_summary"]["rows_missing_usage"] == 1
    assert payload["privacy"]["raw_prompts_retained"] is False
    assert any(
        "fixture set/profile" in note
        for note in payload["suitability_notes"]
    )
    assert "strict_json_routing_control_decision" in serialized
    assert "Return only JSON" not in serialized
    assert '"route":"balanced"' not in serialized


def test_eval_report_latest_ignores_comparison_candidate_runs(tmp_path):
    output = tmp_path / "latest-with-comparison.jsonl"
    _write_eval_rows(
        output,
        [
            _eval_evidence_row(
                run_id="evalrun_single",
                created_at="2026-06-30T12:00:00.000Z",
                model="single-model",
            ),
            _eval_evidence_row(
                run_id="evalcompare_newer_01_fast_model",
                comparison_id="evalcompare_newer",
                candidate="fast:comparison-model",
                created_at="2026-06-30T12:10:00.000Z",
                model="comparison-model",
            ),
        ],
    )

    payload = eval_report("latest", result_path=output).to_dict()

    assert payload["run_id"] == "evalrun_single"
    assert payload["model"] == "single-model"


def test_eval_report_summarizes_partial_failures_timeouts_and_usage(tmp_path):
    output = tmp_path / "results.jsonl"
    _write_eval_rows(
        output,
        [
            {
                "version": 1,
                "run_id": "evalrun_partial",
                "created_at": "2026-06-30T12:00:00.000Z",
                "backend": "fast",
                "model": "mock-model",
                "selected_engine": "fast_local",
                "fixture_id": "strict_json_routing_control_decision",
                "category": "structured_output",
                "score_percent": 100.0,
                "weighted_score": 1.0,
                "exit_status": "passed",
                "status": "completed",
                "latency_ms": 10.0,
                "usage_prompt_tokens": 8,
                "usage_completion_tokens": 4,
                "usage_total_tokens": 12,
                "timeout": False,
                "failure_reasons": [],
            },
            {
                "version": 1,
                "run_id": "evalrun_partial",
                "created_at": "2026-06-30T12:00:01.000Z",
                "backend": "fast",
                "model": "mock-model",
                "selected_engine": "fast_local",
                "fixture_id": "structured_output_schema_following",
                "category": "structured_output",
                "score_percent": 25.0,
                "weighted_score": 0.25,
                "exit_status": "failed",
                "status": "timeout",
                "latency_ms": 1000.0,
                "timeout": True,
                "failure_reasons": [
                    "Eval execution did not complete cleanly: timeout.",
                    "Output was empty.",
                ],
            },
        ],
    )

    payload = eval_report("evalrun_partial", result_path=output).to_dict()

    assert payload["total"] == 2
    assert payload["passed"] == 1
    assert payload["failed"] == 1
    assert payload["timeouts"] == 1
    assert payload["score_mean_percent"] == 62.5
    assert payload["weighted_score_mean"] == 0.625
    assert payload["latency_summary"]["mean_ms"] == 505.0
    assert payload["latency_summary"]["median_ms"] == 505.0
    assert payload["usage_summary"]["rows_with_usage"] == 1
    assert payload["usage_summary"]["rows_missing_usage"] == 1
    assert payload["usage_summary"]["usage_total_tokens"] == 12
    assert payload["by_category"]["structured_output"]["timeouts"] == 1
    assert payload["top_failure_reasons"][0] == {
        "reason": "Eval execution did not complete cleanly: timeout.",
        "count": 1,
    }
    assert any("Timeouts suggest" in note for note in payload["suitability_notes"])


def test_eval_report_tolerates_old_rows_with_missing_fields(tmp_path):
    output = tmp_path / "old-results.jsonl"
    _write_eval_rows(
        output,
        [
            {
                "run_id": "evalrun_old",
                "created_at": "2026-06-30T12:00:00.000Z",
                "backend": "fast",
                "model": "legacy-model",
                "fixture_id": "legacy_fixture",
                "category": "legacy",
            }
        ],
    )

    payload = eval_report("evalrun_old", result_path=output).to_dict()
    serialized = json.dumps(payload)

    assert payload["total"] == 1
    assert payload["passed"] == 0
    assert payload["failed"] == 0
    assert payload["unknown"] == 1
    assert payload["score_mean_percent"] is None
    assert payload["latency_summary"]["missing"] == 1
    assert payload["usage_summary"]["rows_missing_usage"] == 1
    assert payload["by_category"]["legacy"]["failed"] == 0
    assert "Return only JSON" not in serialized
    assert "secret-value" not in serialized


def test_eval_evidence_for_model_reports_cached_advisory_summary(tmp_path):
    output = tmp_path / "evidence-results.jsonl"
    _write_eval_rows(
        output,
        [
            _eval_evidence_row(),
            _eval_evidence_row(
                fixture_id="code_review_judgment",
                category="code_review_judgment",
                score_percent=50.0,
                weighted_score=0.5,
                exit_status="failed",
                failure_reasons=["Required pattern was missing."],
                latency_ms=30.0,
                usage_prompt_tokens=10,
                usage_completion_tokens=5,
                usage_total_tokens=15,
            ),
        ],
    )

    evidence = eval_evidence_for_model(
        "mock-model",
        result_path=output,
        backend="fast",
    )
    serialized = json.dumps(evidence)

    assert evidence["status"] == "evaluated"
    assert evidence["stale"] is False
    assert evidence["latest_run_id"] == "evalrun_evidence"
    assert evidence["fixture_count"] == 2
    assert evidence["passed"] == 1
    assert evidence["failed"] == 1
    assert evidence["score_mean_percent"] == 75.0
    assert evidence["weighted_score_mean"] == 0.75
    assert evidence["usage_summary"]["usage_total_tokens"] == 27
    assert evidence["by_category"]["structured_output"]["passed"] == 1
    assert evidence["by_category"]["code_review_judgment"]["failed"] == 1
    assert evidence["top_failure_reasons"][0] == {
        "reason": "Required pattern was missing.",
        "count": 1,
    }
    assert any("advisory" in note for note in evidence["notes"])
    assert "Return only JSON" not in serialized
    assert '"route":"balanced"' not in serialized


def test_eval_evidence_missing_model_is_not_an_error():
    evidence = eval_evidence_from_rows("absent-model", ())

    assert evidence["status"] == "not_evaluated"
    assert evidence["stale"] is True
    assert evidence["fixture_count"] == 0
    assert evidence["by_category"] == {}
    assert any("does not block routing" in note for note in evidence["notes"])


def test_eval_evidence_marks_old_or_mismatched_rows_stale():
    evidence = eval_evidence_from_rows(
        "legacy-model",
        [
            {
                "run_id": "evalrun_old",
                "created_at": "2026-06-30T12:00:00.000Z",
                "backend": "fast",
                "model": "legacy-model",
                "fixture_id": "legacy_fixture",
                "category": "legacy",
                "score_percent": 80.0,
                "weighted_score": 0.8,
                "exit_status": "passed",
                "status": "completed",
            }
        ],
        backend="fast",
    )

    assert evidence["status"] == "stale"
    assert evidence["stale"] is True
    assert any("Fixture version mismatch" in reason for reason in evidence["stale_reasons"])
    assert any("Scorer version mismatch" in reason for reason in evidence["stale_reasons"])


def test_eval_evidence_cli_is_privacy_safe(tmp_path, capsys):
    output = tmp_path / "cli-evidence.jsonl"
    _write_eval_rows(output, [_eval_evidence_row()])

    json_code = model_router_cli.main(
        [
            "eval",
            "evidence",
            "--model",
            "mock-model",
            "--results",
            str(output),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    readable_code = model_router_cli.main(
        [
            "eval",
            "evidence",
            "--model",
            "missing-model",
            "--results",
            str(output),
        ]
    )
    readable = capsys.readouterr().out

    assert json_code == 0
    assert payload["status"] == "evaluated"
    assert payload["model"] == "mock-model"
    assert readable_code == 0
    assert "ModelRouter Eval Evidence" in readable
    assert "Status: not_evaluated" in readable
    assert "Return only JSON" not in readable
    assert '"route":"balanced"' not in readable


def test_eval_report_cli_supports_latest_and_explicit_run_id(tmp_path, capsys):
    output = tmp_path / "cli-results.jsonl"
    _write_eval_rows(
        output,
        [
            {
                "run_id": "evalrun_first",
                "created_at": "2026-06-30T12:00:00.000Z",
                "backend": "fast",
                "model": "first-model",
                "fixture_id": "one",
                "category": "structured_output",
                "score_percent": 40.0,
                "weighted_score": 0.4,
                "exit_status": "failed",
                "status": "completed",
                "timeout": False,
                "failure_reasons": ["Output was empty."],
            },
            {
                "run_id": "evalrun_second",
                "created_at": "2026-06-30T12:01:00.000Z",
                "backend": "fast",
                "model": "second-model",
                "fixture_id": "two",
                "category": "structured_output",
                "score_percent": 100.0,
                "weighted_score": 1.0,
                "exit_status": "passed",
                "status": "completed",
                "timeout": False,
                "failure_reasons": [],
            },
        ],
    )

    latest_code = model_router_cli.main(
        ["eval", "report", "latest", "--results", str(output), "--json"]
    )
    latest_payload = json.loads(capsys.readouterr().out)
    explicit_code = model_router_cli.main(
        ["eval", "report", "evalrun_first", "--results", str(output)]
    )
    readable = capsys.readouterr().out

    assert latest_code == 0
    assert latest_payload["run_id"] == "evalrun_second"
    assert latest_payload["model"] == "second-model"
    assert explicit_code == 0
    assert "ModelRouter Eval Report" in readable
    assert "Run id: evalrun_first" in readable
    assert "Privacy:" in readable
    assert "Output was empty." in readable
    assert "Return only JSON" not in readable


def test_eval_run_cli_invokes_mocked_backend_and_writes_json(
    tmp_path,
    monkeypatch,
    capsys,
):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "results.jsonl"

    def runner(request):
        return EvalBackendResponse(
            status="completed",
            output=_passing_output_for_fixture("strict_json_routing_control_decision"),
            latency_ms=3.0,
        )

    monkeypatch.setattr(
        "hermes.plugins.model_router.eval_runner.run_backend_eval_request",
        runner,
    )

    exit_code = model_router_cli.main(
        [
            "eval",
            "run",
            "--json",
            "--config",
            str(tmp_path / "routing_proxy.yaml"),
            "--backend",
            "fast",
            "--model",
            "mock-model",
            "--fixture",
            "strict_json_routing_control_decision",
            "--output",
            str(output),
            "--run-id",
            "evalrun_cli",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "evalrun_cli"
    assert payload["results"][0]["score_percent"] == 100.0
    assert output.exists()


def test_eval_compare_cli_invokes_mocked_candidates_and_writes_json(
    tmp_path,
    monkeypatch,
    capsys,
):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "compare-cli-results.jsonl"

    def runner(request):
        if request.model == "model-a":
            return EvalBackendResponse(
                status="completed",
                output=_passing_output_for_fixture(
                    "strict_json_routing_control_decision"
                ),
                latency_ms=3.0,
                usage_total_tokens=10,
            )
        return EvalBackendResponse(
            status="completed",
            output="not json",
            latency_ms=7.0,
        )

    monkeypatch.setattr(
        "hermes.plugins.model_router.eval_runner.run_backend_eval_request",
        runner,
    )

    exit_code = model_router_cli.main(
        [
            "eval",
            "compare",
            "--json",
            "--config",
            str(tmp_path / "routing_proxy.yaml"),
            "--candidate",
            "fast:model-a",
            "--candidate",
            "balanced:model-b",
            "--fixture",
            "strict_json_routing_control_decision",
            "--output",
            str(output),
            "--comparison-id",
            "evalcompare_cli",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    serialized = json.dumps(payload)

    assert exit_code == 0
    assert payload["comparison_id"] == "evalcompare_cli"
    assert payload["request_count"] == 2
    assert payload["winner"]["candidate"] == "fast:model-a"
    assert payload["winner"]["label"] == "best on this fixture set/profile"
    assert payload["candidate_summaries"][0]["usage_summary"]["usage_total_tokens"] == 10
    assert output.exists()
    assert "Return only JSON" not in serialized
    assert '"route":"balanced"' not in serialized
    assert "not json" not in serialized


def test_eval_compare_cli_human_output_handles_zero_latency(
    tmp_path,
    monkeypatch,
    capsys,
):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    def runner(_request):
        return EvalBackendResponse(
            status="completed",
            output=_passing_output_for_fixture("strict_json_routing_control_decision"),
            latency_ms=0.0,
        )

    monkeypatch.setattr(
        "hermes.plugins.model_router.eval_runner.run_backend_eval_request",
        runner,
    )

    exit_code = model_router_cli.main(
        [
            "eval",
            "compare",
            "--config",
            str(tmp_path / "routing_proxy.yaml"),
            "--candidate",
            "fast:model-a",
            "--candidate",
            "balanced:model-b",
            "--fixture",
            "strict_json_routing_control_decision",
            "--output",
            str(tmp_path / "human-compare.jsonl"),
        ]
    )
    readable = capsys.readouterr().out

    assert exit_code == 0
    assert "ModelRouter Eval Comparison" in readable
    assert "Winner: none" in readable
    assert "mean_ms=0.0" in readable
    assert "best on this fixture set/profile" in readable
    assert "Return only JSON" not in readable


def test_eval_compare_cli_rejects_invalid_candidate_format(tmp_path, capsys):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    exit_code = model_router_cli.main(
        [
            "eval",
            "compare",
            "--json",
            "--config",
            str(tmp_path / "routing_proxy.yaml"),
            "--candidate",
            "fast",
            "--candidate",
            "balanced:model-b",
            "--fixture",
            "strict_json_routing_control_decision",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert "backend:model" in payload["error"]


def test_eval_compare_cli_requires_confirmation_for_broad_comparison(
    tmp_path,
    monkeypatch,
    capsys,
):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )

    def runner(_request):
        raise AssertionError("blocked comparison should not call a backend")

    monkeypatch.setattr(
        "hermes.plugins.model_router.eval_runner.run_backend_eval_request",
        runner,
    )

    exit_code = model_router_cli.main(
        [
            "eval",
            "compare",
            "--json",
            "--config",
            str(tmp_path / "routing_proxy.yaml"),
            "--candidate",
            "fast:model-a",
            "--candidate",
            "balanced:model-b",
            "--all-fixtures",
            "--output",
            str(tmp_path / "blocked-compare-cli.jsonl"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert "confirm-large-run" in payload["error"]
    assert "never sweep discovered models" in payload["error"]


def test_eval_run_cli_requires_confirmation_for_all_fixtures(
    tmp_path,
    monkeypatch,
    capsys,
):
    initialize_product_config(
        preset="lmstudio",
        config_dir=tmp_path,
        force=False,
        interactive=False,
    )
    output = tmp_path / "blocked-results.jsonl"

    def runner(_request):
        raise AssertionError("blocked broad eval run should not call a backend")

    monkeypatch.setattr(
        "hermes.plugins.model_router.eval_runner.run_backend_eval_request",
        runner,
    )

    exit_code = model_router_cli.main(
        [
            "eval",
            "run",
            "--json",
            "--config",
            str(tmp_path / "routing_proxy.yaml"),
            "--backend",
            "fast",
            "--all-fixtures",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert "confirm-large-run" in payload["error"]
    assert "sweep discovered models" in payload["error"]
    assert not output.exists()


def test_eval_list_cli_emits_privacy_safe_json(capsys):
    exit_code = model_router_cli.main(["eval", "list", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    serialized = json.dumps(payload)
    assert payload["ok"] is True
    assert len(payload["fixtures"]) == 8
    assert "Return only JSON" not in serialized
