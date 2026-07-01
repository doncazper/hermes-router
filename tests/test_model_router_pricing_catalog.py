import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes.plugins.model_router.pricing_catalog import (
    PRICING_MISSING_PRICE,
    PricingCatalogError,
    apply_pricing_catalog,
    estimate_usage_cost,
    load_packaged_pricing_catalog,
    load_pricing_catalog,
    pricing_diff,
    pricing_status,
)


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_catalog(path: Path, *, input_rate: float = 2.0) -> None:
    path.write_text(
        f"""catalog_version: 9
updated_at: "2026-06-30T00:00:00Z"
notes:
  - test override
entries:
  - provider: test-provider
    model: test-model
    input_per_1m: {input_rate}
    output_per_1m: 4.0
    cached_input_per_1m: 0.5
    currency: USD
    effective_date: "2026-06-30"
    source: test-fixture
    notes: test price
""",
        encoding="utf-8",
    )


def test_packaged_pricing_catalog_is_json_safe_and_local_only():
    catalog = load_packaged_pricing_catalog()
    payload = catalog.to_dict()

    assert catalog.catalog_version >= 1
    assert catalog.source == "packaged"
    assert any(entry["provider"] == "local" for entry in payload["entries"])
    assert json.dumps(payload)


def test_pricing_catalog_rejects_invalid_catalog(tmp_path):
    path = tmp_path / "pricing_catalog.yaml"
    path.write_text(
        """catalog_version: 1
updated_at: "2026-06-30T00:00:00Z"
entries:
  - provider: test
    model: bad
    input_per_1m: -1
    output_per_1m: 1
    cached_input_per_1m: 0
    currency: USD
    effective_date: "2026-06-30"
    source: test
""",
        encoding="utf-8",
    )

    with pytest.raises(PricingCatalogError, match="input_per_1m"):
        load_pricing_catalog(path)


def test_pricing_catalog_missing_override_uses_packaged_catalog(tmp_path):
    catalog = load_pricing_catalog(tmp_path / "missing.yaml")

    assert catalog.source == "packaged"
    assert catalog.lookup(provider="example", model="example-hosted-model")[0] == "matched"


def test_pricing_catalog_override_precedence_and_missing_price(tmp_path):
    path = tmp_path / "pricing_catalog.yaml"
    _write_catalog(path, input_rate=7.0)

    catalog = load_pricing_catalog(path)
    status, entry = catalog.lookup(provider="test-provider", model="test-model")
    missing_status, missing_entry = catalog.lookup(provider="test-provider", model="missing")

    assert status == "matched"
    assert entry is not None
    assert entry.to_dict()["input_per_1m"] == 7
    assert missing_status == PRICING_MISSING_PRICE
    assert missing_entry is None


def test_estimate_usage_cost_uses_cached_input_tokens(tmp_path):
    path = tmp_path / "pricing_catalog.yaml"
    _write_catalog(path)
    catalog = load_pricing_catalog(path)

    estimate = estimate_usage_cost(
        {
            "usage_prompt_tokens": 10,
            "usage_completion_tokens": 5,
            "usage_cached_input_tokens": 3,
        },
        catalog,
        provider="test-provider",
        model_candidates=("test-model",),
    )

    assert estimate["pricing_match_status"] == "matched"
    assert estimate["estimated_input_cost"] == 0.000014
    assert estimate["estimated_output_cost"] == 0.00002
    assert estimate["estimated_cached_input_cost"] == 0.0000015
    assert estimate["estimated_total_cost"] == 0.0000355
    assert estimate["estimated_cost_currency"] == "USD"


def test_pricing_status_diff_and_apply_are_local_only(tmp_path):
    path = tmp_path / "pricing_catalog.yaml"

    status = pricing_status(path)
    diff = pricing_diff(path)
    unconfirmed = apply_pricing_catalog(path, confirmed=False)
    exists_after_preview = path.exists()

    assert status.remote_checks_enabled is False
    assert status.validation_state == "valid"
    assert status.override_exists is False
    assert any("placeholder" in warning.lower() for warning in status.warnings)
    assert diff.action == "create"
    assert diff.has_changes is True
    assert exists_after_preview is False
    assert unconfirmed.ok is False
    assert unconfirmed.executed is False
    assert path.exists() is False
    applied = apply_pricing_catalog(path, confirmed=True)
    assert applied.ok is True
    assert applied.executed is True
    assert path.exists()


def test_pricing_apply_backs_up_existing_override(tmp_path):
    path = tmp_path / "pricing_catalog.yaml"
    _write_catalog(path)

    result = apply_pricing_catalog(path, confirmed=True)

    assert result.ok is True
    assert result.executed is True
    assert result.action == "update"
    assert result.backup_path is not None
    assert Path(result.backup_path).read_text(encoding="utf-8").startswith(
        "catalog_version: 9"
    )


def test_pricing_cli_status_diff_apply_and_invalid_catalog(tmp_path):
    path = tmp_path / "pricing_catalog.yaml"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("catalog_version: nope\n", encoding="utf-8")

    status = _run_cli("pricing", "status", "--override", str(path), "--json")
    diff = _run_cli("pricing", "diff", "--override", str(path), "--json")
    blocked = _run_cli("pricing", "apply", "--override", str(path), "--json")
    exists_after_blocked_apply = path.exists()
    applied = _run_cli(
        "pricing",
        "apply",
        "--override",
        str(path),
        "--yes",
        "--json",
    )
    bad_status = _run_cli("pricing", "status", "--override", str(invalid), "--json")

    assert status.returncode == 0
    status_payload = json.loads(status.stdout)
    assert status_payload["remote_checks_enabled"] is False
    assert status_payload["validation_state"] == "valid"
    assert status_payload["warnings"]
    assert diff.returncode == 0
    assert json.loads(diff.stdout)["action"] == "create"
    assert blocked.returncode == 1
    assert json.loads(blocked.stdout)["ok"] is False
    assert exists_after_blocked_apply is False
    assert applied.returncode == 0
    assert json.loads(applied.stdout)["executed"] is True
    assert bad_status.returncode == 1
    bad_payload = json.loads(bad_status.stdout)
    assert bad_payload["validation_state"] == "invalid"
    assert bad_payload["validation_errors"]


def test_pricing_cli_apply_requires_interactive_confirmation(tmp_path):
    path = tmp_path / "pricing_catalog.yaml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes.plugins.model_router.cli",
            "pricing",
            "apply",
            "--override",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        input="n\n",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Apply packaged pricing metadata locally?" in result.stdout
    assert "OK: false" in result.stdout
    assert path.exists() is False
