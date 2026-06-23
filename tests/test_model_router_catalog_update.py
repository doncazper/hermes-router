import json
import subprocess
import sys
from pathlib import Path

from hermes.plugins.model_router.catalog_update import (
    apply_catalog_update,
    catalog_diff,
    catalog_status,
)
from hermes.plugins.model_router.config import default_config_text


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes.plugins.model_router.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_catalog_status_is_packaged_only_and_no_network(tmp_path):
    path = tmp_path / "model_router.yaml"

    status = catalog_status(path)

    assert status.local_exists is False
    assert status.remote_checks_enabled is False
    assert status.packaged_model_catalog_version >= 1
    assert any("no remote checks" in note for note in status.notes)


def test_catalog_diff_previews_missing_config_without_writing(tmp_path):
    path = tmp_path / "model_router.yaml"

    diff = catalog_diff(path)

    assert diff.action == "create"
    assert diff.has_changes is True
    assert path.exists() is False
    assert any("Apply will create" in line for line in diff.diff_lines)


def test_catalog_apply_requires_confirmation(tmp_path):
    path = tmp_path / "model_router.yaml"

    result = apply_catalog_update(path, confirmed=False)

    assert result.ok is False
    assert result.executed is False
    assert result.action == "create"
    assert path.exists() is False


def test_catalog_apply_backs_up_custom_config_and_logs(tmp_path):
    path = tmp_path / "model_router.yaml"
    path.write_text("custom: true\n", encoding="utf-8")
    log_path = tmp_path / "catalog.jsonl"

    result = apply_catalog_update(path, confirmed=True, migration_log=log_path)

    assert result.ok is True
    assert result.executed is True
    assert result.action == "update"
    assert result.backup_path is not None
    assert Path(result.backup_path).read_text(encoding="utf-8") == "custom: true\n"
    assert path.read_text(encoding="utf-8") == default_config_text()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event_type"] == "catalog_apply"
    assert rows[-1]["backup_path"] == result.backup_path


def test_catalog_apply_noops_when_config_matches_packaged(tmp_path):
    path = tmp_path / "model_router.yaml"
    path.write_text(default_config_text(), encoding="utf-8")

    result = apply_catalog_update(path, confirmed=True)

    assert result.ok is True
    assert result.executed is False
    assert result.action == "noop"
    assert result.backup_path is None


def test_catalog_cli_status_diff_and_apply_json(tmp_path):
    path = tmp_path / "model_router.yaml"

    status = _run_cli("catalog", "status", "--config", str(path), "--json")
    diff = _run_cli("catalog", "diff", "--config", str(path), "--json")
    apply = _run_cli("catalog", "apply", "--config", str(path), "--yes", "--json")

    assert status.returncode == 0
    assert json.loads(status.stdout)["remote_checks_enabled"] is False
    assert diff.returncode == 0
    assert json.loads(diff.stdout)["action"] == "create"
    assert apply.returncode == 0
    payload = json.loads(apply.stdout)
    assert payload["ok"] is True
    assert payload["executed"] is True
    assert path.exists()
