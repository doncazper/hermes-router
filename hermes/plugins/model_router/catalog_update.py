"""Packaged catalog status, diff, and explicit apply helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import difflib
import hashlib
from importlib import resources
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from hermes.plugins.model_router.config import (
    DEFAULT_CONFIG_SOURCE,
    default_config_text,
)
from hermes.plugins.model_router.product import DEFAULT_CONFIG_DIR, PRODUCT_DATA_PACKAGE


DEFAULT_CATALOG_LOG_NAME = "catalog-migrations.jsonl"
CATALOG_UPDATE_VERSION = 1


@dataclass(frozen=True)
class CatalogStatus:
    packaged_model_catalog_version: int
    packaged_router_config_source: str
    packaged_router_config_hash: str
    local_config: str
    local_exists: bool
    local_matches_packaged: bool
    local_router_config_hash: str | None
    last_applied_model_catalog_version: int | None
    migration_log: str
    update_source: str = "packaged"
    remote_checks_enabled: bool = False
    overrides: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "overrides": list(self.overrides),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CatalogDiff:
    status: CatalogStatus
    action: str
    has_changes: bool
    diff_lines: tuple[str, ...]
    truncated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.to_dict(),
            "action": self.action,
            "has_changes": self.has_changes,
            "diff_lines": list(self.diff_lines),
            "truncated": self.truncated,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CatalogApplyResult:
    ok: bool
    executed: bool
    action: str
    config_path: str
    backup_path: str | None
    migration_log: str
    diff: CatalogDiff
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "executed": self.executed,
            "action": self.action,
            "config_path": self.config_path,
            "backup_path": self.backup_path,
            "migration_log": self.migration_log,
            "diff": self.diff.to_dict(),
            "notes": list(self.notes),
        }


def default_local_config_path(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> Path:
    return Path(config_dir).expanduser() / "model_router.yaml"


def default_migration_log_path(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().parent / DEFAULT_CATALOG_LOG_NAME


def catalog_status(
    config_path: str | Path | None = None,
    *,
    migration_log: str | Path | None = None,
) -> CatalogStatus:
    path = Path(config_path).expanduser() if config_path else default_local_config_path()
    log_path = (
        Path(migration_log).expanduser()
        if migration_log
        else default_migration_log_path(path)
    )
    packaged_text = default_config_text()
    local_text = _read_text(path)
    local_data = _yaml_mapping(local_text)
    packaged_data = _yaml_mapping(packaged_text)
    latest = _latest_migration(log_path)
    local_hash = _sha256_text(local_text) if local_text is not None else None
    packaged_hash = _sha256_text(packaged_text)
    matches_packaged = local_hash == packaged_hash if local_hash is not None else False
    return CatalogStatus(
        packaged_model_catalog_version=packaged_model_catalog_version(),
        packaged_router_config_source=DEFAULT_CONFIG_SOURCE,
        packaged_router_config_hash=packaged_hash,
        local_config=str(path),
        local_exists=local_text is not None,
        local_matches_packaged=matches_packaged,
        local_router_config_hash=local_hash,
        last_applied_model_catalog_version=_latest_catalog_version(latest),
        migration_log=str(log_path),
        overrides=_override_summary(local_data, packaged_data),
        notes=(
            "Catalog status uses packaged metadata only; no remote checks were made.",
            "Local config differences are reported as overrides, not errors.",
        ),
    )


def catalog_diff(
    config_path: str | Path | None = None,
    *,
    migration_log: str | Path | None = None,
    context_lines: int = 3,
    max_lines: int = 240,
) -> CatalogDiff:
    path = Path(config_path).expanduser() if config_path else default_local_config_path()
    status = catalog_status(path, migration_log=migration_log)
    packaged_lines = default_config_text().splitlines(keepends=True)
    local_text = _read_text(path)
    if local_text is None:
        return CatalogDiff(
            status=status,
            action="create",
            has_changes=True,
            diff_lines=(
                f"Local config is missing: {path}",
                "Apply will create it from the packaged router catalog.",
            ),
            notes=("No local file exists, so no backup is needed.",),
        )

    if status.local_matches_packaged:
        return CatalogDiff(
            status=status,
            action="noop",
            has_changes=False,
            diff_lines=(),
            notes=("Local config already matches the packaged router catalog.",),
        )

    local_lines = local_text.splitlines(keepends=True)
    lines = tuple(
        line.rstrip("\n")
        for line in difflib.unified_diff(
            local_lines,
            packaged_lines,
            fromfile=str(path),
            tofile=DEFAULT_CONFIG_SOURCE,
            n=context_lines,
        )
    )
    truncated = len(lines) > max_lines
    if truncated:
        shown = (*lines[:max_lines], f"... diff truncated after {max_lines} lines")
    else:
        shown = lines
    return CatalogDiff(
        status=status,
        action="update",
        has_changes=True,
        diff_lines=tuple(shown),
        truncated=truncated,
        notes=(
            "Apply will back up the local config before writing packaged defaults.",
        ),
    )


def apply_catalog_update(
    config_path: str | Path | None = None,
    *,
    confirmed: bool,
    migration_log: str | Path | None = None,
) -> CatalogApplyResult:
    path = Path(config_path).expanduser() if config_path else default_local_config_path()
    log_path = Path(migration_log).expanduser() if migration_log else default_migration_log_path(path)
    diff = catalog_diff(path, migration_log=log_path)
    if not diff.has_changes:
        return CatalogApplyResult(
            ok=True,
            executed=False,
            action="noop",
            config_path=str(path),
            backup_path=None,
            migration_log=str(log_path),
            diff=diff,
            notes=("No update applied; local config already matches packaged defaults.",),
        )
    if not confirmed:
        return CatalogApplyResult(
            ok=False,
            executed=False,
            action=diff.action,
            config_path=str(path),
            backup_path=None,
            migration_log=str(log_path),
            diff=diff,
            notes=("Catalog apply requires explicit confirmation.",),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if path.exists():
        backup_path = _backup_path(path)
        shutil.copy2(path, backup_path)
    path.write_text(default_config_text(), encoding="utf-8")
    _append_migration_log(
        log_path,
        {
            "version": CATALOG_UPDATE_VERSION,
            "event_type": "catalog_apply",
            "timestamp": _now_iso(),
            "action": diff.action,
            "config_path": str(path),
            "backup_path": str(backup_path) if backup_path else None,
            "packaged_model_catalog_version": packaged_model_catalog_version(),
            "packaged_router_config_hash": _sha256_text(default_config_text()),
        },
    )
    return CatalogApplyResult(
        ok=True,
        executed=True,
        action=diff.action,
        config_path=str(path),
        backup_path=str(backup_path) if backup_path else None,
        migration_log=str(log_path),
        diff=diff,
        notes=(
            "Packaged router catalog applied.",
            (
                "Existing config was backed up first."
                if backup_path
                else "Created new local config."
            ),
        ),
    )


def packaged_model_catalog_version() -> int:
    data = _packaged_model_catalog_data()
    value = data.get("version", 1)
    return int(value) if isinstance(value, int) else 1


def _packaged_model_catalog_data() -> dict[str, Any]:
    resource = resources.files(PRODUCT_DATA_PACKAGE).joinpath("model_catalog.yaml")
    data = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _override_summary(
    local: dict[str, Any] | None,
    packaged: dict[str, Any] | None,
) -> tuple[str, ...]:
    if local is None:
        return ()
    if packaged is None:
        return ("packaged config could not be parsed",)
    overrides: list[str] = []
    _mapping_override("routing_targets", local, packaged, overrides)
    _mapping_override("provider_policy", local, packaged, overrides)
    _mapping_override("safety", local, packaged, overrides)
    _mapping_override("scoring", local, packaged, overrides)
    local_engines = local.get("engines") if isinstance(local.get("engines"), dict) else {}
    packaged_engines = packaged.get("engines") if isinstance(packaged.get("engines"), dict) else {}
    if isinstance(local_engines, dict) and isinstance(packaged_engines, dict):
        added = sorted(set(local_engines) - set(packaged_engines))
        removed = sorted(set(packaged_engines) - set(local_engines))
        changed = sorted(
            name
            for name in set(local_engines).intersection(packaged_engines)
            if local_engines.get(name) != packaged_engines.get(name)
        )
        if added:
            overrides.append("custom engines: " + ", ".join(added[:8]))
        if removed:
            overrides.append("removed packaged engines: " + ", ".join(removed[:8]))
        if changed:
            overrides.append(f"engine overrides: {len(changed)}")
    return tuple(overrides)


def _mapping_override(
    key: str,
    local: dict[str, Any],
    packaged: dict[str, Any],
    overrides: list[str],
) -> None:
    if local.get(key) != packaged.get(key):
        overrides.append(f"{key} differs from packaged defaults")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except OSError:
        return None


def _yaml_mapping(text: str | None) -> dict[str, Any] | None:
    if text is None:
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _latest_migration(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event_type") == "catalog_apply":
                latest = row
    except OSError:
        return None
    return latest


def _latest_catalog_version(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    value = row.get("packaged_model_catalog_version")
    return value if isinstance(value, int) else None


def _append_migration_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _backup_path(path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f"{path.name}.bak-{timestamp}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
