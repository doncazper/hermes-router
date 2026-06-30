"""Local versioned pricing catalog and telemetry cost estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import difflib
from importlib import resources
from pathlib import Path
import shutil
from typing import Any

import yaml

from hermes.plugins.model_router.product import DEFAULT_CONFIG_DIR, PRODUCT_DATA_PACKAGE


DEFAULT_PRICING_CATALOG_NAME = "pricing_catalog.yaml"
PRICING_MATCHED = "matched"
PRICING_MISSING_PRICE = "missing_price"
PRICING_AMBIGUOUS_MODEL = "ambiguous_model"
PRICING_MISSING_MODEL = "missing_model"
PRICING_NO_USAGE = "no_usage"


class PricingCatalogError(ValueError):
    """Raised when local pricing catalog metadata is invalid."""


@dataclass(frozen=True)
class PricingCatalogEntry:
    provider: str
    model: str
    input_per_1m: Decimal
    output_per_1m: Decimal
    cached_input_per_1m: Decimal
    currency: str
    effective_date: str
    source: str
    notes: str = ""

    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_per_1m": _decimal_to_json(self.input_per_1m),
            "output_per_1m": _decimal_to_json(self.output_per_1m),
            "cached_input_per_1m": _decimal_to_json(self.cached_input_per_1m),
            "currency": self.currency,
            "effective_date": self.effective_date,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PricingCatalog:
    catalog_version: int
    updated_at: str
    entries: tuple[PricingCatalogEntry, ...]
    source: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "updated_at": self.updated_at,
            "source": self.source,
            "entries": [entry.to_dict() for entry in self.entries],
            "notes": list(self.notes),
        }

    def lookup(
        self,
        *,
        provider: str | None,
        model: str | None,
    ) -> tuple[str, PricingCatalogEntry | None]:
        model_key = (model or "").strip()
        if not model_key:
            return PRICING_MISSING_MODEL, None
        provider_key = (provider or "").strip()
        if provider_key:
            for entry in self.entries:
                if entry.provider == provider_key and entry.model == model_key:
                    return PRICING_MATCHED, entry
            return PRICING_MISSING_PRICE, None
        matches = [entry for entry in self.entries if entry.model == model_key]
        if len(matches) == 1:
            return PRICING_MATCHED, matches[0]
        if len(matches) > 1:
            return PRICING_AMBIGUOUS_MODEL, None
        return PRICING_MISSING_PRICE, None


@dataclass(frozen=True)
class PricingStatus:
    packaged_catalog_version: int | None
    packaged_entry_count: int
    override_path: str
    override_exists: bool
    override_valid: bool
    override_catalog_version: int | None
    active_catalog_version: int | None
    active_catalog_source: str
    active_entry_count: int
    remote_checks_enabled: bool = False
    validation_errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "validation_errors": list(self.validation_errors),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PricingDiff:
    status: PricingStatus
    action: str
    has_changes: bool
    diff_lines: tuple[str, ...]
    truncated: bool = False
    notes: tuple[str, ...] = ()

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
class PricingApplyResult:
    ok: bool
    executed: bool
    action: str
    override_path: str
    backup_path: str | None
    diff: PricingDiff
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "executed": self.executed,
            "action": self.action,
            "override_path": self.override_path,
            "backup_path": self.backup_path,
            "diff": self.diff.to_dict(),
            "notes": list(self.notes),
        }


def default_pricing_override_path(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> Path:
    return Path(config_dir).expanduser() / DEFAULT_PRICING_CATALOG_NAME


def packaged_pricing_catalog_text() -> str:
    resource = resources.files(PRODUCT_DATA_PACKAGE).joinpath(
        DEFAULT_PRICING_CATALOG_NAME,
    )
    return resource.read_text(encoding="utf-8")


def load_packaged_pricing_catalog() -> PricingCatalog:
    return _catalog_from_text(
        packaged_pricing_catalog_text(),
        source="packaged",
    )


def load_pricing_catalog(
    override_path: str | Path | None = None,
) -> PricingCatalog:
    packaged = load_packaged_pricing_catalog()
    if override_path is None:
        return packaged
    path = Path(override_path).expanduser()
    if not path.exists():
        return packaged
    override = _catalog_from_text(path.read_text(encoding="utf-8"), source=str(path))
    return _merge_catalogs(packaged, override)


def pricing_status(
    override_path: str | Path | None = None,
) -> PricingStatus:
    path = Path(override_path).expanduser() if override_path else default_pricing_override_path()
    validation_errors: list[str] = []
    packaged: PricingCatalog | None = None
    override_version: int | None = None
    override_valid = True
    active: PricingCatalog | None = None
    try:
        packaged = load_packaged_pricing_catalog()
    except PricingCatalogError as exc:
        validation_errors.append(f"packaged catalog: {exc}")

    if path.exists():
        try:
            override = _catalog_from_text(path.read_text(encoding="utf-8"), source=str(path))
            override_version = override.catalog_version
        except (OSError, PricingCatalogError) as exc:
            override_valid = False
            validation_errors.append(f"override catalog: {exc}")
    try:
        active = load_pricing_catalog(path)
    except (OSError, PricingCatalogError) as exc:
        validation_errors.append(f"active catalog: {exc}")

    return PricingStatus(
        packaged_catalog_version=packaged.catalog_version if packaged else None,
        packaged_entry_count=len(packaged.entries) if packaged else 0,
        override_path=str(path),
        override_exists=path.exists(),
        override_valid=override_valid,
        override_catalog_version=override_version,
        active_catalog_version=active.catalog_version if active else None,
        active_catalog_source=active.source if active else "unavailable",
        active_entry_count=len(active.entries) if active else 0,
        validation_errors=tuple(validation_errors),
        notes=(
            "Pricing status uses local packaged/override files only; no remote checks were made.",
            "Pricing estimates are reporting metadata and do not affect routing.",
        ),
    )


def pricing_diff(
    override_path: str | Path | None = None,
    *,
    context_lines: int = 3,
    max_lines: int = 240,
) -> PricingDiff:
    path = Path(override_path).expanduser() if override_path else default_pricing_override_path()
    status = pricing_status(path)
    packaged_lines = packaged_pricing_catalog_text().splitlines(keepends=True)
    if not path.exists():
        return PricingDiff(
            status=status,
            action="create",
            has_changes=True,
            diff_lines=(
                f"Local pricing catalog override is missing: {path}",
                "Apply will create it from packaged pricing metadata.",
            ),
            notes=("No local override exists, so no backup is needed.",),
        )
    local_text = path.read_text(encoding="utf-8")
    if local_text == packaged_pricing_catalog_text():
        return PricingDiff(
            status=status,
            action="noop",
            has_changes=False,
            diff_lines=(),
            notes=("Local pricing override already matches packaged metadata.",),
        )
    local_lines = local_text.splitlines(keepends=True)
    lines = tuple(
        line.rstrip("\n")
        for line in difflib.unified_diff(
            local_lines,
            packaged_lines,
            fromfile=str(path),
            tofile=f"packaged:{DEFAULT_PRICING_CATALOG_NAME}",
            n=context_lines,
        )
    )
    truncated = len(lines) > max_lines
    shown = (*lines[:max_lines], f"... diff truncated after {max_lines} lines") if truncated else lines
    return PricingDiff(
        status=status,
        action="update",
        has_changes=True,
        diff_lines=tuple(shown),
        truncated=truncated,
        notes=("Apply will back up the local pricing override before writing.",),
    )


def apply_pricing_catalog(
    override_path: str | Path | None = None,
    *,
    confirmed: bool,
) -> PricingApplyResult:
    path = Path(override_path).expanduser() if override_path else default_pricing_override_path()
    diff = pricing_diff(path)
    if not diff.has_changes:
        return PricingApplyResult(
            ok=True,
            executed=False,
            action="noop",
            override_path=str(path),
            backup_path=None,
            diff=diff,
            notes=("No update applied; local pricing override already matches packaged metadata.",),
        )
    if not confirmed:
        return PricingApplyResult(
            ok=False,
            executed=False,
            action=diff.action,
            override_path=str(path),
            backup_path=None,
            diff=diff,
            notes=("Pricing apply requires explicit confirmation.",),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if path.exists():
        backup_path = _backup_path(path)
        shutil.copy2(path, backup_path)
    path.write_text(packaged_pricing_catalog_text(), encoding="utf-8")
    return PricingApplyResult(
        ok=True,
        executed=True,
        action=diff.action,
        override_path=str(path),
        backup_path=str(backup_path) if backup_path else None,
        diff=diff,
        notes=(
            "Packaged pricing metadata applied locally.",
            "No remote pricing checks were made.",
        ),
    )


def estimate_usage_cost(
    usage: dict[str, Any],
    catalog: PricingCatalog,
    *,
    provider: str | None = None,
    model_candidates: tuple[str, ...] = (),
) -> dict[str, Any]:
    prompt_tokens = _non_negative_int(usage.get("usage_prompt_tokens"))
    completion_tokens = _non_negative_int(usage.get("usage_completion_tokens"))
    cached_tokens = _non_negative_int(usage.get("usage_cached_input_tokens"))
    if prompt_tokens == completion_tokens == cached_tokens == 0:
        return _estimate_unavailable(catalog, PRICING_NO_USAGE)

    cleaned_models = tuple(
        dict.fromkeys(model.strip() for model in model_candidates if model.strip())
    )
    if not cleaned_models:
        return _estimate_unavailable(catalog, PRICING_MISSING_MODEL)

    last_status = PRICING_MISSING_PRICE
    for model in cleaned_models:
        status, entry = catalog.lookup(provider=provider, model=model)
        if status == PRICING_MATCHED and entry is not None:
            return _estimate_from_entry(
                catalog=catalog,
                entry=entry,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
        if status == PRICING_AMBIGUOUS_MODEL:
            return _estimate_unavailable(catalog, status, model=model)
        last_status = status
    return _estimate_unavailable(catalog, last_status, model=cleaned_models[0])


def _estimate_from_entry(
    *,
    catalog: PricingCatalog,
    entry: PricingCatalogEntry,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
) -> dict[str, Any]:
    uncached_input_tokens = max(prompt_tokens - cached_tokens, 0)
    input_cost = _tokens_to_cost(uncached_input_tokens, entry.input_per_1m)
    output_cost = _tokens_to_cost(completion_tokens, entry.output_per_1m)
    cached_cost = _tokens_to_cost(cached_tokens, entry.cached_input_per_1m)
    total = input_cost + output_cost + cached_cost
    return {
        "pricing_match_status": PRICING_MATCHED,
        "estimated_input_cost": _money_to_json(input_cost),
        "estimated_output_cost": _money_to_json(output_cost),
        "estimated_cached_input_cost": _money_to_json(cached_cost),
        "estimated_total_cost": _money_to_json(total),
        "estimated_cost_currency": entry.currency,
        "pricing_catalog_version": catalog.catalog_version,
        "pricing_catalog_source": catalog.source,
        "pricing_source": entry.source,
        "pricing_effective_date": entry.effective_date,
        "pricing_provider": entry.provider,
        "pricing_model": entry.model,
    }


def _estimate_unavailable(
    catalog: PricingCatalog,
    status: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pricing_match_status": status,
        "estimated_input_cost": None,
        "estimated_output_cost": None,
        "estimated_cached_input_cost": None,
        "estimated_total_cost": None,
        "estimated_cost_currency": None,
        "pricing_catalog_version": catalog.catalog_version,
        "pricing_catalog_source": catalog.source,
    }
    if model:
        payload["pricing_model"] = model
    return payload


def _merge_catalogs(packaged: PricingCatalog, override: PricingCatalog) -> PricingCatalog:
    entries: dict[tuple[str, str], PricingCatalogEntry] = {
        entry.key(): entry for entry in packaged.entries
    }
    entries.update({entry.key(): entry for entry in override.entries})
    return PricingCatalog(
        catalog_version=override.catalog_version,
        updated_at=override.updated_at,
        entries=tuple(entries[key] for key in sorted(entries)),
        source=f"packaged+override:{override.source}",
        notes=(*packaged.notes, *override.notes),
    )


def _catalog_from_text(text: str, *, source: str) -> PricingCatalog:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise PricingCatalogError(f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PricingCatalogError("catalog must be a mapping")
    version = data.get("catalog_version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise PricingCatalogError("catalog_version must be a positive integer")
    updated_at = _required_string(data, "updated_at")
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PricingCatalogError("entries must be a non-empty list")
    entries = tuple(_entry_from_mapping(item, index=index) for index, item in enumerate(raw_entries))
    keys = [entry.key() for entry in entries]
    if len(keys) != len(set(keys)):
        raise PricingCatalogError("entries must have unique provider/model pairs")
    return PricingCatalog(
        catalog_version=version,
        updated_at=updated_at,
        entries=entries,
        source=source,
        notes=tuple(str(note) for note in data.get("notes", ()) if note is not None)
        if isinstance(data.get("notes"), list)
        else (),
    )


def _entry_from_mapping(value: Any, *, index: int) -> PricingCatalogEntry:
    if not isinstance(value, dict):
        raise PricingCatalogError(f"entries[{index}] must be a mapping")
    return PricingCatalogEntry(
        provider=_required_string(value, "provider", index=index),
        model=_required_string(value, "model", index=index),
        input_per_1m=_required_decimal(value, "input_per_1m", index=index),
        output_per_1m=_required_decimal(value, "output_per_1m", index=index),
        cached_input_per_1m=_required_decimal(
            value,
            "cached_input_per_1m",
            index=index,
        ),
        currency=_required_string(value, "currency", index=index),
        effective_date=_required_string(value, "effective_date", index=index),
        source=_required_string(value, "source", index=index),
        notes=str(value.get("notes") or ""),
    )


def _required_string(
    data: dict[str, Any],
    key: str,
    *,
    index: int | None = None,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        prefix = f"entries[{index}]." if index is not None else ""
        raise PricingCatalogError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _required_decimal(
    data: dict[str, Any],
    key: str,
    *,
    index: int,
) -> Decimal:
    value = data.get(key)
    if isinstance(value, bool) or value is None:
        raise PricingCatalogError(f"entries[{index}].{key} must be a non-negative number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PricingCatalogError(
            f"entries[{index}].{key} must be a non-negative number"
        ) from exc
    if parsed < 0:
        raise PricingCatalogError(f"entries[{index}].{key} must be non-negative")
    return parsed


def _tokens_to_cost(tokens: int, rate_per_1m: Decimal) -> Decimal:
    return (Decimal(tokens) * rate_per_1m) / Decimal(1_000_000)


def _money_to_json(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.00000001")))


def _decimal_to_json(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _backup_path(path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f"{path.name}.bak-{timestamp}")
