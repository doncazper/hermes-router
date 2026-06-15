"""Setup-time model catalog and hardware-aware recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Sequence

import yaml


@dataclass(frozen=True)
class HardwareProfile:
    system: str
    machine: str
    total_memory_gb: float | None = None
    disk_free_gb: float | None = None
    apple_silicon: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "machine": self.machine,
            "total_memory_gb": self.total_memory_gb,
            "disk_free_gb": self.disk_free_gb,
            "apple_silicon": self.apple_silicon,
        }


@dataclass(frozen=True)
class CatalogModel:
    route: str
    repo_id: str
    provider: str
    adapter: str
    reason: str
    min_memory_gb: float
    recommended_memory_gb: float
    quality_score: int
    speed_score: int
    profiles: tuple[str, ...] = ("balanced",)
    hardware: tuple[str, ...] = ("cpu",)
    include: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelAdvice:
    route: str
    repo_id: str
    provider: str
    adapter: str
    reason: str
    include: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "repo_id": self.repo_id,
            "provider": self.provider,
            "adapter": self.adapter,
            "reason": self.reason,
            "include": list(self.include),
        }


@dataclass(frozen=True)
class ModelCatalog:
    version: int
    models: tuple[CatalogModel, ...] = field(default_factory=tuple)


def detect_hardware_profile(download_root: str | Path | None = None) -> HardwareProfile:
    system = platform.system()
    machine = platform.machine()
    return HardwareProfile(
        system=system,
        machine=machine,
        total_memory_gb=_total_memory_gb(),
        disk_free_gb=_disk_free_gb(download_root),
        apple_silicon=system == "Darwin" and machine in {"arm64", "aarch64"},
    )


def load_model_catalog() -> ModelCatalog:
    catalog_resource = resources.files("hermes.plugins.model_router").joinpath(
        "data",
        "model_catalog.yaml",
    )
    payload = yaml.safe_load(catalog_resource.read_text(encoding="utf-8")) or {}
    models = tuple(_catalog_model(item) for item in payload.get("models", ()))
    return ModelCatalog(version=int(payload.get("version", 1)), models=models)


def recommend_catalog_models(
    *,
    profile: str = "balanced",
    hardware: HardwareProfile | None = None,
    catalog: ModelCatalog | None = None,
) -> tuple[ModelAdvice, ...]:
    hardware = hardware or detect_hardware_profile()
    catalog = catalog or load_model_catalog()
    routes = tuple(dict.fromkeys(model.route for model in catalog.models))
    advice: list[ModelAdvice] = []
    for route in routes:
        candidates = [model for model in catalog.models if model.route == route]
        selected = _select_candidate(candidates, profile, hardware)
        if selected is None:
            continue
        advice.append(_advice_for_candidate(selected, hardware))
    return tuple(advice)


def _catalog_model(item: dict[str, Any]) -> CatalogModel:
    return CatalogModel(
        route=str(item["route"]),
        repo_id=str(item["repo_id"]),
        provider=str(item.get("provider", "huggingface")),
        adapter=str(item["adapter"]),
        reason=str(item["reason"]),
        min_memory_gb=float(item.get("min_memory_gb", 0)),
        recommended_memory_gb=float(item.get("recommended_memory_gb", 0)),
        quality_score=int(item.get("quality_score", 1)),
        speed_score=int(item.get("speed_score", 1)),
        profiles=tuple(str(value) for value in item.get("profiles", ("balanced",))),
        hardware=tuple(str(value) for value in item.get("hardware", ("cpu",))),
        include=tuple(str(value) for value in item.get("include", ())),
    )


def _select_candidate(
    candidates: Sequence[CatalogModel],
    profile: str,
    hardware: HardwareProfile,
) -> CatalogModel | None:
    if not candidates:
        return None
    profile_candidates = [
        candidate for candidate in candidates if profile in candidate.profiles
    ]
    ranked = sorted(
        profile_candidates or list(candidates),
        key=lambda candidate: _candidate_score(candidate, profile, hardware),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _candidate_score(
    candidate: CatalogModel,
    profile: str,
    hardware: HardwareProfile,
) -> float:
    quality_weight, speed_weight = {
        "lightweight": (1.0, 3.0),
        "quality": (3.0, 1.0),
    }.get(profile, (2.0, 2.0))
    score = candidate.quality_score * quality_weight
    score += candidate.speed_score * speed_weight
    score += _memory_score(candidate, hardware)
    score += _hardware_score(candidate, hardware)
    return score


def _memory_score(candidate: CatalogModel, hardware: HardwareProfile) -> float:
    if hardware.total_memory_gb is None:
        return 0
    if hardware.total_memory_gb >= candidate.recommended_memory_gb:
        return 20
    if hardware.total_memory_gb >= candidate.min_memory_gb:
        return 8
    return -100 + hardware.total_memory_gb


def _hardware_score(candidate: CatalogModel, hardware: HardwareProfile) -> float:
    if hardware.apple_silicon and "apple_silicon" in candidate.hardware:
        return 6
    if "cpu" in candidate.hardware:
        return 2
    return 0


def _advice_for_candidate(
    candidate: CatalogModel,
    hardware: HardwareProfile,
) -> ModelAdvice:
    reason_parts = [candidate.reason, _hardware_reason(candidate, hardware)]
    return ModelAdvice(
        route=candidate.route,
        repo_id=candidate.repo_id,
        provider=candidate.provider,
        adapter=candidate.adapter,
        reason=" ".join(part for part in reason_parts if part),
        include=candidate.include,
    )


def _hardware_reason(candidate: CatalogModel, hardware: HardwareProfile) -> str:
    if hardware.total_memory_gb is None:
        return (
            f"Catalog fit: needs about {candidate.min_memory_gb:g} GB RAM minimum, "
            f"{candidate.recommended_memory_gb:g} GB recommended."
        )
    memory = hardware.total_memory_gb
    if memory >= candidate.recommended_memory_gb:
        return f"Fits detected {memory:.0f} GB RAM."
    if memory >= candidate.min_memory_gb:
        return f"Fits detected {memory:.0f} GB RAM, near the lower end."
    return (
        f"Detected {memory:.0f} GB RAM is below the catalog minimum "
        f"of {candidate.min_memory_gb:g} GB; expect pressure."
    )


def _total_memory_gb() -> float | None:
    sysconf_names = getattr(os, "sysconf_names", {})
    if "SC_PAGE_SIZE" in sysconf_names and "SC_PHYS_PAGES" in sysconf_names:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return round((page_size * pages) / (1024**3), 1)
        except (OSError, TypeError, ValueError):
            pass
    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ("sysctl", "-n", "hw.memsize"),
                text=True,
                capture_output=True,
                check=False,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode == 0:
            try:
                return round(int(completed.stdout.strip()) / (1024**3), 1)
            except ValueError:
                return None
    return None


def _disk_free_gb(download_root: str | Path | None) -> float | None:
    root = Path(download_root).expanduser() if download_root is not None else Path.cwd()
    probe = root if root.exists() else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / (1024**3), 1)
    except OSError:
        return None
