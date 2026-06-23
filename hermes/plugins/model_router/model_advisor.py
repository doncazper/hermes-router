"""Setup-time model catalog and hardware-aware recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class HardwareProfile:
    system: str
    machine: str
    total_memory_gb: float | None = None
    available_memory_gb: float | None = None
    disk_free_gb: float | None = None
    apple_silicon: bool = False
    cpu_count: int | None = None
    accelerator_backends: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "machine": self.machine,
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "disk_free_gb": self.disk_free_gb,
            "apple_silicon": self.apple_silicon,
            "cpu_count": self.cpu_count,
            "accelerator_backends": list(self.accelerator_backends),
        }


@dataclass(frozen=True)
class ModelScoreBreakdown:
    fit_score: int
    runtime_match_score: int
    expected_speed_score: int
    quality_role_score: int
    setup_friction_score: int
    benchmark_score: int
    overall_score: int
    label: str
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_score": self.fit_score,
            "runtime_match_score": self.runtime_match_score,
            "expected_speed_score": self.expected_speed_score,
            "quality_role_score": self.quality_role_score,
            "setup_friction_score": self.setup_friction_score,
            "benchmark_score": self.benchmark_score,
            "overall_score": self.overall_score,
            "label": self.label,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
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
    runtime_kind: str = "llama.cpp"
    model_size_b: float | None = None
    quantization: str | None = None
    context_length: int | None = None


@dataclass(frozen=True)
class ModelAdvice:
    route: str
    repo_id: str
    provider: str
    adapter: str
    reason: str
    include: tuple[str, ...] = ()
    score: ModelScoreBreakdown | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "route": self.route,
            "repo_id": self.repo_id,
            "provider": self.provider,
            "adapter": self.adapter,
            "reason": self.reason,
            "include": list(self.include),
        }
        if self.score is not None:
            payload["score"] = self.score.to_dict()
        return payload


@dataclass(frozen=True)
class ModelCatalog:
    version: int
    models: tuple[CatalogModel, ...] = field(default_factory=tuple)


def detect_hardware_profile(download_root: str | Path | None = None) -> HardwareProfile:
    system = platform.system()
    machine = platform.machine()
    accelerator_backends = _detect_accelerator_backends(system, machine)
    return HardwareProfile(
        system=system,
        machine=machine,
        total_memory_gb=_total_memory_gb(),
        available_memory_gb=None,
        disk_free_gb=_disk_free_gb(download_root),
        apple_silicon=system == "Darwin" and machine in {"arm64", "aarch64"},
        cpu_count=os.cpu_count(),
        accelerator_backends=accelerator_backends,
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
    limit_per_route: int = 1,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> tuple[ModelAdvice, ...]:
    hardware = hardware or detect_hardware_profile()
    catalog = catalog or load_model_catalog()
    routes = tuple(dict.fromkeys(model.route for model in catalog.models))
    advice: list[ModelAdvice] = []
    for route in routes:
        candidates = [model for model in catalog.models if model.route == route]
        selected = _rank_candidates(
            candidates,
            profile,
            hardware,
            benchmark_results,
        )[:limit_per_route]
        advice.extend(
            _advice_for_candidate(candidate, hardware, benchmark_results)
            for candidate in selected
        )
    return tuple(advice)


def score_local_model_option(
    *,
    route: str,
    repo_id: str,
    source: str,
    path: str = "",
    roles: Sequence[str] = (),
    hardware: HardwareProfile | None = None,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> ModelScoreBreakdown:
    hardware = hardware or detect_hardware_profile()
    text = f"{repo_id} {path} {source}"
    size_b = _model_size_billions(text)
    quantization = _infer_quantization(text)
    runtime_kind = _infer_runtime_kind(repo_id, source, path, ())
    min_memory, recommended_memory = _estimated_memory_requirements(
        size_b,
        quantization,
        runtime_kind,
    )
    quality_score = _local_quality_score(route, roles, size_b, repo_id)
    speed_score = _local_speed_score(size_b, quantization)
    candidate = CatalogModel(
        route=route,
        repo_id=repo_id,
        provider=source,
        adapter=_adapter_for_route(route),
        reason="Local model discovered on this machine.",
        min_memory_gb=min_memory,
        recommended_memory_gb=recommended_memory,
        quality_score=quality_score,
        speed_score=speed_score,
        hardware=("apple_silicon", "cpu") if runtime_kind == "mlx-lm" else ("cpu",),
        runtime_kind=runtime_kind,
        model_size_b=size_b,
        quantization=quantization,
    )
    return score_catalog_model(
        candidate,
        hardware=hardware,
        benchmark_results=benchmark_results,
    )


def score_catalog_model(
    candidate: CatalogModel,
    *,
    hardware: HardwareProfile,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> ModelScoreBreakdown:
    fit_score, fit_reasons, fit_warnings = _fit_score(candidate, hardware)
    runtime_score, runtime_reasons, runtime_warnings = _runtime_match_score(
        candidate,
        hardware,
    )
    speed_score, speed_reasons, speed_warnings = _expected_speed_score(
        candidate,
        hardware,
    )
    quality_score = _quality_role_score(candidate)
    friction_score, friction_reasons, friction_warnings = _setup_friction_score(
        candidate,
        hardware,
    )
    benchmark_score, benchmark_reasons, benchmark_warnings = _benchmark_score(
        candidate,
        benchmark_results,
    )
    overall = round(
        fit_score * 0.22
        + runtime_score * 0.2
        + speed_score * 0.2
        + quality_score * 0.16
        + friction_score * 0.1
        + benchmark_score * 0.12
    )
    warnings = (
        *fit_warnings,
        *runtime_warnings,
        *speed_warnings,
        *friction_warnings,
        *benchmark_warnings,
    )
    label = _score_label(
        fit_score=fit_score,
        runtime_score=runtime_score,
        speed_score=speed_score,
        benchmark_score=benchmark_score,
        warnings=warnings,
    )
    return ModelScoreBreakdown(
        fit_score=fit_score,
        runtime_match_score=runtime_score,
        expected_speed_score=speed_score,
        quality_role_score=quality_score,
        setup_friction_score=friction_score,
        benchmark_score=benchmark_score,
        overall_score=max(0, min(100, overall)),
        label=label,
        reasons=(
            *fit_reasons,
            *runtime_reasons,
            *speed_reasons,
            f"Role quality score {quality_score}/100.",
            *friction_reasons,
            *benchmark_reasons,
        ),
        warnings=warnings,
    )


def _catalog_model(item: dict[str, Any]) -> CatalogModel:
    repo_id = str(item["repo_id"])
    include = tuple(str(value) for value in item.get("include", ()))
    runtime_kind = str(
        item.get("runtime_kind") or _infer_runtime_kind(repo_id, "", "", include)
    )
    quantization = item.get("quantization")
    quantization_text = str(quantization) if quantization is not None else None
    if quantization_text is None:
        quantization_text = _infer_quantization(" ".join((repo_id, *include)))
    return CatalogModel(
        route=str(item["route"]),
        repo_id=repo_id,
        provider=str(item.get("provider", "huggingface")),
        adapter=str(item["adapter"]),
        reason=str(item["reason"]),
        min_memory_gb=float(item.get("min_memory_gb", 0)),
        recommended_memory_gb=float(item.get("recommended_memory_gb", 0)),
        quality_score=int(item.get("quality_score", 1)),
        speed_score=int(item.get("speed_score", 1)),
        profiles=tuple(str(value) for value in item.get("profiles", ("balanced",))),
        hardware=tuple(str(value) for value in item.get("hardware", ("cpu",))),
        include=include,
        runtime_kind=runtime_kind,
        model_size_b=_optional_float(item.get("model_size_b"))
        or _model_size_billions(repo_id),
        quantization=quantization_text,
        context_length=_optional_int(item.get("context_length")),
    )


def _rank_candidates(
    candidates: Sequence[CatalogModel],
    profile: str,
    hardware: HardwareProfile,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[CatalogModel, ...]:
    if not candidates:
        return ()
    profile_candidates = [
        candidate for candidate in candidates if profile in candidate.profiles
    ]
    return tuple(
        sorted(
            profile_candidates or list(candidates),
            key=lambda candidate: _candidate_score(
                candidate,
                profile,
                hardware,
                benchmark_results,
            ),
            reverse=True,
        )
    )


def _candidate_score(
    candidate: CatalogModel,
    profile: str,
    hardware: HardwareProfile,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> float:
    breakdown = score_catalog_model(
        candidate,
        hardware=hardware,
        benchmark_results=benchmark_results,
    )
    profile_bonus = {
        "lightweight": candidate.speed_score,
        "quality": candidate.quality_score,
    }.get(profile, (candidate.quality_score + candidate.speed_score) / 2)
    benchmark_bonus = max(0, breakdown.benchmark_score - 50) * 0.1
    return breakdown.overall_score + profile_bonus + benchmark_bonus


def _advice_for_candidate(
    candidate: CatalogModel,
    hardware: HardwareProfile,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> ModelAdvice:
    score = score_catalog_model(
        candidate,
        hardware=hardware,
        benchmark_results=benchmark_results,
    )
    reason_parts = [
        candidate.reason,
        _hardware_reason(candidate, hardware),
        f"Recommendation label: {score.label}.",
    ]
    return ModelAdvice(
        route=candidate.route,
        repo_id=candidate.repo_id,
        provider=candidate.provider,
        adapter=candidate.adapter,
        reason=" ".join(part for part in reason_parts if part),
        include=candidate.include,
        score=score,
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


def _fit_score(
    candidate: CatalogModel,
    hardware: HardwareProfile,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    memory = hardware.available_memory_gb or hardware.total_memory_gb
    if memory is None:
        return (
            55,
            (
                "Memory unavailable; RAM can only be treated as an unknown fit gate.",
            ),
            (),
        )
    if memory < candidate.min_memory_gb:
        return (
            max(0, round(35 * (memory / max(candidate.min_memory_gb, 1)))),
            (
                f"Detected {memory:.0f} GB memory is below minimum "
                f"{candidate.min_memory_gb:g} GB.",
            ),
            ("too_large_for_memory_gate",),
        )
    if memory < candidate.recommended_memory_gb:
        return (
            65,
            (
                f"Detected {memory:.0f} GB memory fits minimum but is below "
                f"recommended {candidate.recommended_memory_gb:g} GB.",
            ),
            ("memory_pressure_expected",),
        )
    return (
        95,
        (
            f"Detected {memory:.0f} GB memory clears recommended "
            f"{candidate.recommended_memory_gb:g} GB.",
        ),
        (),
    )


def _runtime_match_score(
    candidate: CatalogModel,
    hardware: HardwareProfile,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    accelerators = set(hardware.accelerator_backends)
    runtime = candidate.runtime_kind
    if runtime == "mlx-lm":
        if hardware.apple_silicon and "metal" in accelerators:
            return 96, ("MLX-LM matches Apple Silicon with Metal.",), ()
        if hardware.apple_silicon:
            return 84, ("MLX-LM matches Apple Silicon.",), ()
        return 18, ("MLX-LM is Apple Silicon-oriented.",), ("needs_apple_silicon",)
    if runtime in {"llama.cpp", "gguf"}:
        if accelerators & {"metal", "cuda", "rocm"}:
            return 86, (f"GGUF runtime can use {', '.join(sorted(accelerators))}.",), ()
        return 66, ("GGUF/llama.cpp can run CPU-only.",), ()
    if runtime in {"ollama", "lmstudio"}:
        return 78, (f"{runtime} is a managed local OpenAI-compatible runtime.",), ()
    return 58, ("Generic runtime compatibility is unknown but possible.",), ()


def _expected_speed_score(
    candidate: CatalogModel,
    hardware: HardwareProfile,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    score = min(100, max(5, candidate.speed_score * 16))
    warnings: list[str] = []
    reasons: list[str] = [f"Catalog speed hint {candidate.speed_score}/5."]
    size_b = candidate.model_size_b
    cpu_count = hardware.cpu_count or 0
    if candidate.quantization:
        quant = candidate.quantization.lower()
        if "4" in quant or "q4" in quant:
            score += 12
            reasons.append(f"Quantization {candidate.quantization} favors local speed.")
        elif "8" in quant or "q8" in quant:
            score -= 4
            reasons.append(f"Quantization {candidate.quantization} may be slower.")
    if size_b is not None:
        if size_b <= 1.5:
            score += 10
            reasons.append(f"{size_b:g}B model is small enough for fast routes.")
        elif size_b >= 7 and cpu_count and cpu_count <= 4:
            score -= 24
            warnings.append("large_model_on_low_core_cpu")
        elif size_b >= 13:
            score -= 18
            warnings.append("large_model_expected_slow")
    if cpu_count:
        if cpu_count >= 10:
            score += 8
            reasons.append(f"{cpu_count} CPU cores improve CPU fallback.")
        elif cpu_count <= 4:
            score -= 12
            warnings.append("low_cpu_core_count")
    if hardware.accelerator_backends:
        score += 8
        reasons.append(
            "Accelerator backend detected: "
            + ", ".join(hardware.accelerator_backends)
            + "."
        )
    return max(0, min(100, round(score))), tuple(reasons), tuple(warnings)


def _quality_role_score(candidate: CatalogModel) -> int:
    return max(0, min(100, candidate.quality_score * 18))


def _setup_friction_score(
    candidate: CatalogModel,
    hardware: HardwareProfile,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    score = 70
    reasons = [f"Runtime kind: {candidate.runtime_kind}."]
    warnings: list[str] = []
    if candidate.provider == "huggingface":
        score -= 8
        reasons.append("Requires an explicit Hugging Face download.")
    if candidate.runtime_kind == "mlx-lm" and not hardware.apple_silicon:
        score -= 45
        warnings.append("mlx_lm_runtime_mismatch")
    if candidate.runtime_kind in {"llama.cpp", "gguf"}:
        reasons.append("Requires a llama.cpp-compatible server or wrapper.")
    return max(0, min(100, score)), tuple(reasons), tuple(warnings)


def _benchmark_score(
    candidate: CatalogModel,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    record = _benchmark_record(candidate.repo_id, benchmark_results)
    if record is None:
        return 50, ("No benchmark result yet; score is neutral.",), ()
    status = str(record.get("status", "unknown"))
    if status != "completed":
        return (
            10,
            (f"Benchmark status for this model is {status}.",),
            ("benchmark_failed_or_missing",),
        )
    tokens_per_second = _optional_float(record.get("tokens_per_second"))
    latency_ms = _optional_float(record.get("total_latency_ms"))
    score = 68
    reasons = ["Successful local benchmark recorded."]
    if tokens_per_second is not None:
        if tokens_per_second >= 25:
            score = 96
        elif tokens_per_second >= 10:
            score = 82
        elif tokens_per_second >= 4:
            score = 62
        else:
            score = 38
        reasons.append(f"Measured {tokens_per_second:g} tokens/sec.")
    if latency_ms is not None and latency_ms > 10_000:
        score -= 12
    warnings = ("benchmark_slow",) if score < 50 else ()
    return max(0, min(100, round(score))), tuple(reasons), warnings


def _score_label(
    *,
    fit_score: int,
    runtime_score: int,
    speed_score: int,
    benchmark_score: int,
    warnings: tuple[str, ...],
) -> str:
    if fit_score < 45 or "too_large_for_memory_gate" in warnings:
        return "too_large"
    if runtime_score < 45:
        return "needs_runtime"
    if benchmark_score >= 82:
        return "benchmark_recommended"
    if speed_score < 48 or "large_model_expected_slow" in warnings:
        return "fits_but_likely_slow"
    return "recommended"


def _benchmark_record(
    model: str,
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    rows = _benchmark_rows(benchmark_results)
    matches = [
        row
        for row in rows
        if str(row.get("model", "")) == model or str(row.get("repo_id", "")) == model
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda row: (
            row.get("status") == "completed",
            float(row.get("tokens_per_second") or 0),
        ),
        reverse=True,
    )[0]


def _benchmark_rows(
    benchmark_results: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    if benchmark_results is None:
        return ()
    if isinstance(benchmark_results, Mapping):
        rows = benchmark_results.get("results")
        if isinstance(rows, list):
            return tuple(row for row in rows if isinstance(row, Mapping))
        return (benchmark_results,)
    return tuple(row for row in benchmark_results if isinstance(row, Mapping))


def _detect_accelerator_backends(system: str, machine: str) -> tuple[str, ...]:
    backends: list[str] = []
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        backends.append("metal")
    if shutil.which("nvidia-smi"):
        backends.append("cuda")
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        backends.append("rocm")
    if not backends:
        backends.append("cpu")
    return tuple(dict.fromkeys(backends))


def _infer_runtime_kind(
    repo_id: str,
    source: str,
    path: str,
    include: Sequence[str],
) -> str:
    text = " ".join((repo_id, source, path, *include)).lower()
    if "mlx" in text or repo_id.startswith("mlx-community/"):
        return "mlx-lm"
    if "ollama" in source:
        return "ollama"
    if "lm_studio" in source:
        return "lmstudio"
    if "gguf" in text or any("gguf" in item.lower() for item in include):
        return "llama.cpp"
    return "generic"


def _infer_quantization(text: str) -> str | None:
    lowered = text.lower()
    patterns = (
        r"q[2-8]_[a-z]_[a-z]",
        r"q[2-8]_[a-z]",
        r"[248]bit",
        r"[248]-bit",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(0)
    return None


def _model_size_billions(text: str) -> float | None:
    lowered = text.lower()
    sizes = [
        float(match.group(1))
        for match in re.finditer(r"(?<![a-z])(\d+(?:\.\d+)?)b\b", lowered)
    ]
    return max(sizes) if sizes else None


def _estimated_memory_requirements(
    size_b: float | None,
    quantization: str | None,
    runtime_kind: str,
) -> tuple[float, float]:
    if size_b is None:
        return 8.0, 16.0
    quant = (quantization or "").lower()
    if "4" in quant or "q4" in quant:
        bytes_per_param = 0.65
    elif "5" in quant or "q5" in quant:
        bytes_per_param = 0.78
    elif "8" in quant or "q8" in quant:
        bytes_per_param = 1.15
    else:
        bytes_per_param = 1.35 if runtime_kind == "mlx-lm" else 1.0
    estimated_model_gb = size_b * bytes_per_param
    minimum = max(4.0, round(estimated_model_gb + 2.0, 1))
    recommended = max(8.0, round(estimated_model_gb * 1.75 + 4.0, 1))
    return minimum, recommended


def _local_quality_score(
    route: str,
    roles: Sequence[str],
    size_b: float | None,
    repo_id: str,
) -> int:
    score = 3
    if route in roles:
        score += 1
    text = repo_id.lower()
    if route == "code_agent" and any(token in text for token in ("code", "coder")):
        score += 1
    if route == "reasoning_local" and (
        "reason" in text or (size_b is not None and size_b >= 7)
    ):
        score += 1
    if route == "fast_local" and size_b is not None and size_b <= 1.5:
        score += 1
    return max(1, min(5, score))


def _local_speed_score(size_b: float | None, quantization: str | None) -> int:
    if size_b is None:
        score = 3
    elif size_b <= 1.5:
        score = 5
    elif size_b <= 4:
        score = 4
    elif size_b <= 8:
        score = 3
    else:
        score = 2
    quant = (quantization or "").lower()
    if "4" in quant or "q4" in quant:
        score += 1
    return max(1, min(5, score))


def _adapter_for_route(route: str) -> str:
    if route == "image_generation":
        return "local_image_generation"
    if route == "multimodal_vision":
        return "local_vision"
    if route == "reasoning_local":
        return "local_reasoning"
    if route == "web_research":
        return "web_research"
    if route == "code_agent":
        return "local_code"
    return "local_chat"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
