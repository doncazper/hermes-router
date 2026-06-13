"""Typed data models for deterministic model routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _as_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class EngineAvailability:
    status: str = "auto"
    required_env: tuple[str, ...] = field(default_factory=tuple)
    required_commands: tuple[str, ...] = field(default_factory=tuple)
    required_paths: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EngineAvailability":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("availability must be a mapping")

        status = data.get("status", "auto")
        if status not in {"auto", "available", "unavailable"}:
            raise ValueError(
                "availability status must be auto, available, or unavailable"
            )

        return cls(
            status=status,
            required_env=_string_tuple(data, "required_env"),
            required_commands=_string_tuple(data, "required_commands"),
            required_paths=_string_tuple(data, "required_paths"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "required_env": list(self.required_env),
            "required_commands": list(self.required_commands),
            "required_paths": list(self.required_paths),
        }


def _string_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"availability {key!r} must be a list of strings")
    return tuple(item for item in value if item.strip())


@dataclass(frozen=True)
class ModelEngine:
    name: str
    provider: str
    model: str
    adapter: str
    strengths: tuple[str, ...]
    max_context: int
    cost_tier: str
    latency_tier: str
    enabled: bool
    fallback: str | None = None
    availability: EngineAvailability = field(default_factory=EngineAvailability)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ModelEngine":
        if not isinstance(data, dict):
            raise ValueError(f"engine {name!r} must be a mapping")

        required = (
            "provider",
            "model",
            "adapter",
            "strengths",
            "max_context",
            "cost_tier",
            "latency_tier",
            "enabled",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"engine {name!r} missing fields: {', '.join(missing)}")

        strengths = data["strengths"]
        if not isinstance(strengths, list) or not all(
            isinstance(item, str) for item in strengths
        ):
            raise ValueError(f"engine {name!r} strengths must be a list of strings")

        fallback = data.get("fallback")
        if fallback is not None and not isinstance(fallback, str):
            raise ValueError(f"engine {name!r} fallback must be a string or null")

        try:
            availability = EngineAvailability.from_dict(data.get("availability"))
        except ValueError as exc:
            raise ValueError(f"engine {name!r} {exc}") from exc

        return cls(
            name=name,
            provider=_require_string(data, "provider", name),
            model=_require_string(data, "model", name),
            adapter=_require_string(data, "adapter", name),
            strengths=tuple(strengths),
            max_context=_require_int(data, "max_context", name),
            cost_tier=_require_string(data, "cost_tier", name),
            latency_tier=_require_string(data, "latency_tier", name),
            enabled=_require_bool(data, "enabled", name),
            fallback=fallback,
            availability=availability,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "adapter": self.adapter,
            "strengths": list(self.strengths),
            "max_context": self.max_context,
            "cost_tier": self.cost_tier,
            "latency_tier": self.latency_tier,
            "enabled": self.enabled,
            "fallback": self.fallback,
            "availability": self.availability.to_dict(),
        }


@dataclass(frozen=True)
class EngineAvailabilityResult:
    engine: str
    available: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "available": self.available,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RouterAvailabilityReport:
    engines: dict[str, EngineAvailabilityResult]

    @property
    def all_available(self) -> bool:
        return bool(self.engines) and all(
            result.available for result in self.engines.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_available": self.all_available,
            "engines": {
                name: result.to_dict()
                for name, result in sorted(self.engines.items())
            },
        }


def _require_string(data: dict[str, Any], key: str, engine_name: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"engine {engine_name!r} field {key!r} must be a string")
    return value


def _require_int(data: dict[str, Any], key: str, engine_name: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"engine {engine_name!r} field {key!r} must be a positive int")
    return value


def _require_bool(data: dict[str, Any], key: str, engine_name: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"engine {engine_name!r} field {key!r} must be a bool")
    return value


@dataclass(frozen=True)
class PromptFeatures:
    prompt_length: int
    estimated_tokens: int
    simple_transform: bool = False
    coding_intent: bool = False
    research_intent: bool = False
    current_info_intent: bool = False
    multi_step_reasoning: bool = False
    tool_intent: bool = False
    file_intent: bool = False
    email_intent: bool = False
    calendar_intent: bool = False
    shell_intent: bool = False
    github_intent: bool = False
    vision_intent: bool = False
    image_generation_intent: bool = False
    legal_domain: bool = False
    medical_domain: bool = False
    financial_domain: bool = False
    sensitive_domain: bool = False
    destructive_action: bool = False
    external_action: bool = False
    purchase_action: bool = False
    send_action: bool = False
    structured_output: bool = False
    ambiguous: bool = False
    long_context: bool = False
    requires_tools: bool = False
    requires_freshness: bool = False
    requires_code_execution: bool = False
    requires_vision: bool = False
    requires_image_generation: bool = False
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComplexityScore:
    value: int
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class RiskScore:
    value: int
    requires_confirmation: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "requires_confirmation": self.requires_confirmation,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PromptAnalysis:
    complexity_score: ComplexityScore
    risk_score: RiskScore
    confidence_score: int
    features: PromptFeatures
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return _as_jsonable(asdict(self))


@dataclass(frozen=True)
class RoutingDecision:
    selected_engine: str
    fallback_engine: str | None
    complexity_score: int
    risk_score: int
    confidence_score: int
    reasons: tuple[str, ...]
    requires_confirmation: bool
    requires_tools: bool
    requires_freshness: bool
    requires_code_execution: bool
    requires_vision: bool
    requires_image_generation: bool
    config_valid: bool
    availability_valid: bool
    availability_reasons: tuple[str, ...]
    features: PromptFeatures

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "fallback_engine": self.fallback_engine,
            "complexity_score": self.complexity_score,
            "risk_score": self.risk_score,
            "confidence_score": self.confidence_score,
            "reasons": list(self.reasons),
            "requires_confirmation": self.requires_confirmation,
            "requires_tools": self.requires_tools,
            "requires_freshness": self.requires_freshness,
            "requires_code_execution": self.requires_code_execution,
            "requires_vision": self.requires_vision,
            "requires_image_generation": self.requires_image_generation,
            "config_valid": self.config_valid,
            "availability_valid": self.availability_valid,
            "availability_reasons": list(self.availability_reasons),
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class RoutingReceipt:
    selected_engine: str
    complexity_score: int
    risk_score: int
    confidence_score: int
    reasons: tuple[str, ...]
    fallback_engine: str | None
    requires_confirmation: bool
    requires_tools: bool
    requires_freshness: bool
    requires_code_execution: bool
    requires_vision: bool
    requires_image_generation: bool
    config_valid: bool
    availability_valid: bool
    availability_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "complexity_score": self.complexity_score,
            "risk_score": self.risk_score,
            "confidence_score": self.confidence_score,
            "reasons": list(self.reasons),
            "fallback_engine": self.fallback_engine,
            "requires_confirmation": self.requires_confirmation,
            "requires_tools": self.requires_tools,
            "requires_freshness": self.requires_freshness,
            "requires_code_execution": self.requires_code_execution,
            "requires_vision": self.requires_vision,
            "requires_image_generation": self.requires_image_generation,
            "config_valid": self.config_valid,
            "availability_valid": self.availability_valid,
            "availability_reasons": list(self.availability_reasons),
        }


@dataclass(frozen=True)
class RouterConfig:
    engines: dict[str, ModelEngine]
    routing_targets: dict[str, str]
    source_path: str | None = None

    def get_engine(self, name: str) -> ModelEngine | None:
        return self.engines.get(name)

    def target_engine(self, target: str) -> str | None:
        return self.routing_targets.get(target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engines": {
                name: engine.to_dict() for name, engine in sorted(self.engines.items())
            },
            "routing_targets": dict(sorted(self.routing_targets.items())),
            "source_path": self.source_path,
        }
