"""Typed data models for deterministic model routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hermes.plugins.model_router.profiles import (
    RoutingProfile,
    coerce_routing_profile,
)


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
    supports_tools: bool = False
    modalities: tuple[str, ...] = field(default_factory=tuple)
    capability_tier: str = "standard"
    trust_tier: str = "standard"
    capability: int = 50
    trust: int = 60
    cost: int = 0
    latency: int = 50

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

        capability_tier = _optional_string(
            data,
            "capability_tier",
            _default_capability_tier(data),
            name,
        )
        trust_tier = _optional_string(
            data,
            "trust_tier",
            _default_trust_tier(data),
            name,
        )
        cost_tier = _require_string(data, "cost_tier", name)
        latency_tier = _require_string(data, "latency_tier", name)

        declared_modalities = _optional_string_tuple(data, "modalities", name)
        modalities = (
            declared_modalities
            if declared_modalities is not None
            else _default_modalities(name, data)
        )

        return cls(
            name=name,
            provider=_require_string(data, "provider", name),
            model=_require_string(data, "model", name),
            adapter=_require_string(data, "adapter", name),
            strengths=tuple(strengths),
            max_context=_require_int(data, "max_context", name),
            cost_tier=cost_tier,
            latency_tier=latency_tier,
            enabled=_require_bool(data, "enabled", name),
            fallback=fallback,
            availability=availability,
            supports_tools=_optional_bool(
                data,
                "supports_tools",
                _default_supports_tools(name, data),
                name,
            ),
            modalities=modalities,
            capability_tier=capability_tier,
            trust_tier=trust_tier,
            capability=_optional_score(
                data,
                "capability",
                _default_capability_score(capability_tier, data),
                name,
            ),
            trust=_optional_score(
                data,
                "trust",
                _default_trust_score(trust_tier),
                name,
            ),
            cost=_optional_score(
                data,
                "cost",
                _default_cost_score(cost_tier),
                name,
            ),
            latency=_optional_score(
                data,
                "latency",
                _default_latency_score(latency_tier),
                name,
            ),
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
            "supports_tools": self.supports_tools,
            "modalities": list(self.modalities),
            "capability_tier": self.capability_tier,
            "trust_tier": self.trust_tier,
            "capability": self.capability,
            "trust": self.trust,
            "cost": self.cost,
            "latency": self.latency,
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


def _optional_bool(
    data: dict[str, Any],
    key: str,
    default: bool,
    engine_name: str,
) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"engine {engine_name!r} field {key!r} must be a bool")
    return value


def _optional_string(
    data: dict[str, Any],
    key: str,
    default: str,
    engine_name: str,
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"engine {engine_name!r} field {key!r} must be a string")
    return value


def _optional_string_tuple(
    data: dict[str, Any],
    key: str,
    engine_name: str,
) -> tuple[str, ...] | None:
    """Return the parsed tuple, or ``None`` when the key is absent.

    Returning ``None`` for an absent key lets callers distinguish "not declared"
    (apply a default) from an explicit empty list (honor it).
    """
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"engine {engine_name!r} field {key!r} must be strings")
    return tuple(item for item in value if item.strip())


def _optional_score(
    data: dict[str, Any],
    key: str,
    default: int,
    engine_name: str,
) -> int:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(
            f"engine {engine_name!r} field {key!r} must be an int from 0 to 100"
        )
    return value


def _default_supports_tools(name: str, data: dict[str, Any]) -> bool:
    text = " ".join(
        str(part).lower()
        for part in (
            name,
            data.get("provider", ""),
            data.get("adapter", ""),
            data.get("model", ""),
            " ".join(data.get("strengths", []))
            if isinstance(data.get("strengths"), list)
            else "",
        )
    )
    return any(
        token in text
        for token in (
            "code",
            "codex",
            "claude",
            "tool",
            "web",
            "research",
            "rag",
            "vision",
            "ocr",
            "image generation",
            "diffusion",
            "confirm",
        )
    )


def _default_capability_tier(data: dict[str, Any]) -> str:
    max_context = data.get("max_context", 0)
    if isinstance(max_context, int) and max_context >= 65536:
        return "high"
    if isinstance(max_context, int) and max_context >= 16384:
        return "medium"
    return "standard"


def _default_trust_tier(data: dict[str, Any]) -> str:
    provider = str(data.get("provider", "")).lower()
    adapter = str(data.get("adapter", "")).lower()
    if provider == "human" or "confirmation" in adapter:
        return "critical"
    if provider in {"anthropic", "openai"}:
        return "high"
    return "standard"


def _default_modalities(name: str, data: dict[str, Any]) -> tuple[str, ...]:
    text = " ".join(
        str(part).lower()
        for part in (
            name,
            data.get("adapter", ""),
            data.get("model", ""),
            " ".join(data.get("strengths", []))
            if isinstance(data.get("strengths"), list)
            else "",
        )
    )
    modalities: list[str] = []
    if any(token in text for token in ("vision", "ocr", "screenshot", "chart", "image")):
        modalities.append("image")
    if "pdf" in text:
        modalities.append("pdf")
    if "audio" in text:
        modalities.append("audio")
    return tuple(dict.fromkeys(modalities))


def _default_capability_score(tier: str, data: dict[str, Any]) -> int:
    tier_scores = {
        "standard": 50,
        "medium": 70,
        "high": 85,
        "critical": 95,
    }
    score = tier_scores.get(tier, 50)
    max_context = data.get("max_context", 0)
    if isinstance(max_context, int) and max_context >= 128000:
        score = max(score, 90)
    elif isinstance(max_context, int) and max_context >= 65536:
        score = max(score, 80)
    elif isinstance(max_context, int) and max_context >= 16384:
        score = max(score, 65)
    return min(score, 100)


def _default_trust_score(tier: str) -> int:
    return {
        "standard": 60,
        "medium": 75,
        "high": 85,
        "critical": 100,
    }.get(tier, 60)


def _default_cost_score(tier: str) -> int:
    return {
        "none": 0,
        "free": 0,
        "low": 20,
        "medium": 50,
        "paid": 50,
        "high": 80,
    }.get(tier, 50)


def _default_latency_score(tier: str) -> int:
    return {
        "low": 15,
        "medium": 45,
        "high": 75,
        "manual": 100,
    }.get(tier, 50)


VALID_ATTACHMENTS = ("image", "pdf", "audio", "code")
VALID_COST_TIERS = ("none", "free", "low", "medium", "paid", "high")
VALID_LATENCY_TIERS = ("low", "medium", "high", "manual")
SCORING_DIMENSIONS = ("complexity", "risk", "confidence")


@dataclass(frozen=True)
class ConfirmationOverrides:
    allow_destructive_actions: bool = False
    allow_send_actions: bool = False
    allow_purchase_actions: bool = False
    allow_high_impact_external_actions: bool = False
    allow_ambiguous_high_impact: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ConfirmationOverrides":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("safety confirmation_overrides must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                "safety confirmation_overrides unknown keys: " + ", ".join(unknown)
            )
        return cls(
            allow_destructive_actions=_config_bool(
                data,
                "allow_destructive_actions",
                False,
                "safety confirmation_overrides",
            ),
            allow_send_actions=_config_bool(
                data,
                "allow_send_actions",
                False,
                "safety confirmation_overrides",
            ),
            allow_purchase_actions=_config_bool(
                data,
                "allow_purchase_actions",
                False,
                "safety confirmation_overrides",
            ),
            allow_high_impact_external_actions=_config_bool(
                data,
                "allow_high_impact_external_actions",
                False,
                "safety confirmation_overrides",
            ),
            allow_ambiguous_high_impact=_config_bool(
                data,
                "allow_ambiguous_high_impact",
                False,
                "safety confirmation_overrides",
            ),
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyConfig:
    require_human_confirmation: bool = True
    confirmation_overrides: ConfirmationOverrides = field(
        default_factory=ConfirmationOverrides
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SafetyConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("safety config must be a mapping")
        allowed = {"require_human_confirmation", "confirmation_overrides"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("safety config unknown keys: " + ", ".join(unknown))
        return cls(
            require_human_confirmation=_config_bool(
                data,
                "require_human_confirmation",
                True,
                "safety",
            ),
            confirmation_overrides=ConfirmationOverrides.from_dict(
                data.get("confirmation_overrides")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_human_confirmation": self.require_human_confirmation,
            "confirmation_overrides": self.confirmation_overrides.to_dict(),
        }


def _config_bool(
    data: dict[str, Any],
    key: str,
    default: bool,
    context: str,
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context} {key} must be a bool")
    return value


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, dict[str, int]] = field(default_factory=dict)
    saturation_k: int = 50

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScoringConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("scoring config must be a mapping")

        weights_data = data.get("weights", {})
        if not isinstance(weights_data, dict):
            raise ValueError("scoring weights must be a mapping")

        weights: dict[str, dict[str, int]] = {}
        for dimension, dimension_weights in weights_data.items():
            if dimension not in SCORING_DIMENSIONS:
                raise ValueError(
                    "scoring weights dimension must be complexity, risk, or "
                    "confidence"
                )
            if not isinstance(dimension_weights, dict):
                raise ValueError(f"scoring weights {dimension!r} must be a mapping")
            weights[dimension] = {}
            for feature, value in dimension_weights.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError("scoring weight feature names must be strings")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 100
                ):
                    raise ValueError(
                        f"scoring weight {dimension}.{feature} must be an int "
                        "from 0 to 100"
                    )
                weights[dimension][feature] = value

        saturation_k = data.get("saturation_k", 50)
        if (
            isinstance(saturation_k, bool)
            or not isinstance(saturation_k, int)
            or saturation_k <= 0
        ):
            raise ValueError("scoring saturation_k must be a positive int")

        return cls(weights=weights, saturation_k=saturation_k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {
                dimension: dict(sorted(weights.items()))
                for dimension, weights in sorted(self.weights.items())
            },
            "saturation_k": self.saturation_k,
        }


@dataclass(frozen=True)
class PromptSignal:
    dimension: str
    feature: str
    weight: int
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "feature": self.feature,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RoutingHints:
    force_engine: str | None = None
    profile: RoutingProfile = RoutingProfile.BALANCED
    latency_sensitive: bool = False
    max_cost_tier: str | None = None
    max_latency_tier: str | None = None
    attachments: tuple[str, ...] = field(default_factory=tuple)
    allowed_providers: tuple[str, ...] = field(default_factory=tuple)
    denied_providers: tuple[str, ...] = field(default_factory=tuple)
    local_only: bool = False
    hosted_allowed: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RoutingHints":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("routing hints must be a mapping")

        force_engine = data.get("force_engine")
        if force_engine is not None and (
            not isinstance(force_engine, str) or not force_engine.strip()
        ):
            raise ValueError("routing hint force_engine must be a string")

        attachments = data.get("attachments", [])
        if not isinstance(attachments, list) or not all(
            isinstance(item, str) and item in VALID_ATTACHMENTS
            for item in attachments
        ):
            raise ValueError(
                "routing hint attachments must be image, pdf, audio, or code"
            )

        return cls(
            force_engine=force_engine,
            profile=coerce_routing_profile(data.get("profile")),
            latency_sensitive=_hint_bool(data, "latency_sensitive"),
            max_cost_tier=_hint_tier(data, "max_cost_tier", VALID_COST_TIERS),
            max_latency_tier=_hint_tier(data, "max_latency_tier", VALID_LATENCY_TIERS),
            attachments=tuple(attachments),
            allowed_providers=_hint_string_tuple(
                data,
                "provider_allowlist",
                fallback_key="allowed_providers",
            ),
            denied_providers=_hint_string_tuple(data, "provider_denylist"),
            local_only=_hint_bool(data, "local_only"),
            hosted_allowed=_hint_optional_bool(data, "hosted_allowed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_engine": self.force_engine,
            "profile": self.profile.value,
            "latency_sensitive": self.latency_sensitive,
            "max_cost_tier": self.max_cost_tier,
            "max_latency_tier": self.max_latency_tier,
            "attachments": list(self.attachments),
            "allowed_providers": list(self.allowed_providers),
            "provider_allowlist": list(self.allowed_providers),
            "provider_denylist": list(self.denied_providers),
            "local_only": self.local_only,
            "hosted_allowed": self.hosted_allowed,
        }


def _hint_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"routing hint {key} must be a bool")
    return value


def _hint_optional_bool(data: dict[str, Any], key: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"routing hint {key} must be a bool")
    return value


def _hint_tier(
    data: dict[str, Any],
    key: str,
    valid: tuple[str, ...],
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"routing hint {key} must be one of: {', '.join(valid)}")
    return value


def _hint_string_tuple(
    data: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
) -> tuple[str, ...]:
    value = data.get(key, data.get(fallback_key, []) if fallback_key else [])
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"routing hint {key} must be a list of strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


@dataclass(frozen=True)
class ProviderPolicy:
    version: int = 1
    provider_allowlist: tuple[str, ...] = field(default_factory=tuple)
    provider_denylist: tuple[str, ...] = field(default_factory=tuple)
    local_only: bool = False
    hosted_allowed: bool = True
    max_cost_tier: str | None = None
    max_latency_tier: str | None = None
    route_pools: dict[str, "ProviderPolicy"] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        context: str = "provider_policy",
        allow_route_pools: bool = True,
    ) -> "ProviderPolicy":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"{context} must be a mapping")

        allowed = {
            "version",
            "provider_allowlist",
            "provider_denylist",
            "local_only",
            "hosted_allowed",
            "max_cost_tier",
            "max_latency_tier",
            "route_pools",
        }
        if not allow_route_pools:
            allowed.remove("route_pools")
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"{context} unknown keys: " + ", ".join(unknown))

        route_pools: dict[str, ProviderPolicy] = {}
        raw_pools = data.get("route_pools", {})
        if raw_pools is None:
            raw_pools = {}
        if not isinstance(raw_pools, dict):
            raise ValueError(f"{context} route_pools must be a mapping")
        for route, raw_policy in raw_pools.items():
            if not isinstance(route, str) or not route.strip():
                raise ValueError(f"{context} route_pools keys must be strings")
            route_pools[route.strip()] = cls.from_dict(
                raw_policy,
                context=f"{context}.route_pools.{route}",
                allow_route_pools=False,
            )

        return cls(
            version=_policy_version(data, context),
            provider_allowlist=_policy_string_tuple(
                data,
                "provider_allowlist",
                context=context,
            ),
            provider_denylist=_policy_string_tuple(
                data,
                "provider_denylist",
                context=context,
            ),
            local_only=_policy_bool(data, "local_only", False, context),
            hosted_allowed=_policy_bool(data, "hosted_allowed", True, context),
            max_cost_tier=_policy_tier(
                data,
                "max_cost_tier",
                VALID_COST_TIERS,
                context,
            ),
            max_latency_tier=_policy_tier(
                data,
                "max_latency_tier",
                VALID_LATENCY_TIERS,
                context,
            ),
            route_pools=route_pools,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "provider_allowlist": list(self.provider_allowlist),
            "provider_denylist": list(self.provider_denylist),
            "local_only": self.local_only,
            "hosted_allowed": self.hosted_allowed,
            "max_cost_tier": self.max_cost_tier,
            "max_latency_tier": self.max_latency_tier,
            "route_pools": {
                route: policy.to_dict()
                for route, policy in sorted(self.route_pools.items())
            },
        }


def _policy_version(data: dict[str, Any], context: str) -> int:
    value = data.get("version", 1)
    if isinstance(value, bool) or value != 1:
        raise ValueError(f"{context} version must be 1")
    return 1


def _policy_string_tuple(
    data: dict[str, Any],
    key: str,
    *,
    context: str,
) -> tuple[str, ...]:
    value = data.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{context} {key} must be a list of strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _policy_bool(
    data: dict[str, Any],
    key: str,
    default: bool,
    context: str,
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context} {key} must be a bool")
    return value


def _policy_tier(
    data: dict[str, Any],
    key: str,
    valid: tuple[str, ...],
    context: str,
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"{context} {key} must be one of: {', '.join(valid)}")
    return value


@dataclass(frozen=True)
class RoutingRequirements:
    routing_profile: RoutingProfile = RoutingProfile.BALANCED
    needs_tools: bool = False
    required_modalities: tuple[str, ...] = field(default_factory=tuple)
    max_cost_tier: str | None = None
    max_latency_tier: str | None = None
    allowed_providers: tuple[str, ...] = field(default_factory=tuple)
    denied_providers: tuple[str, ...] = field(default_factory=tuple)
    profile_reasons: tuple[str, ...] = field(default_factory=tuple)
    provider_policy_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_profile": self.routing_profile.value,
            "needs_tools": self.needs_tools,
            "required_modalities": list(self.required_modalities),
            "max_cost_tier": self.max_cost_tier,
            "max_latency_tier": self.max_latency_tier,
            "allowed_providers": list(self.allowed_providers),
            "provider_denylist": list(self.denied_providers),
            "profile_reasons": list(self.profile_reasons),
            "provider_policy_reasons": list(self.provider_policy_reasons),
        }


@dataclass(frozen=True)
class EngineRejection:
    engine: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "reason": self.reason}


@dataclass(frozen=True)
class RoutingAlternative:
    engine: str
    rank_score: int
    capability: int
    trust: int
    cost: int
    latency: int
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "rank_score": self.rank_score,
            "capability": self.capability,
            "trust": self.trust,
            "cost": self.cost,
            "latency": self.latency,
            "reasons": list(self.reasons),
        }


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
    high_impact_external_action: bool = False
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
    signals: tuple[PromptSignal, ...] = field(default_factory=tuple)

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
    requirements: RoutingRequirements = field(default_factory=RoutingRequirements)
    rejected_engines: tuple[EngineRejection, ...] = field(default_factory=tuple)
    alternatives: tuple[RoutingAlternative, ...] = field(default_factory=tuple)
    fallback_used: bool = False
    routing_profile: RoutingProfile = RoutingProfile.BALANCED

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "routing_profile": self.routing_profile.value,
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
            "requirements": self.requirements.to_dict(),
            "rejected_engines": [
                rejection.to_dict() for rejection in self.rejected_engines
            ],
            "alternatives": [
                alternative.to_dict() for alternative in self.alternatives
            ],
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class RoutingReceipt:
    selected_engine: str
    routing_profile: RoutingProfile
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
    requirements: RoutingRequirements = field(default_factory=RoutingRequirements)
    rejected_engines: tuple[EngineRejection, ...] = field(default_factory=tuple)
    alternatives: tuple[RoutingAlternative, ...] = field(default_factory=tuple)
    fallback_used: bool = False
    summary: str = ""
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    selected_route_explanation: str = ""
    policy_explanation: str = ""
    rejection_explanation: str = ""
    fallback_explanation: str = ""
    safety_explanation: str = ""
    privacy_explanation: str = ""
    wrong_route_next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "routing_profile": self.routing_profile.value,
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
            "requirements": self.requirements.to_dict(),
            "rejected_engines": [
                rejection.to_dict() for rejection in self.rejected_engines
            ],
            "alternatives": [
                alternative.to_dict() for alternative in self.alternatives
            ],
            "fallback_used": self.fallback_used,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
            "selected_route_explanation": self.selected_route_explanation,
            "policy_explanation": self.policy_explanation,
            "rejection_explanation": self.rejection_explanation,
            "fallback_explanation": self.fallback_explanation,
            "safety_explanation": self.safety_explanation,
            "privacy_explanation": self.privacy_explanation,
            "wrong_route_next_action": self.wrong_route_next_action,
        }


@dataclass(frozen=True)
class RouterConfig:
    engines: dict[str, ModelEngine]
    routing_targets: dict[str, str]
    source_path: str | None = None
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    provider_policy: ProviderPolicy = field(default_factory=ProviderPolicy)

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
            "scoring": self.scoring.to_dict(),
            "safety": self.safety.to_dict(),
            "provider_policy": self.provider_policy.to_dict(),
        }
