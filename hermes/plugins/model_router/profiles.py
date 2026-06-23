"""Routing profile definitions and hint-level constraints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RoutingProfile(StrEnum):
    """Plain-language routing modes, separate from engine categories."""

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    PRIVATE = "private"
    SAFE = "safe"


ROUTING_PROFILE_VALUES = tuple(profile.value for profile in RoutingProfile)


@dataclass(frozen=True)
class RoutingProfileConstraints:
    profile: RoutingProfile
    latency_sensitive: bool = False
    max_cost_tier: str | None = None
    max_latency_tier: str | None = None
    allowed_providers: tuple[str, ...] = ()
    strict_confirmation: bool = False
    reasons: tuple[str, ...] = ()


def coerce_routing_profile(value: Any) -> RoutingProfile:
    if isinstance(value, RoutingProfile):
        return value
    if value is None:
        return RoutingProfile.BALANCED
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "routing profile must be one of: " + ", ".join(ROUTING_PROFILE_VALUES)
        )
    normalized = value.strip().lower().replace("_", "-")
    try:
        return RoutingProfile(normalized)
    except ValueError as exc:
        raise ValueError(
            "routing profile must be one of: " + ", ".join(ROUTING_PROFILE_VALUES)
        ) from exc


def profile_constraints(profile: RoutingProfile) -> RoutingProfileConstraints:
    if profile == RoutingProfile.FAST:
        return RoutingProfileConstraints(
            profile=profile,
            latency_sensitive=True,
            max_cost_tier="low",
            max_latency_tier="medium",
            reasons=(
                "routing profile fast prefers low-latency and low-cost backends",
            ),
        )
    if profile == RoutingProfile.QUALITY:
        return RoutingProfileConstraints(
            profile=profile,
            reasons=(
                "routing profile quality permits stronger configured fallbacks",
            ),
        )
    if profile == RoutingProfile.PRIVATE:
        return RoutingProfileConstraints(
            profile=profile,
            allowed_providers=("local", "human"),
            reasons=(
                "routing profile private applies local-only provider policy",
            ),
        )
    if profile == RoutingProfile.SAFE:
        return RoutingProfileConstraints(
            profile=profile,
            strict_confirmation=True,
            reasons=(
                "routing profile safe applies stricter confirmation policy",
            ),
        )
    return RoutingProfileConstraints(
        profile=RoutingProfile.BALANCED,
        reasons=("routing profile balanced uses default deterministic routing",),
    )
