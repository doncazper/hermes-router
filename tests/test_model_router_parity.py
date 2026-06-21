"""Parity and routing-correctness tests for the default catalog.

These cover the bugs fixed in the routing-correctness change:

- Benign tool-mentioning prompts must not fail closed to ``human_confirm``.
- The default catalog fallback chains must terminate (no cycles).
- ``route_fast`` must agree with ``route`` on a curated corpus and must never
  be *weaker* than ``route`` for lexical high-risk actions.
"""

from __future__ import annotations

import pytest

from hermes.plugins.model_router import ModelRouter
from hermes.plugins.model_router.config import load_router_config


@pytest.fixture(scope="module")
def router() -> ModelRouter:
    return ModelRouter.from_config()


# Prompts where the fast path and the rich path must select the same engine.
PARITY_CORPUS = (
    # original golden set
    ("rewrite this text", "fast_local"),
    ("Summarize the attached meeting notes into three bullets.", "balanced_local"),
    ("fix the repo and run tests", "code_agent"),
    ("search the web for the latest TypeScript release notes", "web_research"),
    ("Extract text from this screenshot.", "multimodal_vision"),
    ("Generate an image of a Hermes router dashboard.", "image_generation"),
    ("drop the production database", "human_confirm"),
    # previously-divergent cases now expected to agree
    ("write a test for this function", "code_agent"),
    ("fix this", "reasoning_local"),
    ("search for 2024 tax brackets", "web_research"),
    ("latest Qwen model releases", "web_research"),
    ("what is the current LM Studio version", "web_research"),
    ("recent llama.cpp benchmark results", "web_research"),
    ("email formatting tips", "balanced_local"),
    ("summarize my inbox", "balanced_local"),
    ("check my calendar for next week", "balanced_local"),
    ("plan the rollout and update the config files", "reasoning_local"),
    ("refactor this codebase and fix the failing build", "code_agent"),
    ("analyze this stack trace and explain the exception", "code_agent"),
)


@pytest.mark.parametrize("prompt,expected", PARITY_CORPUS)
def test_route_matches_expected_engine(router, prompt, expected):
    assert router.route(prompt).selected_engine == expected


@pytest.mark.parametrize("prompt,expected", PARITY_CORPUS)
def test_route_fast_matches_route(router, prompt, expected):
    assert router.route_fast(prompt) == router.route(prompt).selected_engine


BENIGN_TOOL_PROMPTS = (
    "summarize my inbox",
    "check my calendar for next week",
    "plan the rollout and update the config files",
    "read the file and tell me what it does",
)


@pytest.mark.parametrize("prompt", BENIGN_TOOL_PROMPTS)
def test_benign_tool_prompts_do_not_fail_closed(router, prompt):
    decision = router.route(prompt)
    assert decision.selected_engine != "human_confirm"
    assert decision.requires_confirmation is False


# Lexical high-risk actions the fast path must escalate exactly like the rich path.
RISKY_CORPUS = (
    "delete all my emails",
    "send the report to the team",
    "buy two plane tickets",
    "deploy to production",
    "merge this pull request",
    "wire the payment now",
    "please delete.",
)


@pytest.mark.parametrize("prompt", RISKY_CORPUS)
def test_route_fast_is_never_weaker_than_route(router, prompt):
    rich = router.route(prompt).selected_engine
    if rich == "human_confirm":
        assert router.route_fast(prompt) == "human_confirm"


def test_default_config_fallback_chains_terminate():
    config = load_router_config()
    for name, engine in config.engines.items():
        seen: set[str] = set()
        current: str | None = name
        while current is not None:
            assert current not in seen, f"fallback cycle reachable from {name!r}: {current!r}"
            seen.add(current)
            node = config.get_engine(current)
            assert node is not None, f"engine {current!r} referenced but undefined"
            current = node.fallback
