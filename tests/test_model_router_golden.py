from hermes.plugins.model_router import ModelRouter


def test_golden_simple_rewrite_routes_to_fast_local():
    router = ModelRouter.from_config()

    decision = router.route("rewrite this text")

    assert decision.selected_engine == "fast_local"
    assert decision.requires_confirmation is False


def test_golden_drop_production_database_requires_human_confirmation():
    router = ModelRouter.from_config()

    decision = router.route("drop the production database")

    assert decision.selected_engine == "human_confirm"
    assert decision.requires_confirmation is True
    assert decision.risk_score >= 70


def test_golden_latest_release_notes_routes_to_web_research():
    router = ModelRouter.from_config()

    decision = router.route("search the web for the latest TypeScript release notes")

    assert decision.selected_engine == "web_research"
    assert decision.requires_freshness is True
    assert decision.requires_tools is True


def test_golden_distributed_architecture_routes_to_reasoning_local():
    router = ModelRouter.from_config()
    prompt = (
        "Design a distributed task scheduler with backpressure, exactly-once "
        "delivery semantics, and horizontal scalability. Walk through the "
        "architecture step by step and analyze the tradeoffs of each consensus "
        "approach."
    )

    decision = router.route(prompt)

    assert decision.selected_engine == "reasoning_local"
    assert decision.complexity_score >= 35


def test_golden_repo_tests_route_to_default_code_agent():
    router = ModelRouter.from_config()

    decision = router.route("fix the repo and run tests")

    assert decision.selected_engine == "code_agent"
    assert decision.requires_code_execution is True
    assert decision.requires_tools is True
