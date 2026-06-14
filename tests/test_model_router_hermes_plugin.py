import argparse
import json

from hermes.plugins.model_router import hermes_plugin


class FakeContext:
    def __init__(self) -> None:
        self.cli_commands = {}
        self.slash_commands = {}

    def register_cli_command(self, *, name, help, setup_fn, handler_fn):
        self.cli_commands[name] = {
            "help": help,
            "setup_fn": setup_fn,
            "handler_fn": handler_fn,
        }

    def register_command(self, name, *, handler, description="", args_hint=""):
        self.slash_commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }


def test_plugin_registers_cli_and_slash_command_without_router_init(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("register must not instantiate ModelRouter")

    monkeypatch.setattr(hermes_plugin.ModelRouter, "from_config", explode)
    ctx = FakeContext()

    hermes_plugin.register(ctx)

    assert set(ctx.cli_commands) == {"router"}
    assert set(ctx.slash_commands) == {"router"}
    assert ctx.slash_commands["router"]["args_hint"] == "status | decide <prompt>"


def test_plugin_cli_setup_delegates_router_subcommands():
    ctx = FakeContext()
    hermes_plugin.register(ctx)
    parser = argparse.ArgumentParser()

    ctx.cli_commands["router"]["setup_fn"](parser)
    args = parser.parse_args(["decide", "--json", "rewrite this text"])

    assert args.command == "decide"
    assert args.prompt == ["rewrite this text"]
    assert callable(args.func)


def test_slash_status_returns_json_payload():
    ctx = FakeContext()
    hermes_plugin.register(ctx)

    response = ctx.slash_commands["router"]["handler"]("status")

    payload = json.loads(response)
    assert payload["ok"] is True
    assert payload["plugin"] == "hermes-router"
    assert payload["api"] == "route_fast"
    assert payload["config_valid"] is True
    assert payload["config_source"]


def test_slash_decide_uses_route_fast_and_returns_selected_engine():
    ctx = FakeContext()
    hermes_plugin.register(ctx)

    response = ctx.slash_commands["router"]["handler"](
        "decide fix the repo and run tests"
    )

    payload = json.loads(response)
    assert payload["ok"] is True
    assert payload["api"] == "route_fast"
    assert payload["selected_engine"] == "code_agent"


def test_slash_handler_never_raises_on_bad_command():
    response = hermes_plugin._handle_slash_command("unknown")

    payload = json.loads(response)
    assert payload["ok"] is False
    assert "unknown router command" in payload["error"]
