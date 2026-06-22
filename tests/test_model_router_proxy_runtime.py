import asyncio
from pathlib import Path
import subprocess
from typing import Any

import pytest

from hermes.plugins.model_router.proxy_config import (
    ProxyBackendConfig,
    ProxyRuntimeConfig,
)
from hermes.plugins.model_router.proxy_runtime import (
    ManagedRuntimeManager,
    RuntimeStartError,
)


class _FakeProcess:
    def __init__(self, *, wait_timeout: bool = False) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_timeout = wait_timeout

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.wait_timeout:
            self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_timeout and not self.killed:
            raise subprocess.TimeoutExpired("fake-runtime", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _backend(tmp_path: Path, *, idle_timeout: float = 900.0) -> ProxyBackendConfig:
    return ProxyBackendConfig(
        name="fast",
        base_url="http://127.0.0.1:8090/v1",
        model="fast-model",
        runtime=ProxyRuntimeConfig(
            enabled=True,
            kind="llama-server",
            command=("llama-server", "-m", "/models/fast.gguf", "--port", "8090"),
            readiness_url="http://127.0.0.1:8090/v1/models",
            readiness_timeout_seconds=2.0,
            idle_timeout_seconds=idle_timeout,
            shutdown_timeout_seconds=0.01,
            log_path=str(tmp_path / "fast.log"),
        ),
    )


def test_managed_runtime_starts_once_and_keeps_warm(tmp_path):
    started: list[list[str]] = []
    process_kwargs: list[dict[str, Any]] = []

    async def ready(_url: str, _timeout: float) -> bool:
        return True

    def process_factory(command: list[str], **_kwargs: Any) -> _FakeProcess:
        started.append(command)
        process_kwargs.append(_kwargs)
        return _FakeProcess()

    async def run() -> None:
        backend = _backend(tmp_path)
        manager = ManagedRuntimeManager(
            {"fast": backend},
            readiness_probe=ready,
            process_factory=process_factory,
        )

        await manager.ensure_running(backend)
        await manager.ensure_running(backend)
        await manager.stop_all()

    asyncio.run(run())

    assert started == [["llama-server", "-m", "/models/fast.gguf", "--port", "8090"]]
    assert process_kwargs[0]["shell"] is False
    assert process_kwargs[0]["stdin"] == subprocess.DEVNULL
    assert process_kwargs[0]["stderr"] == subprocess.STDOUT
    assert process_kwargs[0]["text"] is True


def test_managed_runtime_readiness_failure_stops_process(tmp_path):
    process = _FakeProcess()
    tick = 0.0

    def monotonic() -> float:
        nonlocal tick
        tick += 1.0
        return tick

    async def not_ready(_url: str, _timeout: float) -> bool:
        return False

    async def sleep(_seconds: float) -> None:
        return None

    async def run() -> None:
        backend = _backend(tmp_path)
        manager = ManagedRuntimeManager(
            {"fast": backend},
            readiness_probe=not_ready,
            process_factory=lambda *_args, **_kwargs: process,
            monotonic=monotonic,
            sleep=sleep,
        )

        with pytest.raises(RuntimeStartError, match="readiness failed"):
            await manager.ensure_running(backend)

    asyncio.run(run())

    assert process.terminated is True


def test_managed_runtime_stops_after_idle_timeout(tmp_path):
    process = _FakeProcess()
    now = 0.0

    def monotonic() -> float:
        return now

    async def ready(_url: str, _timeout: float) -> bool:
        return True

    async def run() -> None:
        nonlocal now
        backend = _backend(tmp_path, idle_timeout=5.0)
        manager = ManagedRuntimeManager(
            {"fast": backend},
            readiness_probe=ready,
            process_factory=lambda *_args, **_kwargs: process,
            monotonic=monotonic,
        )

        await manager.ensure_running(backend)
        now = 6.0
        await manager.stop_idle()

    asyncio.run(run())

    assert process.terminated is True


def test_managed_runtime_does_not_stop_active_request_until_released(tmp_path):
    process = _FakeProcess()
    now = 0.0

    def monotonic() -> float:
        return now

    async def ready(_url: str, _timeout: float) -> bool:
        return True

    async def run() -> None:
        nonlocal now
        backend = _backend(tmp_path, idle_timeout=5.0)
        manager = ManagedRuntimeManager(
            {"fast": backend},
            readiness_probe=ready,
            process_factory=lambda *_args, **_kwargs: process,
            monotonic=monotonic,
        )

        await manager.ensure_running(backend)
        manager.begin_request("fast")
        now = 6.0
        await manager.stop_idle()
        assert process.terminated is False
        manager.end_request("fast")
        now = 12.0
        await manager.stop_idle()

    asyncio.run(run())

    assert process.terminated is True


def test_managed_runtime_kills_process_after_shutdown_timeout(tmp_path):
    process = _FakeProcess(wait_timeout=True)

    async def ready(_url: str, _timeout: float) -> bool:
        return True

    async def run() -> None:
        backend = _backend(tmp_path)
        manager = ManagedRuntimeManager(
            {"fast": backend},
            readiness_probe=ready,
            process_factory=lambda *_args, **_kwargs: process,
        )

        await manager.ensure_running(backend)
        await manager.stop_all()

    asyncio.run(run())

    assert process.terminated is True
    assert process.killed is True
