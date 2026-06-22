"""Managed local runtime processes for the optional proxy."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
from typing import Any, TextIO

from hermes.plugins.model_router.proxy_config import ProxyBackendConfig


ReadinessProbe = Callable[[str, float], Awaitable[bool]]
ProcessFactory = Callable[..., Any]


class RuntimeStartError(RuntimeError):
    """Raised when a configured managed runtime cannot become ready."""


@dataclass
class _RuntimeState:
    backend: ProxyBackendConfig
    process: Any | None = None
    log_handle: TextIO | None = None
    last_used: float = 0.0
    active_requests: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ManagedRuntimeManager:
    """Demand-loads configured local runtime processes for proxy backends."""

    def __init__(
        self,
        backends: dict[str, ProxyBackendConfig],
        *,
        readiness_probe: ReadinessProbe,
        process_factory: ProcessFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._states = {
            name: _RuntimeState(backend)
            for name, backend in backends.items()
            if backend.runtime.enabled
        }
        self._readiness_probe = readiness_probe
        self._process_factory = process_factory or subprocess.Popen
        self._monotonic = monotonic
        self._sleep = sleep

    @property
    def has_managed_backends(self) -> bool:
        return bool(self._states)

    def is_managed(self, backend_name: str) -> bool:
        return backend_name in self._states

    async def ensure_running(self, backend: ProxyBackendConfig) -> None:
        state = self._states.get(backend.name)
        if state is None:
            return
        async with state.lock:
            if _process_running(state.process):
                state.last_used = self._monotonic()
                return
            await self._start_locked(state)

    def touch(self, backend_name: str) -> None:
        state = self._states.get(backend_name)
        if state is not None:
            state.last_used = self._monotonic()

    def begin_request(self, backend_name: str) -> None:
        state = self._states.get(backend_name)
        if state is not None:
            state.active_requests += 1
            state.last_used = self._monotonic()

    def end_request(self, backend_name: str) -> None:
        state = self._states.get(backend_name)
        if state is not None:
            state.active_requests = max(0, state.active_requests - 1)
            state.last_used = self._monotonic()

    async def stop_idle(self) -> None:
        now = self._monotonic()
        for name, state in self._states.items():
            runtime = state.backend.runtime
            if not _process_running(state.process):
                continue
            if state.active_requests:
                continue
            if now - state.last_used >= runtime.idle_timeout_seconds:
                await self.stop_backend(name)

    async def reap_idle_forever(self, interval_seconds: float = 5.0) -> None:
        while True:
            await self._sleep(interval_seconds)
            await self.stop_idle()

    async def stop_backend(self, backend_name: str) -> None:
        state = self._states.get(backend_name)
        if state is None:
            return
        async with state.lock:
            await self._stop_locked(state)

    async def stop_all(self) -> None:
        for name in tuple(self._states):
            await self.stop_backend(name)

    async def _start_locked(self, state: _RuntimeState) -> None:
        runtime = state.backend.runtime
        log_path = Path(runtime.log_path).expanduser()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            raise RuntimeStartError(
                f"runtime {state.backend.name} log unavailable: {exc.__class__.__name__}"
            ) from exc
        try:
            process = self._process_factory(
                list(runtime.command),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
        except FileNotFoundError as exc:
            log_handle.close()
            raise RuntimeStartError(
                f"runtime {state.backend.name} command not found: {runtime.command[0]}"
            ) from exc
        except OSError as exc:
            log_handle.close()
            raise RuntimeStartError(
                f"runtime {state.backend.name} failed to start: {exc.__class__.__name__}"
            ) from exc

        state.process = process
        state.log_handle = log_handle
        state.last_used = self._monotonic()
        try:
            await self._wait_until_ready(state)
        except Exception:
            await self._stop_locked(state)
            raise

    async def _wait_until_ready(self, state: _RuntimeState) -> None:
        runtime = state.backend.runtime
        deadline = self._monotonic() + runtime.readiness_timeout_seconds
        last_error = "not ready"
        while self._monotonic() <= deadline:
            process = state.process
            if process is not None and process.poll() is not None:
                raise RuntimeStartError(
                    f"runtime {state.backend.name} exited before readiness"
                )
            timeout = max(0.1, min(1.0, deadline - self._monotonic()))
            try:
                if await self._readiness_probe(runtime.readiness_url, timeout):
                    state.last_used = self._monotonic()
                    return
                last_error = "readiness endpoint returned not ready"
            except Exception as exc:
                last_error = exc.__class__.__name__
            await self._sleep(0.2)
        raise RuntimeStartError(
            f"runtime {state.backend.name} readiness failed: {last_error}"
        )

    async def _stop_locked(self, state: _RuntimeState) -> None:
        process = state.process
        runtime = state.backend.runtime
        if process is not None and _process_running(process):
            process.terminate()
            try:
                await asyncio.to_thread(
                    process.wait,
                    timeout=runtime.shutdown_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, timeout=1.0)
        _close_log_handle(state)
        state.process = None
        state.last_used = 0.0
        state.active_requests = 0


def _process_running(process: Any | None) -> bool:
    return process is not None and process.poll() is None


def _close_log_handle(state: _RuntimeState) -> None:
    handle = state.log_handle
    state.log_handle = None
    if handle is not None:
        handle.close()
