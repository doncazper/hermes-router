"""Admin-owned proxy process supervision."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass
class ProxyProcessStatus:
    state: str
    pid: int | None = None
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": self.state}
        if self.pid is not None:
            payload["pid"] = self.pid
        if self.log_path is not None:
            payload["log_path"] = self.log_path
        return payload


class ProxyProcessSupervisor:
    """Settings-owned supervisor for the proxy process only."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        log_path: str | Path,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.config_path = Path(config_path).expanduser()
        self.log_path = Path(log_path).expanduser()
        self.process_factory = process_factory
        self._process: subprocess.Popen | None = None
        self._log_handle: Any | None = None

    def status(self) -> ProxyProcessStatus:
        if self._process is None:
            return ProxyProcessStatus("stopped", log_path=str(self.log_path))
        returncode = self._process.poll()
        if returncode is None:
            return ProxyProcessStatus(
                "running",
                pid=self._process.pid,
                log_path=str(self.log_path),
            )
        self._close_log_handle()
        return ProxyProcessStatus("stopped", log_path=str(self.log_path))

    def start(self) -> ProxyProcessStatus:
        status = self.status()
        if status.state == "running":
            return status
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        command = [
            sys.executable,
            "-m",
            "hermes.plugins.model_router.proxy",
            "--config",
            str(self.config_path),
        ]
        self._process = self.process_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        return self.status()

    def stop(self, *, timeout_seconds: float = 5.0) -> ProxyProcessStatus:
        if self._process is None:
            self._close_log_handle()
            return ProxyProcessStatus("stopped", log_path=str(self.log_path))
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=timeout_seconds)
        self._close_log_handle()
        return ProxyProcessStatus("stopped", log_path=str(self.log_path))

    def restart(self) -> ProxyProcessStatus:
        self.stop()
        return self.start()

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
