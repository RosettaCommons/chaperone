"""Host-side asynchronous health watchdog for long-running agent trials."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable


class ControllerHealthWatchdog:
    def __init__(
        self,
        *,
        report_path: Path,
        probe: Callable[[], Awaitable[None]],
        on_failure: Callable[[BaseException], Awaitable[None]],
        interval_seconds: float = 60.0,
    ) -> None:
        self.report_path = report_path
        self.probe = probe
        self.on_failure = on_failure
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task | None = None

    def _write(self, status: str, **detail) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "status": status,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            **detail,
        }
        temporary = self.report_path.with_name(
            f".{self.report_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.report_path)

    def start(self) -> None:
        if self._task is not None:
            return
        self._write("healthy", message="watchdog started")
        self._task = asyncio.create_task(self._run(), name="controller-health-watchdog")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        current = {}
        try:
            current = json.loads(self.report_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        if current.get("status") != "failed":
            self._write("stopped", message="trial environment stopped cleanly")

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            try:
                await self.probe()
                self._write("healthy", message="controller execution path verified")
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                self._write(
                    "failed",
                    error={"type": type(error).__name__, "message": str(error)},
                )
                await self.on_failure(error)
                return
