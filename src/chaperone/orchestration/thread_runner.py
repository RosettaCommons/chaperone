"""Thread pool task runner.

Lightweight task runner using concurrent.futures.ThreadPoolExecutor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Self

from chaperone.orchestration.base_runner import (
    _ConcurrentTaskRunner,
    _validate_map_parameters,
    partition_parameters,
)
from chaperone.orchestration.futures import _ConcurrentTaskFuture, TaskState


class ThreadFuture(_ConcurrentTaskFuture):
    """TaskFuture backed by a concurrent.futures.Future."""

    @property
    def state(self) -> TaskState:
        if self._future.done():
            try:
                self._future.result(timeout=0)
                return TaskState.COMPLETED
            except Exception:
                return TaskState.FAILED
        return TaskState.RUNNING

    def wait(self, timeout: float | None = None) -> None:
        self._future.result(timeout=timeout)


class ThreadPoolRunner(_ConcurrentTaskRunner):
    """Task runner that executes functions in a thread pool.

    Args:
        max_workers: Maximum number of concurrent threads.
    """

    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> Self:
        super().__enter__()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def submit(self, fn: Callable, **parameters: Any) -> ThreadFuture:
        if self._executor is None:
            raise RuntimeError("ThreadPoolRunner must be used as context manager")
        future = self._executor.submit(fn, **parameters)
        return ThreadFuture(future)

    def map(self, fn: Callable, **parameters: Any) -> list[ThreadFuture]:
        if self._executor is None:
            raise RuntimeError("ThreadPoolRunner must be used as context manager")

        iterable_params, static_params = partition_parameters(parameters)

        map_length = _validate_map_parameters(iterable_params)
        futures = []
        for i in range(map_length):
            params = {key: values[i] for key, values in iterable_params.items()}
            params.update(static_params)
            future = self._executor.submit(fn, **params)
            futures.append(ThreadFuture(future))

        return futures
