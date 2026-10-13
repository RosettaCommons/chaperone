"""Task future and state abstractions.

Provides TaskFuture and TaskState used by all task runners.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any


class TaskState(enum.Enum):
    """Possible states for a task execution."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskFuture(ABC):
    """Abstract base for task futures.

    Wraps the result of an asynchronous task submission, providing
    a uniform API for waiting on and retrieving results.
    """

    @property
    @abstractmethod
    def state(self) -> TaskState:
        """Return the current state of the task."""
        ...

    @abstractmethod
    def wait(self, timeout: float | None = None) -> None:
        """Block until the task completes or timeout expires.

        Args:
            timeout: Maximum seconds to wait. None = wait forever.

        Raises:
            TimeoutError: If the task does not complete within timeout.
        """
        ...

    @abstractmethod
    def result(
        self, timeout: float | None = None, raise_on_failure: bool = True
    ) -> Any:
        """Wait for completion and return the result.

        Args:
            timeout: Maximum seconds to wait.
            raise_on_failure: If True, re-raise exceptions from the task.

        Returns:
            The task return value.
        """
        ...


class _ConcurrentTaskFuture(TaskFuture):
    """Shared adapter for futures from Python's concurrent executors."""

    def __init__(self, future: Future):
        self._future = future

    def result(
        self, timeout: float | None = None, raise_on_failure: bool = True
    ) -> Any:
        try:
            return self._future.result(timeout=timeout)
        except Exception:
            if raise_on_failure:
                raise
            return None
