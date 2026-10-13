"""Task runner base class.

Provides the TaskRunner ABC that all runners (SLURM, GPU local, thread pool)
implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from typing import Any, Callable, Self

from pipelines.orchestration.annotations import unmapped
from pipelines.orchestration.futures import TaskFuture

logger = logging.getLogger(__name__)


class TaskRunner(ABC):
    """Abstract base class for task runners.

    Task runners manage the execution of functions, either locally
    (threads) or on a cluster (SLURM). They are used as context managers.

    Example:
        with SlurmTaskRunner(job_args=JobArgs(...)) as runner:
            futures = runner.map(my_fn, items=[1, 2, 3])
            results = [f.result() for f in futures]
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    @abstractmethod
    def submit(self, fn: Callable, **parameters: Any) -> TaskFuture:
        """Submit a single function call for execution.

        Args:
            fn: The function to execute.
            **parameters: Keyword arguments to pass to fn.

        Returns:
            A TaskFuture representing the pending result.
        """
        ...

    @abstractmethod
    def map(self, fn: Callable, **parameters: Any) -> list[TaskFuture]:
        """Submit multiple function calls (one per item in iterable params).

        Args:
            fn: The function to execute for each set of parameters.
            **parameters: Keyword arguments. Iterable values are mapped over;
                         non-iterable values are passed to every call.

        Returns:
            List of TaskFutures, one per mapped invocation.
        """
        ...

    def __enter__(self) -> Self:
        """Initialize resources on context entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Cleanup resources on context exit."""
        pass


class _ConcurrentTaskRunner(TaskRunner):
    """Shared lifecycle for runners backed by a concurrent executor."""

    _executor: Executor | None

    def __exit__(self, *args: Any) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        super().__exit__(*args)


def partition_parameters(
    parameters: dict[str, Any],
) -> tuple[dict[str, list], dict[str, Any]]:
    """Separate iterable parameters from static ones.

    Parameters wrapped with ``unmapped()`` are always treated as static.
    Iterables (except str, bytes, dict) are mapped over; everything else is static.

    Returns:
        Tuple of (iterable_params, static_params).
    """
    iterable_params: dict[str, list] = {}
    static_params: dict[str, Any] = {}

    for key, value in parameters.items():
        unwrapped = value
        is_unmapped = False

        if isinstance(value, unmapped):
            unwrapped = value.value
            is_unmapped = True

        is_iterable = hasattr(unwrapped, "__iter__") and not isinstance(
            unwrapped, (str, bytes, dict)
        )

        if is_iterable and not is_unmapped:
            iterable_params[key] = list(unwrapped)
        else:
            static_params[key] = unwrapped if is_unmapped else value

    return iterable_params, static_params


def _validate_map_parameters(iterable_params: dict[str, list]) -> int:
    """Return the common iterable length required by local task runners."""
    if not iterable_params:
        raise ValueError("No iterable parameters found for map()")

    lengths = {key: len(values) for key, values in iterable_params.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ValueError(f"Iterable parameters have mismatched lengths: {lengths}")
    if any(length == 0 for length in lengths.values()):
        raise ValueError("Empty iterable passed to map()")

    return unique_lengths.pop()
