"""GPU-aware Local TaskRunner.

Provides GPU-aware local execution for pipelines by:
1. Auto-detecting available CUDA GPUs
2. Spawning one worker process per GPU with fixed CUDA_VISIBLE_DEVICES affinity
3. Running tasks via ProcessPoolExecutor — each worker owns its GPU environment

Usage:
    from pipelines.orchestration import GPULocalTaskRunner, detect_cuda_gpus

    gpu_ids = detect_cuda_gpus()
    if gpu_ids:
        runner = GPULocalTaskRunner(gpu_ids=gpu_ids)
    else:
        runner = ThreadPoolRunner(max_workers=4)

    with runner:
        futures = runner.map(my_fn, items=[1, 2, 3])
        results = [f.result() for f in futures]
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import resource
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional, Self

from pipelines.execution_tracking import detect_cuda_gpus
from pipelines.orchestration.base_runner import (
    _ConcurrentTaskRunner,
    _validate_map_parameters,
    partition_parameters,
)
from pipelines.orchestration.futures import _ConcurrentTaskFuture, TaskState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker initializer and task wrapper
# ─────────────────────────────────────────────────────────────────────────────

_worker_gpu_id: Optional[int] = None

# Minimum RLIMIT_NPROC soft limit per worker. JAX/XLA can spawn 10+ threads
# per process; with 8 GPUs the total easily exceeds the default 1024.
# See docs/issues/pthread-eagain-af3-scoring.md for context.
_MIN_NPROC_SOFT: int = 65536

# Thread-limiting env vars. On GPU workloads the heavy lifting is on-device,
# so constraining CPU thread pools prevents thread explosion without
# meaningful performance loss.  Only set if the user hasn't already.
_THREAD_LIMIT_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "XLA_GLOBAL_CPU_THREAD_COUNT": "4",
}


def _raise_nproc_limit() -> None:
    """Try to raise RLIMIT_NPROC soft limit to avoid pthread_create EAGAIN."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        if soft < _MIN_NPROC_SOFT:
            new_soft = min(_MIN_NPROC_SOFT, hard)
            resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, hard))
            logger.debug(f"Raised RLIMIT_NPROC soft limit: {soft} → {new_soft}")
    except (ValueError, OSError) as exc:
        logger.debug(f"Could not raise RLIMIT_NPROC: {exc}")


def _limit_worker_threads() -> None:
    """Cap CPU thread pools per worker to prevent pthread_create EAGAIN.

    Only sets env vars that aren't already present, so user overrides
    (e.g. OMP_NUM_THREADS=8) are respected.
    """
    for var, default in _THREAD_LIMIT_VARS.items():
        if var not in os.environ:
            os.environ[var] = default


def _gpu_worker_init(gpu_queue: multiprocessing.Queue) -> None:
    """Per-worker initializer: claim a GPU ID and set CUDA_VISIBLE_DEVICES."""
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)
    _raise_nproc_limit()
    _limit_worker_threads()
    logger.info(f"Worker {os.getpid()} bound to GPU {_worker_gpu_id}")


def _run_task_in_process(fn: Callable, parameters: dict[str, Any]) -> Any:
    """Execute a function in the worker process (GPU env already set)."""
    return fn(**parameters)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-Aware Future
# ─────────────────────────────────────────────────────────────────────────────


class GPULocalFuture(_ConcurrentTaskFuture):
    """TaskFuture wrapper for GPU-local task execution."""

    @property
    def state(self) -> TaskState:
        if not self._future.done():
            return TaskState.RUNNING
        try:
            self._future.result(timeout=0)
            return TaskState.COMPLETED
        except Exception:
            return TaskState.FAILED

    def wait(self, timeout: float = None) -> None:
        try:
            self._future.result(timeout=timeout)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# GPU-Aware Task Runner
# ─────────────────────────────────────────────────────────────────────────────


class GPULocalTaskRunner(_ConcurrentTaskRunner):
    """TaskRunner that executes tasks locally with one process per GPU.

    Spawns a ProcessPoolExecutor with one worker per GPU. Each worker has
    CUDA_VISIBLE_DEVICES permanently set in its environment, eliminating
    race conditions from shared-env threading approaches.

    Args:
        gpu_ids: List of GPU indices to use. If None, auto-detects.
        max_workers: Ignored — always set to len(gpu_ids) for fixed affinity.
    """

    def __init__(
        self,
        gpu_ids: List[int] = None,
        max_workers: int = None,
    ):
        super().__init__()

        if gpu_ids is None:
            gpu_ids = detect_cuda_gpus()
            if not gpu_ids:
                raise ValueError(
                    "No GPUs detected. Use ThreadPoolRunner for CPU-only execution."
                )

        self.gpu_ids = gpu_ids
        self.max_workers = len(gpu_ids)
        self._executor: Optional[ProcessPoolExecutor] = None

        if max_workers is not None and max_workers != len(gpu_ids):
            logger.warning(
                f"max_workers={max_workers} ignored — forced to {len(gpu_ids)} "
                f"(one worker per GPU)"
            )

        logger.info(
            f"GPULocalTaskRunner initialized with {len(gpu_ids)} GPUs, "
            f"max_workers={self.max_workers}"
        )

    def __enter__(self) -> Self:
        super().__enter__()
        _raise_nproc_limit()
        mp_ctx = multiprocessing.get_context("spawn")
        gpu_queue = mp_ctx.Queue()
        for gid in self.gpu_ids:
            gpu_queue.put(gid)

        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp_ctx,
            initializer=_gpu_worker_init,
            initargs=(gpu_queue,),
        )
        return self

    def submit(self, fn: Callable, **parameters: Any) -> GPULocalFuture:
        """Submit a task for GPU-local execution."""
        if self._executor is None:
            raise RuntimeError("GPULocalTaskRunner must be used as context manager")

        logger.info(
            f"Submitting task {getattr(fn, '__name__', 'unknown')} "
            f"to GPU worker pool ({len(self.gpu_ids)} GPUs)"
        )

        future = self._executor.submit(_run_task_in_process, fn, parameters)
        return GPULocalFuture(future=future)

    def map(self, fn: Callable, **parameters: Any) -> list[GPULocalFuture]:
        """Submit multiple tasks to the GPU worker pool."""
        if self._executor is None:
            raise RuntimeError("GPULocalTaskRunner must be used as context manager")

        iterable_params, static_params = partition_parameters(parameters)

        map_length = _validate_map_parameters(iterable_params)

        logger.info(
            f"Submitting {map_length} GPU tasks for {getattr(fn, '__name__', 'unknown')} "
            f"(pool has {len(self.gpu_ids)} GPUs)"
        )

        futures = []
        for i in range(map_length):
            params = {key: values[i] for key, values in iterable_params.items()}
            params.update(static_params)

            logger.debug(f"Submitting item {i}/{map_length}")

            future = self._executor.submit(_run_task_in_process, fn, params)
            futures.append(GPULocalFuture(future=future))

        return futures
