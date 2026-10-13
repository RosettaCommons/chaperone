"""SLURM TaskRunner.

Submits tasks to SLURM via submitit with support for:
- Single task submission
- Job arrays for map() operations
- Batching (multiple items per SLURM job)

Usage:
    from chaperone.orchestration import SlurmTaskRunner, JobArgs

    runner = SlurmTaskRunner(
        job_args=JobArgs(gpu_class="small", mem_gb=8, job_name="my_pipeline")
    )
    with runner:
        futures = runner.map(my_fn, items=[1, 2, 3])
        results = [f.result() for f in futures]
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from enum import Enum
from typing import Any, Callable, Self

import submitit
from chaperone.common import get_console
from chaperone.orchestration.base_runner import TaskRunner, partition_parameters
from chaperone.orchestration.job_args import JobArgs, _create_slurm_executor
from chaperone.orchestration.slurm_executors import (
    _run_batch_in_slurm,
    _run_fn_in_slurm,
    _sanitize_env,
    cloudpickle_fn,
)
from chaperone.orchestration.slurm_futures import (
    ArrayStatePoller,
    SlurmArrayFuture,
    SlurmBatchedItemFuture,
    SlurmFuture,
)

console = get_console()
logger = logging.getLogger(__name__)


class ComputeBackend(str, Enum):
    """Compute backend for task execution."""

    SLURM = "slurm"
    LOCAL = "local"


# Constants for SLURM configuration
DEFAULT_MAX_ARRAY_SIZE = 1000
DEFAULT_POLL_TIME_MULTIPLIER = 2
# Floor on the auto-computed max_poll_time. SLURM can hold a job for
# many hours before it ever runs (e.g. JobHeldUser / "user env retrieval
# failed requeued held"), so a poll budget derived purely from the
# block's time_limit (3h * 2 = 6h) gives up while the job is still
# legitimately queued. 24h is generous enough to ride out queue stalls
# without indefinitely blocking a wedged driver.
MIN_MAX_POLL_TIME_S = 24 * 3600


def _default_max_poll(job_args_timeout_min: int) -> float:
    return max(
        job_args_timeout_min * 60 * DEFAULT_POLL_TIME_MULTIPLIER,
        MIN_MAX_POLL_TIME_S,
    )


def _check_existing_slurm_jobs(job_name: str) -> list[dict]:
    """Check for existing PENDING or RUNNING SLURM jobs with the given name."""
    try:
        result = subprocess.run(
            ["squeue", "-n", job_name, "--me", "-h", "-o", "%i %T %K"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                job_id = parts[0]
                state = parts[1]
                array_count = parts[2] if len(parts) > 2 else "1"
                jobs.append(
                    {
                        "job_id": job_id,
                        "state": state,
                        "array_task_count": array_count,
                    }
                )
        return jobs
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


class SlurmTaskRunner(TaskRunner):
    """TaskRunner that submits tasks to SLURM via submitit.

    Configure SLURM parameters via JobArgs. Use partition='local' for local execution.
    Without a gres, a job that wants a GPU (a gpu-named partition) gets one GPU per node;
    otherwise none. Partition, gres, account, qos and extra sbatch options that JobArgs
    leaves unset come from the CHAPERONE_* site settings.

    Args:
        job_args: SLURM job configuration (partition, mem_gb, time_limit, job_name).
        log_folder: Directory for job logs (default: "slurm_logs").
        slurm_array_parallelism: Max concurrent array tasks. Resolution order is
            this argument, JobArgs, CHAPERONE_SLURM_ARRAY_PARALLELISM, then 1000.
        poll_interval: Seconds between state polls (default: 1s local, 30s SLURM).
        max_poll_time: Max seconds to wait for job (default: max(2x time_limit, 24h)).
        fail_on_error: If True, raise on first task failure (default: True).
        items_per_job: Items per job for batching in map() (default: 1=no batching).
        max_array_size: Max jobs per array (default: auto-detect from cluster).
        check_duplicates: If True, warn when submitting jobs with same name as
            existing PENDING/RUNNING jobs (default: True).
        **slurm_kwargs: Additional submitit parameters.
    """

    def __init__(
        self,
        job_args: JobArgs | None = None,
        log_folder: str = "slurm_logs",
        slurm_array_parallelism: int | None = None,
        poll_interval: float | None = None,
        max_poll_time: float | None = None,
        fail_on_error: bool = True,
        items_per_job: int = 1,
        max_array_size: int | None = None,
        check_duplicates: bool = False,
        slurm_kwargs: Any = {},
        # Backwards compatibility: accept compute_backend kwarg
        compute_backend: str | ComputeBackend | None = None,
    ):
        super().__init__()

        # Backwards compatibility: convert compute_backend kwarg to partition
        if compute_backend is not None:
            if job_args is None:
                job_args = JobArgs()
            if compute_backend == "local" or (
                isinstance(compute_backend, ComputeBackend)
                and compute_backend == ComputeBackend.LOCAL
            ):
                job_args.partition = "local"

        # Also check slurm_kwargs for backwards compatibility
        if "compute_backend" in slurm_kwargs:
            cb = slurm_kwargs.pop("compute_backend")
            if job_args is None:
                job_args = JobArgs()
            if cb == "local" or (
                isinstance(cb, ComputeBackend) and cb == ComputeBackend.LOCAL
            ):
                job_args.partition = "local"

        if job_args is None:
            job_args = JobArgs()

        self._job_args = job_args

        self.partition = job_args.effective_partition()
        self.time_limit = job_args.time_limit
        self.mem_gb = job_args.mem_gb
        self.job_name = job_args.job_name
        self.gres = job_args.effective_gres()
        self.constraint = job_args.constraint
        self.cpus_per_task = job_args.cpus_per_task

        if (self.partition or "").lower() == "local":
            self.compute_backend = ComputeBackend.LOCAL
        else:
            self.compute_backend = ComputeBackend.SLURM

        self.gpus_per_node = 1 if job_args.wants_gpu else 0

        self.log_folder = log_folder
        self.slurm_array_parallelism = job_args.effective_slurm_array_parallelism(
            slurm_array_parallelism
        )
        self.max_poll_time = max_poll_time
        self.fail_on_error = fail_on_error
        self.max_array_size = max_array_size
        self.check_duplicates = check_duplicates
        self.slurm_kwargs = slurm_kwargs

        if job_args.exclude:
            self.slurm_kwargs["exclude"] = job_args.exclude

        if items_per_job < 1:
            raise ValueError(f"items_per_job must be >= 1, got {items_per_job}")
        self.items_per_job = items_per_job

        if poll_interval is None:
            self.poll_interval = (
                1.0 if self.compute_backend == ComputeBackend.LOCAL else 30.0
            )
        else:
            self.poll_interval = poll_interval

        self._executor: submitit.AutoExecutor | submitit.LocalExecutor | None = None
        self._cached_max_array_size: int | None = None

    def _format_resources(self) -> str:
        """Format resource specification for logging."""
        parts = [f"partition={self.partition or '(cluster default)'}"]
        if self.gres:
            parts.append(f"gres={self.gres}")
        parts.append(f"cpus={self.cpus_per_task}")
        parts.append(f"mem={self.mem_gb}GB")
        parts.append(f"time={self.time_limit}")
        return ", ".join(parts)

    def _log_submission(
        self,
        job_id: str,
        num_jobs: int = 1,
        num_rows: int | None = None,
        task_name: str = None,
    ):
        """Log a job submission message."""
        resource_info = self._format_resources()
        num_rows = num_rows or num_jobs
        rows_per_job = num_rows // num_jobs if num_jobs else 0
        if num_jobs == 1:
            job_desc = f"job {job_id} ({num_rows} rows)"
        else:
            job_desc = (
                f"job array {job_id} ({num_rows} rows, {num_jobs} jobs, "
                f"{rows_per_job} rows/job)"
            )
        task_info = f" for {task_name}" if task_name else ""
        logger.info(f"Submitted {job_desc}{task_info} ({resource_info})")

    def __enter__(self) -> Self:
        """Initialize submitit executor on context entry."""
        super().__enter__()

        if self.compute_backend == ComputeBackend.LOCAL:
            self._executor = submitit.LocalExecutor(folder=self.log_folder)
            self._executor.update_parameters(
                timeout_min=self._job_args.timeout_min,
            )
        else:
            self._executor = _create_slurm_executor(folder=f"{self.log_folder}/%j")

            params = {
                "timeout_min": self._job_args.timeout_min,
                "mem_gb": self.mem_gb,
                "cpus_per_task": self.cpus_per_task,
                "slurm_array_parallelism": self.slurm_array_parallelism,
                "slurm_srun_args": ["--export=ALL"],
                **self.slurm_kwargs,
            }

            if self.partition:
                params.setdefault("slurm_partition", self.partition)

            if self.gres is not None:
                params["slurm_gres"] = self.gres
            else:
                params["gpus_per_node"] = self.gpus_per_node

            if self.constraint is not None:
                params["slurm_constraint"] = self.constraint

            if self.job_name is not None:
                params["name"] = self.job_name

            if self._job_args.nice is not None:
                params.setdefault("slurm_additional_parameters", {})["nice"] = (
                    self._job_args.nice
                )

            self._job_args.add_site_sbatch_params(params)
            self._executor.update_parameters(**params)

        return self

    def __exit__(self, *args: Any) -> None:
        """Cleanup on context exit."""
        self._executor = None
        super().__exit__(*args)

    def duplicate(self) -> Self:
        """Return new instance with same config."""
        return SlurmTaskRunner(
            job_args=JobArgs(
                partition=self.partition,
                time_limit=self.time_limit,
                mem_gb=self.mem_gb,
                job_name=self.job_name,
                gres=self.gres,
                account=self._job_args.account,
                qos=self._job_args.qos,
                extra_sbatch=self._job_args.extra_sbatch,
                constraint=self.constraint,
                cpus_per_task=self.cpus_per_task,
                exclude=self._job_args.exclude,
                nice=self._job_args.nice,
                rows_per_job=self._job_args.rows_per_job,
                num_chunks=self._job_args.num_chunks,
                local_workers=self._job_args.local_workers,
                slurm_array_parallelism=self._job_args.slurm_array_parallelism,
            ),
            log_folder=self.log_folder,
            slurm_array_parallelism=self.slurm_array_parallelism,
            poll_interval=self.poll_interval,
            max_poll_time=self.max_poll_time,
            fail_on_error=self.fail_on_error,
            items_per_job=self.items_per_job,
            max_array_size=self.max_array_size,
            check_duplicates=getattr(self, "check_duplicates", False),
            slurm_kwargs=dict(**self.slurm_kwargs),
        )

    def submit(self, fn: Callable, **parameters: Any) -> SlurmFuture:
        """Submit a single function for execution on SLURM.

        Args:
            fn: Function to execute.
            **parameters: Keyword arguments to pass to fn.

        Returns:
            SlurmFuture wrapping the submitit Job.
        """
        if self._executor is None:
            raise RuntimeError("SlurmTaskRunner must be used as context manager")

        import functools

        env = _sanitize_env(dict(os.environ))
        # Supports functions defined in notebooks and tests.
        pickled_fn = cloudpickle_fn(fn)
        callable_fn = functools.partial(
            _run_fn_in_slurm, pickled_fn=pickled_fn, parameters=parameters, env=env
        )
        job = self._executor.submit(callable_fn)

        if self.compute_backend == ComputeBackend.SLURM:
            self._log_submission(
                job.job_id, num_jobs=1, task_name=getattr(fn, "__name__", None)
            )

        max_poll = self.max_poll_time
        if max_poll is None:
            max_poll = _default_max_poll(self._job_args.timeout_min)

        future = SlurmFuture(
            job=job,
            poll_interval=self.poll_interval,
            max_poll_time=max_poll,
        )

        return future

    def _make_array_poller(self, array_job_id: str) -> ArrayStatePoller | None:
        """Build a shared, batched state poller for one SLURM job array.

        Returns None for local execution (no `sacct`) or when the array id is
        empty, in which case array futures fall back to submitit's per-job state.
        """
        if self.compute_backend != ComputeBackend.SLURM or not array_job_id:
            return None
        return ArrayStatePoller(
            array_job_id=array_job_id, poll_interval=self.poll_interval
        )

    def _validate_iterable_lengths(self, iterable_params: dict[str, list]) -> int:
        """Ensure all iterables have same length and are non-empty."""
        if not iterable_params:
            raise ValueError("No iterable parameters found for map()")

        lengths = {k: len(v) for k, v in iterable_params.items()}

        if any(length == 0 for length in lengths.values()):
            raise ValueError(
                f"Empty iterable passed to map(). Lengths: {lengths}. "
                f"If intentional, guard with: `if items: runner.map(fn, items=items)`"
            )

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"Iterable parameters have mismatched lengths: {lengths}")

        return unique_lengths.pop()

    def _get_cluster_max_array_size(self) -> int:
        """Get the maximum job array size."""
        if self.max_array_size is not None:
            return self.max_array_size

        if self._cached_max_array_size is not None:
            return self._cached_max_array_size

        if self.compute_backend == ComputeBackend.LOCAL:
            self._cached_max_array_size = DEFAULT_MAX_ARRAY_SIZE
            return self._cached_max_array_size

        try:
            result = subprocess.run(
                ["scontrol", "show", "config"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                match = re.search(r"MaxArraySize\s*=\s*(\d+)", result.stdout)
                if match:
                    self._cached_max_array_size = int(match.group(1))
                    self.logger.debug(
                        f"Detected cluster MaxArraySize: {self._cached_max_array_size}"
                    )
                    return self._cached_max_array_size
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            self.logger.warning(f"Could not detect MaxArraySize: {e}")

        self._cached_max_array_size = DEFAULT_MAX_ARRAY_SIZE
        self.logger.warning(
            f"Using fallback MaxArraySize: {self._cached_max_array_size}"
        )
        return self._cached_max_array_size

    def _build_array_callable(
        self,
        pickled_fn: bytes,
        index: int,
        iterable_params: dict[str, list],
        static_params: dict[str, Any],
        env: dict,
    ) -> Callable:
        """Build a callable for one array task."""
        params = {key: values[index] for key, values in iterable_params.items()}
        params.update(static_params)

        import functools

        return functools.partial(
            _run_fn_in_slurm, pickled_fn=pickled_fn, parameters=params, env=env
        )

    def _batch_items(self, items: list[Any]) -> list[list[Any]]:
        """Group items into batches of items_per_job."""
        return [
            items[i : i + self.items_per_job]
            for i in range(0, len(items), self.items_per_job)
        ]

    def _build_batch_callable(
        self,
        pickled_fn: bytes,
        batch: list[Any],
        param_name: str,
        static_params: dict[str, Any],
        env: dict,
    ) -> Callable:
        """Build a callable that processes a batch of items."""
        import functools

        return functools.partial(
            _run_batch_in_slurm,
            pickled_fn=pickled_fn,
            batch_items=batch,
            param_name=param_name,
            static_params=static_params,
            env=env,
        )

    def _submit_batched_job_array(
        self,
        fn: Callable,
        items: list[Any],
        param_name: str,
        static_params: dict[str, Any],
    ) -> list[SlurmBatchedItemFuture]:
        """Submit batched items as a SLURM job array."""
        batches = self._batch_items(items)
        num_batches = len(batches)
        env = _sanitize_env(dict(os.environ))
        pickled_fn = cloudpickle_fn(fn)

        wrapped_calls = [
            self._build_batch_callable(
                pickled_fn=pickled_fn,
                batch=batch,
                param_name=param_name,
                static_params=static_params,
                env=env,
            )
            for batch in batches
        ]

        max_array_size = self._get_cluster_max_array_size()

        num_items = len(items)
        fn_name = getattr(fn, "__name__", None)

        if num_batches <= max_array_size:
            job_futures = self._submit_array_chunk(
                wrapped_calls, num_batches, task_name=fn_name, num_rows=num_items
            )
        else:
            num_chunks = (num_batches + max_array_size - 1) // max_array_size
            self.logger.info(
                f"Batch count {num_batches} exceeds cluster limit {max_array_size}, "
                f"splitting into {num_chunks} array submissions"
            )

            job_futures: list[SlurmArrayFuture] = []
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * max_array_size
                chunk_end = min(chunk_start + max_array_size, num_batches)
                chunk_length = chunk_end - chunk_start

                chunk_futures = self._submit_array_chunk(
                    wrapped_calls[chunk_start:chunk_end],
                    chunk_length,
                    task_name=fn_name,
                    num_rows=num_items,
                )
                job_futures.extend(chunk_futures)

        item_futures: list[SlurmBatchedItemFuture] = []
        global_item_index = 0

        for job_idx, batch in enumerate(batches):
            job_future = job_futures[job_idx]
            for item_idx_in_job in range(len(batch)):
                item_futures.append(
                    SlurmBatchedItemFuture(
                        slurm_job_future=job_future,
                        item_index_in_job=item_idx_in_job,
                        global_item_index=global_item_index,
                    )
                )
                global_item_index += 1

        return item_futures

    def _submit_array_chunk(
        self,
        wrapped_calls: list[Callable],
        array_size: int,
        task_name: str = None,
        num_rows: int | None = None,
    ) -> list[SlurmArrayFuture]:
        """Submit a chunk of callables as a single SLURM job array."""
        jobs: list[submitit.Job[Any]] = []
        with self._executor.batch():
            for wrapped_call in wrapped_calls:
                job = self._executor.submit(wrapped_call)
                jobs.append(job)

        array_job_id = jobs[0].job_id.split("_")[0] if jobs else ""

        if self.compute_backend == ComputeBackend.SLURM and array_job_id:
            self._log_submission(
                array_job_id,
                num_jobs=array_size,
                num_rows=num_rows,
                task_name=task_name,
            )

        max_poll = self.max_poll_time or _default_max_poll(self._job_args.timeout_min)

        # Submitit turns a one-element ``executor.batch()`` into a plain SLURM
        # job rather than an array.  Let its normal job watcher handle that
        # case; the batched array poller expects ``<job_id>_<task_index>`` rows
        # and some clusters expose only the parent ``<job_id>`` row.
        poller = self._make_array_poller(array_job_id) if array_size > 1 else None
        futures = [
            SlurmArrayFuture(
                job=jobs[i],
                poll_interval=self.poll_interval,
                max_poll_time=max_poll,
                array_job_id=array_job_id,
                array_task_index=i,
                array_size=array_size,
                fail_on_error=self.fail_on_error,
                poller=poller,
            )
            for i in range(array_size)
        ]

        return futures

    def _submit_single_job_array(
        self,
        pickled_fn: bytes,
        fn_name: str,
        iterable_params: dict[str, list],
        static_params: dict[str, Any],
        map_length: int,
        env: dict,
    ) -> list[SlurmArrayFuture]:
        """Submit tasks as a single SLURM job array."""
        wrapped_calls = [
            self._build_array_callable(
                pickled_fn, i, iterable_params, static_params, env
            )
            for i in range(map_length)
        ]

        jobs: list[submitit.Job[Any]] = []
        with self._executor.batch():
            for wrapped_call in wrapped_calls:
                job = self._executor.submit(wrapped_call)
                jobs.append(job)

        array_job_id = jobs[0].job_id.split("_")[0] if jobs else ""

        if self.compute_backend == ComputeBackend.SLURM and array_job_id:
            self._log_submission(array_job_id, num_jobs=map_length, task_name=fn_name)

        max_poll = self.max_poll_time or _default_max_poll(self._job_args.timeout_min)

        poller = self._make_array_poller(array_job_id) if map_length > 1 else None
        futures = [
            SlurmArrayFuture(
                job=jobs[i],
                poll_interval=self.poll_interval,
                max_poll_time=max_poll,
                array_job_id=array_job_id,
                array_task_index=i,
                array_size=map_length,
                fail_on_error=self.fail_on_error,
                poller=poller,
            )
            for i in range(map_length)
        ]

        return futures

    def _submit_job_array(
        self,
        fn: Callable,
        iterable_params: dict[str, list],
        static_params: dict[str, Any],
        map_length: int,
    ) -> list[SlurmArrayFuture]:
        """Submit tasks as job array(s), chunking if necessary."""
        env = _sanitize_env(dict(os.environ))
        pickled_fn = cloudpickle_fn(fn)
        fn_name = getattr(fn, "__name__", None)

        max_array_size = self._get_cluster_max_array_size()

        if map_length <= max_array_size:
            return self._submit_single_job_array(
                pickled_fn,
                fn_name,
                iterable_params,
                static_params,
                map_length,
                env,
            )

        num_chunks = (map_length + max_array_size - 1) // max_array_size
        self.logger.info(
            f"Array size {map_length} exceeds cluster limit {max_array_size}, "
            f"splitting into {num_chunks} chunks"
        )

        all_futures: list[SlurmArrayFuture] = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * max_array_size
            chunk_end = min(chunk_start + max_array_size, map_length)
            chunk_length = chunk_end - chunk_start

            chunk_iterable_params = {
                k: v[chunk_start:chunk_end] for k, v in iterable_params.items()
            }

            chunk_futures = self._submit_single_job_array(
                pickled_fn,
                fn_name,
                chunk_iterable_params,
                static_params,
                chunk_length,
                env,
            )
            all_futures.extend(chunk_futures)

        return all_futures

    def map(
        self,
        fn: Callable,
        **parameters: Any,
    ) -> list[SlurmArrayFuture | SlurmBatchedItemFuture]:
        """Submit multiple function calls as a SLURM job array.

        Args:
            fn: The function to execute.
            **parameters: Dict where iterable values are mapped over.

        Returns:
            List of futures, one per item.
        """
        if self._executor is None:
            raise RuntimeError("SlurmTaskRunner must be used as context manager")

        iterable_params, static_params = partition_parameters(parameters)
        map_length = self._validate_iterable_lengths(iterable_params)

        if (
            self.check_duplicates
            and self.compute_backend == ComputeBackend.SLURM
            and self.job_name
        ):
            existing_jobs = _check_existing_slurm_jobs(self.job_name)
            if existing_jobs:
                pending_running = [
                    j
                    for j in existing_jobs
                    if j["state"] in ("PENDING", "RUNNING", "CONFIGURING")
                ]
                if pending_running:
                    job_ids = [j["job_id"] for j in pending_running]
                    total_tasks = sum(
                        int(j["array_task_count"])
                        if j["array_task_count"].isdigit()
                        else 1
                        for j in pending_running
                    )
                    console.print(
                        f"WARNING: Found {len(pending_running)} existing "
                        f"PENDING/RUNNING job(s) named '{self.job_name}' with ~{total_tasks} tasks: "
                        f"{job_ids[:5]}{'...' if len(job_ids) > 5 else ''}. "
                        f"Submitting {map_length} additional tasks. "
                        f"Use check_duplicates=False to suppress this warning."
                    )

        if self.items_per_job > 1:
            if len(iterable_params) > 1:
                raise ValueError(
                    f"Multi-argument map() is not supported with items_per_job > 1. "
                    f"Found {len(iterable_params)} iterable parameters: {list(iterable_params.keys())}. "
                    f"Use items_per_job=1 for multi-arg map, or pre-zip arguments into single iterable."
                )

            param_name = list(iterable_params.keys())[0]
            items = iterable_params[param_name]

            return self._submit_batched_job_array(fn, items, param_name, static_params)

        return self._submit_job_array(fn, iterable_params, static_params, map_length)
