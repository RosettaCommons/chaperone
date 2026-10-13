"""Future wrappers for SLURM jobs.

These classes adapt submitit.Job objects to our TaskFuture protocol,
enabling seamless integration with the orchestration layer.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chaperone.orchestration.futures import TaskFuture, TaskState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import submitit


# States that mean the array task has left the queue successfully or otherwise.
_TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "NODE_FAIL",
        "TIMEOUT",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
    }
)


class ArrayStatePoller:
    """Batched state poller shared across the tasks of a single SLURM array.

    Instead of every :class:`SlurmArrayFuture` shelling out to ``sacct``/``squeue``
    independently (O(N) subprocess calls per poll cycle for an N-task array), all
    futures of one array share a single poller. One ``sacct -j <array_job_id>``
    call per poll cycle fetches every task's state, which is then distributed to
    the individual futures from an in-memory cache.

    The cache is throttled by ``poll_interval``: back-to-back state reads from
    different futures within one interval reuse the same ``sacct`` output, so a
    serial ``[f.result() for f in futures]`` gather over a 200-task array issues
    O(1) subprocess calls per poll cycle rather than O(200).
    """

    def __init__(self, array_job_id: str, poll_interval: float):
        self._array_job_id = array_job_id
        self._poll_interval = max(0.0, poll_interval)
        # Maps "<array_job_id>_<task_index>" -> SLURM state string.
        self._states: dict[str, str] = {}
        self._last_poll: float = float("-inf")
        self._num_calls = 0

    @property
    def num_calls(self) -> int:
        """Number of ``sacct`` subprocess calls issued (for testing/inspection)."""
        return self._num_calls

    def _run_sacct(self) -> dict[str, str]:
        """Issue ONE ``sacct`` call for the whole array and parse per-task states."""
        states: dict[str, str] = {}
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "-j",
                    self._array_job_id,
                    "--format=JobID,State",
                    "--parsable2",
                    "--noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"sacct poll for array {self._array_job_id} failed: {e}")
            return states

        if result.returncode != 0:
            logger.debug(
                f"sacct poll for array {self._array_job_id} returned "
                f"{result.returncode}: {result.stderr.strip()}"
            )
            return states

        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            job_id, state = parts[0].strip(), parts[1].strip()
            # Skip step rows like "12345_0.batch" / "12345_0.extern".
            if not job_id or "." in job_id:
                continue
            # Normalise states like "CANCELLED by 1234" -> "CANCELLED".
            states[job_id] = state.split()[0] if state else state
        return states

    def refresh(self, force: bool = False) -> None:
        """Poll ``sacct`` for the whole array if the throttle interval has elapsed."""
        now = time.time()
        if not force and (now - self._last_poll) < self._poll_interval:
            return
        self._num_calls += 1
        self._states.update(self._run_sacct())
        self._last_poll = now

    def get_state(self, task_index: int, force: bool = False) -> str:
        """Return the SLURM state string for one array task ("" if unknown)."""
        self.refresh(force=force)
        return self._states.get(f"{self._array_job_id}_{task_index}", "")

    def get_job_state(self, force: bool = False) -> str:
        """Return the parent job state (used when Submitit batches one job)."""
        self.refresh(force=force)
        return self._states.get(self._array_job_id, "")

    def is_done(self, task_index: int, force: bool = False) -> bool:
        """Return whether one array task has reached a terminal state."""
        state = self.get_state(task_index, force=force)
        return state in _TERMINAL_STATES


# A job stuck PENDING for one of these reasons will never start on its own.
# SLURM spells every hold variant with "held" in the reason string
# ("JobHeldUser", "JobHeldAdmin", "... requeued held"), so a substring test
# covers them without enumerating cluster-specific wording.
_HELD_REASON_MARKER = "held"

# Don't probe squeue for the first minute: a freshly submitted job can report
# a transient reason before the scheduler settles. After that, re-check at the
# same cadence so a job held mid-queue is still caught.
HELD_CHECK_DELAY_S = 60.0
HELD_CHECK_INTERVAL_S = 60.0


def _detect_held_reason(job_id: str) -> str | None:
    """Return the SLURM hold reason for ``job_id``, or None if it isn't held.

    A held job sits PENDING indefinitely, so ``wait()`` would otherwise burn
    the full ``max_poll_time`` on a job the scheduler will never start.
    Returns None on any squeue failure — an unreadable queue is not evidence
    of a hold, and a false positive would kill a healthy run.
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T %r"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"squeue hold-check for {job_id} failed: {e}")
        return None

    if result.returncode != 0:
        return None

    lines = (result.stdout or "").strip().splitlines()
    if not lines:
        return None

    # "<STATE> <REASON>" — the reason itself may contain spaces, e.g.
    # "user env retrieval failed requeued held".
    parts = lines[0].strip().split(None, 1)
    if len(parts) < 2:
        return None

    state, reason = parts[0].strip(), parts[1].strip()
    if state != "PENDING":
        return None
    if _HELD_REASON_MARKER not in reason.lower():
        return None
    return reason


class SlurmJobFailed(Exception):
    """Raised when a SLURM job fails."""

    pass


class SlurmJobCancelled(SlurmJobFailed):
    """Raised when a SLURM job is cancelled."""

    pass


class SlurmJobHeld(SlurmJobFailed):
    """Raised when a SLURM job is held and will never start on its own.

    Subclasses :class:`SlurmJobFailed` so existing callers that catch job
    failure keep working; catch it specifically to distinguish "the scheduler
    is holding this" from "the job ran and failed".
    """

    pass


class SlurmFuture(TaskFuture):
    """Wraps submitit.Job to implement the TaskFuture protocol.

    This adapter translates between submitit's Job API and our Future API.
    """

    TERMINAL_FAILURE_STATES = frozenset(
        {"FAILED", "NODE_FAIL", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"}
    )

    def __init__(
        self,
        job: "submitit.Job[Any]",
        poll_interval: float,
        max_poll_time: float,
    ):
        self._job = job
        self._poll_interval = poll_interval
        self._max_poll_time = max_poll_time
        self._result_cache: Any = None
        self._result_retrieved = False
        self._done = False
        self._is_failed = False
        self._failure_message: str = ""

    @property
    def slurm_job_id(self) -> str:
        """SLURM job ID for debugging/monitoring."""
        return self._job.job_id

    @property
    def is_done(self) -> bool:
        """Whether the future has completed (success or failure)."""
        return self._done

    def _poll_state(self) -> str:
        """Return the raw SLURM state string for this future.

        Default implementation defers to submitit's (shared, throttled) watcher.
        Subclasses that share a batched poller override this.
        """
        return self._job.state

    def _poll_done(self) -> bool:
        """Return whether the underlying SLURM job has reached a terminal state.

        Default implementation defers to submitit, OR-ed with an on-disk result
        check. Subclasses that share a batched poller override this.
        """
        return self._job.done() or self._result_ready()

    def _result_ready(self) -> bool:
        """Whether submitit has written this job's result pickle to disk.

        ``submitit``'s ``job.done()`` (and a transient ``sacct`` gap on the
        batched poller) can fail to flip to a terminal state when a job leaves
        the queue between polls — leaving ``wait()`` to poll for the full
        ``2x time_limit`` on a job that has actually finished. The result pickle
        is an authoritative, source-independent terminal signal: submitit writes
        it (atomically) once the wrapped command exits, success or failure, so
        its presence means the job is done and ``result()`` can read it.

        A directory read precedes ``exists()`` to invalidate stale negative
        dentries on shared NFS. Without it, a submit host that checked before
        the compute node's atomic rename can keep reporting the result absent
        for the full attribute-cache TTL even though the job has completed.
        """
        try:
            path = Path(self._job.paths.result_pickle)
            os.listdir(path.parent)
            return path.exists()
        except (TypeError, ValueError, OSError, AttributeError):
            return False

    @property
    def state(self) -> TaskState:
        """Return TaskState based on SLURM job state."""
        if self._result_retrieved and not self._is_failed:
            return TaskState.COMPLETED

        if self._is_failed:
            return TaskState.FAILED

        slurm_state = self._poll_state()
        if slurm_state == "COMPLETED":
            return TaskState.COMPLETED
        elif slurm_state in self.TERMINAL_FAILURE_STATES:
            return TaskState.FAILED
        elif slurm_state == "RUNNING":
            return TaskState.RUNNING
        return TaskState.PENDING

    def wait(self, timeout: float | None = None) -> None:
        """Block until job completes or timeout.

        Args:
            timeout: Maximum seconds to wait after the job starts running. Uses
                max_poll_time if None. Queue wait time is not counted — the clock
                starts only once the job leaves PENDING state.

        Raises:
            TimeoutError: If job doesn't complete within timeout after starting.
            SlurmJobFailed: If job enters a terminal failure state.
            SlurmJobHeld: If the job is held and will never start.
            KeyboardInterrupt: If interrupted by user (Ctrl+C).
        """
        effective_timeout = timeout or self._max_poll_time
        # Only start the timeout clock once the job is actually running.
        # Queue wait time is excluded so jobs that sit pending for hours
        # (e.g. submitted with --nice behind many other jobs) don't time out.
        run_start: float | None = None
        # Queue time is not bounded by effective_timeout, so a held job would
        # otherwise sit here forever. Probe squeue periodically to catch it.
        wait_start = time.time()
        last_held_check = wait_start

        try:
            while not self._poll_done():
                slurm_state = self._poll_state()

                if run_start is None and slurm_state not in ("PENDING", ""):
                    run_start = time.time()

                if effective_timeout and run_start is not None:
                    elapsed = time.time() - run_start
                    if elapsed > effective_timeout:
                        raise TimeoutError(
                            f"Job {self.slurm_job_id} did not complete within {effective_timeout:.0f}s"
                        )

                if slurm_state in self.TERMINAL_FAILURE_STATES:
                    self._done = True
                    self._is_failed = True
                    self._failure_message = (
                        f"Job {self.slurm_job_id}: {slurm_state}" + self._tail_logs()
                    )
                    if "CANCELLED" in slurm_state:
                        raise SlurmJobCancelled(self._failure_message)
                    raise SlurmJobFailed(self._failure_message)

                now = time.time()
                if (now - wait_start) >= HELD_CHECK_DELAY_S and (
                    now - last_held_check
                ) >= HELD_CHECK_INTERVAL_S:
                    last_held_check = now
                    held_reason = _detect_held_reason(self.slurm_job_id)
                    if held_reason:
                        self._done = True
                        self._is_failed = True
                        self._failure_message = (
                            f"Job {self.slurm_job_id} is held ({held_reason}) "
                            f"and will not start on its own"
                        )
                        raise SlurmJobHeld(self._failure_message)

                # Sleep in small increments to be responsive to interrupts
                sleep_remaining = self._poll_interval
                while sleep_remaining > 0:
                    time.sleep(min(1.0, sleep_remaining))
                    sleep_remaining -= 1.0

            # Check final state after loop exits
            slurm_state = self._poll_state()
            if slurm_state in self.TERMINAL_FAILURE_STATES:
                self._done = True
                self._is_failed = True
                self._failure_message = (
                    f"Job {self.slurm_job_id}: {slurm_state}" + self._tail_logs()
                )
                if "CANCELLED" in slurm_state:
                    raise SlurmJobCancelled(self._failure_message)
                raise SlurmJobFailed(self._failure_message)

        except KeyboardInterrupt:
            logger.warning(f"Interrupted while waiting for job {self.slurm_job_id}")
            self._done = True
            self._is_failed = True
            self._failure_message = f"Job {self.slurm_job_id}: interrupted by user"
            raise
        self._done = True

    def result(
        self,
        timeout: float | None = None,
        raise_on_failure: bool = True,
    ) -> Any:
        """Wait for job completion and return result.

        Args:
            timeout: Maximum seconds to wait.
            raise_on_failure: Whether to raise on job failure.

        Returns:
            The task return value.

        Raises:
            SlurmJobFailed: If job failed and raise_on_failure is True.
        """
        from submitit.core.utils import FailedJobError, UncompletedJobError

        if self._result_retrieved:
            return self._result_cache

        self.wait(timeout)

        try:
            self._result_cache = self._job.result()
            self._result_retrieved = True
            return self._result_cache

        except FailedJobError as e:
            self._is_failed = True
            self._failure_message = (
                f"Job {self.slurm_job_id} failed: {e}" + self._tail_logs()
            )
            if raise_on_failure:
                raise SlurmJobFailed(self._failure_message) from e
            return None
        except UncompletedJobError as e:
            error_str = str(e)
            self._is_failed = True
            if "CANCELLED" in error_str:
                self._failure_message = f"Job {self.slurm_job_id} was cancelled"
                if raise_on_failure:
                    raise SlurmJobCancelled(self._failure_message) from e
                return None
            stderr = self._job.stderr()
            self._failure_message = (
                f"Job {self.slurm_job_id} produced no output: {e}\nstderr: {stderr}"
            )
            if raise_on_failure:
                raise SlurmJobFailed(self._failure_message) from e
            return None

    def logs(self) -> tuple[str, str]:
        """Return stdout and stderr from SLURM job logs."""
        return self._job.stdout(), self._job.stderr()

    def _tail_logs(self, n_lines: int = 30) -> str:
        """Return last n_lines of stderr (and stdout if stderr is empty) for error context."""
        try:
            stderr = self._job.stderr()
            stdout = self._job.stdout()
            tail = ""
            label = ""
            if stderr and stderr.strip():
                lines = stderr.strip().splitlines()
                tail = "\n".join(lines[-n_lines:])
                label = "stderr"
            elif stdout and stdout.strip():
                lines = stdout.strip().splitlines()
                tail = "\n".join(lines[-n_lines:])
                label = "stdout"
            if tail:
                return f"\n--- Last {min(n_lines, len(lines))} lines of {label} ---\n{tail}"
        except Exception:
            pass  # Log reading can fail if files don't exist yet
        return ""


class SlurmArrayFuture(SlurmFuture):
    """Future for a task within a SLURM job array.

    Extends SlurmFuture with array-specific metadata.
    """

    def __init__(
        self,
        job: "submitit.Job[Any]",
        poll_interval: float,
        max_poll_time: float,
        array_job_id: str,
        array_task_index: int,
        array_size: int,
        fail_on_error: bool = True,
        poller: ArrayStatePoller | None = None,
    ):
        super().__init__(job, poll_interval, max_poll_time)
        self._array_job_id = array_job_id
        self._array_task_index = array_task_index
        self._array_size = array_size
        self._fail_on_error = fail_on_error
        # Shared, batched state poller for this array. When set, all futures of
        # the array consult it so one `sacct` call serves every task per poll
        # cycle instead of each future shelling out independently.
        self._poller = poller

    def _poll_state(self) -> str:
        """Read this task's state from the shared array poller when available."""
        if self._poller is None:
            return super()._poll_state()
        state = self._poller.get_state(self._array_task_index)
        # Submitit optimizes a one-element ``executor.batch()`` into a plain
        # SLURM job.  Its sacct row is ``<job_id>`` rather than
        # ``<job_id>_0``, so the array-style lookup above cannot see it.
        if not state and self._array_size == 1:
            return self._poller.get_job_state()
        return state

    def _poll_done(self) -> bool:
        """Read terminal status from the shared array poller when available.

        Falls back to the on-disk result pickle so an intermittent ``sacct``
        gap (poller returns "" for a task that has actually finished) can't pin
        ``wait()`` for the full ``2x time_limit``.
        """
        if self._poller is None:
            return super()._poll_done()
        return self._poll_state() in _TERMINAL_STATES or self._result_ready()

    @property
    def array_job_id(self) -> str:
        return self._array_job_id

    @property
    def array_task_index(self) -> int:
        return self._array_task_index

    @property
    def array_size(self) -> int:
        return self._array_size

    @property
    def slurm_job_id(self) -> str:
        return f"{self._array_job_id}_{self._array_task_index}"

    def __repr__(self) -> str:
        return (
            f"SlurmArrayFuture("
            f"job_id={self.slurm_job_id}, "
            f"index={self._array_task_index}/{self._array_size}, "
            f"state={self.state.value})"
        )

    def cancel(self) -> bool:
        """Cancel the entire job array (all tasks)."""
        try:
            subprocess.run(
                ["scancel", self._array_job_id],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def cancel_task(self) -> bool:
        """Cancel only this specific array task."""
        try:
            subprocess.run(
                ["scancel", f"{self._array_job_id}_{self._array_task_index}"],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def result(
        self,
        timeout: float | None = None,
        raise_on_failure: bool | None = None,
    ) -> Any:
        if raise_on_failure is None:
            raise_on_failure = self._fail_on_error
        return super().result(timeout=timeout, raise_on_failure=raise_on_failure)


class SlurmBatchedItemFuture(TaskFuture):
    """Future representing a single item within a batched SLURM job.

    Provides one-future-per-item semantics while sharing SLURM job execution.
    """

    def __init__(
        self,
        slurm_job_future: SlurmArrayFuture,
        item_index_in_job: int,
        global_item_index: int,
    ):
        self._slurm_job_future = slurm_job_future
        self._item_index_in_job = item_index_in_job
        self._global_item_index = global_item_index
        self._result_cache: Any = None
        self._result_retrieved = False

    @property
    def slurm_job_future(self) -> SlurmArrayFuture:
        return self._slurm_job_future

    @property
    def item_index_in_job(self) -> int:
        return self._item_index_in_job

    @property
    def global_item_index(self) -> int:
        return self._global_item_index

    @property
    def slurm_job_index(self) -> int:
        return self._slurm_job_future.array_task_index

    @property
    def slurm_job_id(self) -> str:
        return self._slurm_job_future.slurm_job_id

    @property
    def state(self) -> TaskState:
        return self._slurm_job_future.state

    def wait(self, timeout: float | None = None) -> None:
        self._slurm_job_future.wait(timeout)

    def result(
        self,
        timeout: float | None = None,
        raise_on_failure: bool = True,
    ) -> Any:
        if self._result_retrieved:
            return self._result_cache

        job_result = self._slurm_job_future.result(
            timeout=timeout, raise_on_failure=raise_on_failure
        )

        if job_result is None:
            return None

        self._result_cache = job_result[self._item_index_in_job]
        self._result_retrieved = True
        return self._result_cache

    def __repr__(self) -> str:
        return (
            f"SlurmBatchedItemFuture("
            f"global_index={self._global_item_index}, "
            f"job_index={self.slurm_job_index}, "
            f"item_in_job={self._item_index_in_job}, "
            f"slurm_job={self.slurm_job_id}, "
            f"state={self.state.value})"
        )
