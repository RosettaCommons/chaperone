"""Orchestration layer for Chaperone.

Provides local and Slurm task runners.
"""

from chaperone.orchestration.annotations import unmapped
from chaperone.orchestration.base_runner import TaskRunner
from chaperone.orchestration.futures import TaskFuture, TaskState
from chaperone.orchestration.slurm_futures import (
    SlurmArrayFuture,
    SlurmBatchedItemFuture,
    SlurmFuture,
    SlurmJobCancelled,
    SlurmJobFailed,
)
from chaperone.orchestration.job_args import JobArgs
from chaperone.orchestration.slurm_taskrunner import SlurmTaskRunner
from chaperone.orchestration.thread_runner import ThreadPoolRunner
from chaperone.execution_tracking import detect_cuda_gpus
from chaperone.orchestration.gpu_local_taskrunner import GPULocalTaskRunner

__all__ = [
    "detect_cuda_gpus",
    "GPULocalTaskRunner",
    "JobArgs",
    "SlurmArrayFuture",
    "SlurmBatchedItemFuture",
    "SlurmFuture",
    "SlurmJobCancelled",
    "SlurmJobFailed",
    "SlurmTaskRunner",
    "TaskFuture",
    "TaskRunner",
    "TaskState",
    "ThreadPoolRunner",
    "unmapped",
]
