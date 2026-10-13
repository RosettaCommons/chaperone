"""Orchestration layer for pipelines.

Provides local and Slurm task runners.
"""

from pipelines.orchestration.annotations import unmapped
from pipelines.orchestration.base_runner import TaskRunner
from pipelines.orchestration.futures import TaskFuture, TaskState
from pipelines.orchestration.slurm_futures import (
    SlurmArrayFuture,
    SlurmBatchedItemFuture,
    SlurmFuture,
    SlurmJobCancelled,
    SlurmJobFailed,
)
from pipelines.orchestration.job_args import JobArgs
from pipelines.orchestration.slurm_taskrunner import SlurmTaskRunner
from pipelines.orchestration.thread_runner import ThreadPoolRunner
from pipelines.execution_tracking import detect_cuda_gpus
from pipelines.orchestration.gpu_local_taskrunner import GPULocalTaskRunner

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
