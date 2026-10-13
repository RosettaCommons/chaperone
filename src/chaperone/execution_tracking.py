"""Durable execution accounting written by the executing worker process.

Execution records are stored as one JSON file per attempt under a step-local
``.execution`` directory. This keeps accounting independent from the submitting
process and avoids relying on post-submission head-node aggregation.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _cuda_device_present() -> bool:
    """Is a CUDA device actually attached, as opposed to merely advertised?

    CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS describe an intended allocation and
    survive into environments that never received one — an inherited value, or
    a CPU-only allocation launched from a GPU context. Believing them there
    routes GPU work to a host with no GPU, where it fails well downstream of
    the wrong decision.
    """
    if any(Path("/dev").glob("nvidia[0-9]*")):
        return True
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def detect_cuda_gpus() -> list[int]:
    """Detect CUDA GPUs visible to the current process.

    Returns an empty list when no device is attached, whatever the environment
    claims, so callers fall through to cluster execution instead of trying to
    run locally without a GPU.
    """
    if not _cuda_device_present():
        return []

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        try:
            if "GPU-" in cuda_visible:
                gpu_count = len(cuda_visible.split(","))
                return list(range(gpu_count))
            return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        except ValueError:
            pass

    slurm_gpus = os.environ.get("SLURM_JOB_GPUS")
    if slurm_gpus:
        try:
            return [int(x.strip()) for x in slurm_gpus.split(",") if x.strip()]
        except ValueError:
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [
                int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()
            ]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
        pass

    return []


@dataclass
class ExecutionSummary:
    """Aggregated wall-time accounting for a step."""

    records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.records = int(self.records)
        self.successful_records = int(self.successful_records)
        self.failed_records = int(self.failed_records)
        self.wall_seconds = float(self.wall_seconds)
        self.gpu_seconds = float(self.gpu_seconds)

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0

    def add(self, other: "ExecutionSummary") -> None:
        self.records += other.records
        self.successful_records += other.successful_records
        self.failed_records += other.failed_records
        self.wall_seconds += other.wall_seconds
        self.gpu_seconds += other.gpu_seconds


@dataclass
class ExecutionRecord:
    """Single execution attempt for a block step or chunk."""

    block_name: str
    sandbox: str
    source: str
    status: str
    started_at: str
    ended_at: str
    wall_seconds: float
    allocated_gpus: int
    gpu_seconds: float
    attempt: int
    hostname: str
    pid: int
    chunk_id: int | None = None
    command: str | None = None
    backend: str | None = None
    job_id: str | None = None
    gpu_ids: list[int] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        self.wall_seconds = float(self.wall_seconds)
        self.allocated_gpus = int(self.allocated_gpus)
        self.gpu_seconds = float(self.gpu_seconds)
        self.attempt = int(self.attempt)
        self.pid = int(self.pid)
        if self.chunk_id is not None:
            self.chunk_id = int(self.chunk_id)
        if self.gpu_ids is not None:
            self.gpu_ids = [int(gpu_id) for gpu_id in self.gpu_ids]


def summarize_execution_records(records: list[ExecutionRecord]) -> ExecutionSummary:
    """Aggregate execution records into wall-time totals.

    Counts terminal records only: ``status="succeeded"`` rolls into
    ``successful_records`` and ``status="failed"`` into ``failed_records``.
    In-flight records (``status="running"``) are skipped so a summary
    snapshot taken mid-step doesn't double-count attempts that haven't
    finished yet.
    """
    summary = ExecutionSummary()
    for record in records:
        if record.status == "running":
            continue
        summary.records += 1
        if record.status == "succeeded":
            summary.successful_records += 1
        else:
            summary.failed_records += 1
        summary.wall_seconds += record.wall_seconds
        summary.gpu_seconds += record.gpu_seconds
    return summary


def load_execution_records(step_dir: str | Path) -> list[ExecutionRecord]:
    """Load all durable execution records for a step directory."""
    execution_dir = Path(step_dir) / ".execution"
    if not execution_dir.exists():
        return []

    records: list[ExecutionRecord] = []
    for path in sorted(execution_dir.glob("*.json")):
        if path.name.startswith(".tmp_"):
            continue
        try:
            with open(path) as f:
                records.append(ExecutionRecord(**json.load(f)))
        except (OSError, json.JSONDecodeError):
            # Path.glob() doesn't hide dotfiles, so it can pick up another
            # worker's in-flight .tmp_*.json right before _write_execution_record's
            # os.replace() renames it away underneath us.
            continue
    return records


def summarize_execution_dir(step_dir: str | Path) -> ExecutionSummary | None:
    """Summarize all execution records for a step directory."""
    records = load_execution_records(step_dir)
    if not records:
        return None
    return summarize_execution_records(records)


def infer_allocated_gpus(env: dict[str, str] | None = None) -> int:
    """Infer allocated GPU count from process environment only."""
    env = env or os.environ

    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        visible = [part.strip() for part in cuda_visible.split(",") if part.strip()]
        return len(visible)

    for key in ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS"):
        raw = env.get(key, "").strip()
        if not raw:
            continue
        if raw.isdigit():
            return int(raw)
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if parts:
            return len(parts)
    return 0


def next_execution_attempt(step_dir: str | Path, sandbox_dir: str | Path) -> int:
    """Return the next execution attempt number for a specific sandbox."""
    execution_dir = Path(step_dir) / ".execution"
    if not execution_dir.exists():
        return 1

    sandbox = str(Path(sandbox_dir))
    attempts = 0
    for path in execution_dir.glob("*.json"):
        if path.name.startswith(".tmp_"):
            continue
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Path.glob() doesn't hide dotfiles, so it can pick up another
            # worker's in-flight .tmp_*.json right before _write_execution_record's
            # os.replace() renames it away underneath us.
            continue
        if payload.get("sandbox") == sandbox:
            attempts += 1
    return attempts + 1


def _execution_record_path(step_dir: Path, record: ExecutionRecord) -> Path:
    """Stable on-disk path for an attempt's record file.

    Path is derived from the attempt's ``started_at`` + ``pid`` so a
    single attempt always writes back to the same file, whether it's
    the initial ``status="running"`` record or the terminal one.
    """
    execution_dir = step_dir / ".execution"
    safe_ts = record.started_at.replace(":", "").replace("-", "")
    return execution_dir / f"{safe_ts}_{record.pid}_{record.attempt}.json"


def _write_execution_record(step_dir: Path, record: ExecutionRecord) -> None:
    """Atomic write (uuid-named tmp + os.replace) to the attempt's stable path.

    Avoid `tempfile.NamedTemporaryFile`: its RNG can collide across submitit
    workers launched in close succession on shared NFS, producing identical
    `.tmp_XXX.json` paths. When two workers race, one os.replace's the file
    out from under the other → FileNotFoundError on the second replace.

    Parallel chunks can also cache a negative dentry for the shared
    ``.execution`` directory on NFS mounts configured with nocto/acdirmin.
    Mutating the already-visible parent invalidates that cache; retrying then
    sees the directory another chunk created.
    """
    final_path = _execution_record_path(step_dir, record)
    execution_dir = final_path.parent
    for attempt in range(5):
        try:
            execution_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = execution_dir / f".tmp_{record.pid}_{uuid.uuid4().hex}.json"
            with open(tmp_path, "w") as tmp:
                json.dump(asdict(record), tmp, indent=2)
                tmp.write("\n")
            os.replace(tmp_path, final_path)
            return
        except (FileExistsError, FileNotFoundError):
            refresh = step_dir / f".nfs_execution_refresh_{record.pid}_{attempt}"
            try:
                refresh.touch()
            except OSError:
                pass
            time.sleep(0.25 * (attempt + 1))
    raise FileNotFoundError(
        f"Unable to write execution record after NFS retries: {final_path}"
    )


@contextmanager
def track_execution(
    *,
    step_dir: str | Path,
    sandbox_dir: str | Path,
    block_name: str,
    source: str,
    chunk_id: int | None = None,
    command: str | list[str] | None = None,
) -> Iterator[None]:
    """Record a durable execution attempt around an execute phase.

    Writes one atomic terminal record (``"succeeded"`` or ``"failed"``)
    after the execute phase. This retains cost and failure accounting without
    generating a second shared-filesystem write for live status telemetry.
    """
    step_dir = Path(step_dir)
    sandbox_dir = Path(sandbox_dir)
    attempt = next_execution_attempt(step_dir, sandbox_dir)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    # Inline backend inference (same logic as ChaperoneSettings.infer_backend)
    # to keep this module importable by lightweight standalone scripts
    # (e.g. runner_wait) that must NOT trigger chaperone/__init__.py — which
    # would pull in PipelineDataHandler/polars/the block registry.
    if os.environ.get("SLURM_JOB_ID"):
        backend = "slurm"
    else:
        _allocated_gpus_for_backend = infer_allocated_gpus()
        backend = "gpu_local" if _allocated_gpus_for_backend > 0 else "local"
    allocated_gpus = infer_allocated_gpus()
    gpu_ids = None
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        gpu_ids = [
            int(part) for part in cuda_visible.split(",") if part.strip().isdigit()
        ]
    hostname = socket.gethostname()
    pid = os.getpid()
    command_str = " ".join(command) if isinstance(command, list) else command
    job_id = os.environ.get("SLURM_JOB_ID")

    def _make_record(
        *,
        status: str,
        wall_seconds: float,
        ended_at: str,
        error: str | None,
    ) -> ExecutionRecord:
        return ExecutionRecord(
            block_name=block_name,
            sandbox=str(sandbox_dir),
            source=source,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            wall_seconds=wall_seconds,
            allocated_gpus=allocated_gpus,
            gpu_seconds=wall_seconds * allocated_gpus,
            attempt=attempt,
            hostname=hostname,
            pid=pid,
            chunk_id=chunk_id,
            command=command_str,
            backend=backend,
            job_id=job_id,
            gpu_ids=gpu_ids,
            error=error,
        )

    error: str | None = None
    status = "succeeded"
    try:
        yield
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        wall_seconds = time.perf_counter() - started
        _write_execution_record(
            step_dir,
            _make_record(
                status=status,
                wall_seconds=wall_seconds,
                ended_at=datetime.now(timezone.utc).isoformat(),
                error=error,
            ),
        )
