"""
Chunking utilities for parallel block execution.

This module handles splitting data into chunks for parallel execution,
either locally or on SLURM clusters. It provides:

- ChunkContext: Dataclass holding slice info for parallel chunk execution
- execute_chunks: Unified orchestrator for single or chained block execution
- execute_chunk_job: Unified job function using safe_call
"""

import inspect
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import cloudpickle
import submitit
import polars as pl
from tqdm import tqdm

from chaperone.block_config._log_tee import _TeeWriter
from chaperone.common import get_console, mkdir_p, progress_disabled
from chaperone.data import PipelineDataHandler

if TYPE_CHECKING:
    from chaperone.block_config.base import BlockFunction

logger = logging.getLogger(__name__)

console = get_console()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _write_terminal_report(path: Path, payload: dict) -> None:
    """Atomically publish the authoritative outcome of one worker chunk."""
    mkdir_p(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_write_parquet(handler: PipelineDataHandler, output_path: Path) -> None:
    mkdir_p(output_path.parent)
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.parquet")
    try:
        handler.write_parquet(temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _reset_chunk_sandbox_for_execution(chunk_dir: Path) -> None:
    """Give every real chunk execution an empty, attempt-local sandbox.

    Failed scientific commands commonly leave partial ``inputs/`` and
    ``outputs/`` trees. Reusing those trees is unsafe when a retry changes
    chunk size or membership: compilers discover stale outputs and then fail
    lineage matching against the new input slice. Durable chunk results live
    in ``output.parquet`` and are handled by the cache check before this reset.
    """
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    mkdir_p(chunk_dir)


def safe_call(block, **kwargs):
    """Call block.__call__ forwarding only the parameters it accepts."""
    sig = inspect.signature(block.__call__)
    accepted = set(sig.parameters)
    return block(**{k: v for k, v in kwargs.items() if k in accepted})


def _should_use_exec_finisher(
    *,
    block,
    is_last: bool,
    handler,
    recompile: bool,
    dedicated_submitit_worker: bool = False,
) -> bool:
    """Decide whether to take the low-memory ``os.execvp`` finisher path.

    Returns True only when ALL of these hold:

    * The dispatcher explicitly identified a dedicated submitit worker.
      A local driver or GPU pool may inherit both submitit environment
      variables from its enclosing allocation; replacing either process
      loses the rest of the pipeline or the pool's result channel.
    * ``is_last`` — execvp replaces the process, so any subsequent step in
      the chunk's chain would never run. Only safe on the last step.
    * not ``recompile`` — recompile skips Phase 2, so there is no long
      subprocess wait to drop memory across.
    * ``handler is not None`` — the finisher needs an input handler to
      merge with the block's compile output.
    * Both ``SUBMITIT_FOLDER`` AND ``SUBMITIT_EXECUTOR`` are set —
      ``SUBMITIT_EXECUTOR`` is what ``submitit.core.job_environment.JobEnvironment.activated()``
      itself checks (set by both the LocalExecutor controller and the
      SLURM plugin). These variables are necessary for the result handoff,
      but are insufficient to identify the process that owns that result.
    * ``isinstance(block, CompiledBlockFunction)`` — the finisher path
      only knows how to run that class's lifecycle.
    There is intentionally no opt-in flag: every situation where execvp
    is safe (the gates above) is also a situation where it strictly helps
    (drops 600+ MiB during a long wait), so we always take it.
    """
    import os

    if not dedicated_submitit_worker or getattr(block, "cloud", False):
        return False
    if not is_last:
        return False
    if recompile or handler is None:
        return False
    if "SUBMITIT_FOLDER" not in os.environ:
        return False
    if not os.environ.get("SUBMITIT_EXECUTOR"):
        return False
    # Lazy import to avoid a circular dep between chunking.py and compiled.py
    # at module load time.
    from chaperone.block_config.compiled import CompiledBlockFunction

    if not isinstance(block, CompiledBlockFunction):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ChunkContext
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChainStep:
    """A single block+directory pair within a chunk.

    Provides lazy block loading from a pickled cache file, avoiding
    N pickle operations when submitting job arrays to SLURM.
    """

    chunk_dir: Path
    block_path: Optional[Path] = None
    _cached_block: Optional["BlockFunction"] = field(default=None, repr=False)

    def get_block(self) -> "BlockFunction":
        """Load block from cache. Must call cache_blocks() first."""
        if self._cached_block is not None:
            return self._cached_block
        if self.block_path is None:
            raise ValueError("Block not cached. Call cache_blocks() first.")
        with open(self.block_path, "rb") as f:
            self._cached_block = cloudpickle.load(f)
        return self._cached_block


@dataclass
class ChunkContext:
    """Holds slice info for parallel chunk execution.

    steps contains one ChainStep per block. Single block = one step.
    Properties delegate to steps[0] for backward compatibility.
    """

    input_path: Path
    chunk_slice: tuple
    chunk_id: int
    steps: List[ChainStep] = field(default_factory=list)

    @property
    def chunk_dir(self) -> Path:
        """Primary chunk directory (first step)."""
        return self.steps[0].chunk_dir

    @property
    def block_path(self) -> Optional[Path]:
        """Primary block cache path (first step)."""
        return self.steps[0].block_path if self.steps else None

    def get_handler(self) -> PipelineDataHandler:
        """Load input data for this chunk from shared parquet."""
        return PipelineDataHandler(
            pl.read_parquet(self.input_path).slice(*self.chunk_slice)
        )

    def get_block(self) -> "BlockFunction":
        """Load primary block (first step) from cache."""
        return self.steps[0].get_block()

    @classmethod
    def create_chunks(
        cls,
        handler: PipelineDataHandler,
        rows_per_chunk: int,
        block_names: List[str],
        rundir: Path,
        num_chunks: Optional[int] = None,
    ) -> List["ChunkContext"]:
        """Create chunk contexts with one ChainStep per block name.

        Args:
            num_chunks: If set, overrides rows_per_chunk by computing
                        ceil(total_rows / num_chunks).
        """
        shared_parquet = rundir / ".pipeline" / f"input.{block_names[0]}.parquet"
        mkdir_p(shared_parquet.parent)
        handler = PipelineDataHandler(handler)
        handler.to_parquet(shared_parquet)

        if num_chunks is not None:
            rows_per_chunk = max(1, -(-len(handler) // num_chunks))

        return [
            cls(
                input_path=shared_parquet,
                chunk_slice=chunk_slice,
                chunk_id=chunk_id,
                steps=[
                    ChainStep(chunk_dir=rundir / name / f"chunk_{chunk_id}")
                    for name in block_names
                ],
            )
            for chunk_id, chunk_slice in handler.get_slices(rows_per_chunk).items()
        ]

    @staticmethod
    def cache_blocks(
        blocks: List["BlockFunction"],
        contexts: List["ChunkContext"],
        rundir: Path,
    ) -> None:
        """Cache blocks under .pipeline/block.<name>.pkl and set block_path on steps.

        Campaigns commonly define small custom metric blocks in a sibling
        ``score.py``.  Those modules are importable by the driver, but not
        necessarily by a fresh Slurm worker.  Register the defining module for
        by-value serialization while writing the shared block cache so workers
        do not need the campaign directory on ``sys.path``.
        """
        if not contexts:
            return
        pipeline_dir = rundir / ".pipeline"
        mkdir_p(pipeline_dir)
        for i, block in enumerate(blocks):
            safe_name = block.name.replace("/", "--")
            cache_path = pipeline_dir / f"block.{safe_name}.pkl"
            module = sys.modules.get(type(block).__module__)
            registered = False
            if module is not None:
                try:
                    cloudpickle.register_pickle_by_value(module)
                    registered = True
                except ValueError:
                    pass
            try:
                with open(cache_path, "wb") as f:
                    cloudpickle.dump(block, f)
            finally:
                if registered:
                    try:
                        cloudpickle.unregister_pickle_by_value(module)
                    except ValueError:
                        pass
            for ctx in contexts:
                ctx.steps[i].block_path = cache_path


# ─────────────────────────────────────────────────────────────────────────────
# Job function
# ─────────────────────────────────────────────────────────────────────────────


def execute_chunk_job(
    ctx: ChunkContext,
    allow_errors: bool = False,
    recompile: bool = False,
    force_exec: bool = False,
    skip_chunk_cache: bool = False,
    dedicated_submitit_worker: bool = False,
) -> Optional[Path]:
    """Execute all steps in a chunk sequentially via safe_call.

    For single blocks, the loop runs once. For chains, each block's output
    feeds into the next. Intermediate outputs are written so partial
    failures can resume.
    """
    del allow_errors  # Coordinator policy; a worker failure always exits nonzero.
    started = datetime.now(timezone.utc)
    terminal_path = ctx.steps[-1].chunk_dir / "terminal.json"
    current_step = None
    input_count = 0
    output_count = 0
    output_path = ctx.steps[-1].chunk_dir / "output.parquet"
    base_report = {
        "schema_version": 1,
        "chunk_id": ctx.chunk_id,
        "input_row_ids": list(
            range(ctx.chunk_slice[0], ctx.chunk_slice[0] + ctx.chunk_slice[1])
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_at": started.isoformat(),
        "blocks": [],
        "block_sha256": {},
    }
    try:
        for step in ctx.steps:
            block = step.get_block()
            base_report["blocks"].append(block.name)
            base_report["block_sha256"][block.name] = (
                _sha256(Path(step.block_path)) if step.block_path else None
            )
        handler = ctx.get_handler()
        input_count = len(handler)
        for i, step in enumerate(ctx.steps):
            block = step.get_block()
            current_step = block.name
            mkdir_p(step.chunk_dir)

            # Intermediate cache (allows chain resumption)
            output_path = step.chunk_dir / "output.parquet"
            if len(handler) == 0 and getattr(block, "_requires_input", True):
                logger.info("Skipping %s: no input rows remain", block.name)
                # Empty upstream results must also supersede a stale nonempty
                # child cache. Publish the empty result for normal collection.
                _atomic_write_parquet(handler, output_path)
                continue
            if (
                output_path.exists()
                and not recompile
                and not force_exec
                and not skip_chunk_cache
            ):
                handler = PipelineDataHandler(output_path)
                continue

            # Recompilation intentionally consumes the existing scientific
            # outputs. Every actual execution must start empty so a failed or
            # differently-chunked prior attempt cannot contaminate compile().
            if not recompile:
                _reset_chunk_sandbox_for_execution(step.chunk_dir)

            log_path = _get_local_log_path(step.chunk_dir)

            # Low-memory exec-finisher path: when running under submitit and
            # this is the last step of the chunk's chain, replace the heavy
            # harness with a thin runner_wait process via os.execvp. The
            # 2-3 GB of interpreter state goes back to the OS during the
            # long subprocess wait. The chunk log is tee'd by runner_wait
            # and runner_finish, so on-disk output matches the legacy path.
            # See `_should_use_exec_finisher` for the preconditions and
            # `CompiledBlockFunction.stage_then_exec_finisher` for the
            # handoff mechanics.
            if _should_use_exec_finisher(
                block=block,
                is_last=(i == len(ctx.steps) - 1),
                handler=handler,
                recompile=recompile,
                dedicated_submitit_worker=dedicated_submitit_worker,
            ):
                step_dir = step.chunk_dir.parent
                block.stage_then_exec_finisher(
                    handler=handler,
                    block_path=step.block_path,
                    sandbox_dir=step.chunk_dir,
                    chunk_id=ctx.chunk_id,
                    output_path=output_path,
                    log_path=log_path,
                    step_dir=step_dir,
                    input_row_ids=list(
                        range(
                            ctx.chunk_slice[0],
                            ctx.chunk_slice[0] + ctx.chunk_slice[1],
                        )
                    ),
                )
                # execvp does not return.
                raise RuntimeError("unreachable: stage_then_exec_finisher returned")

            with _tee_output(log_path):
                handler = safe_call(
                    block,
                    handler=handler,
                    sandbox_dir=step.chunk_dir,
                    chunk_id=ctx.chunk_id,
                    recompile=recompile,
                    force_exec=force_exec,
                )
            _atomic_write_parquet(handler, output_path)

        output_path = ctx.steps[-1].chunk_dir / "output.parquet"
        output_count = len(handler)
        _write_terminal_report(
            terminal_path,
            {
                **base_report,
                "status": "success",
                "step": current_step,
                "input_count": input_count,
                "output_count": output_count,
                "output_path": str(output_path),
                "output_sha256": _sha256(output_path),
                "exit_code": 0,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            },
        )
        return output_path
    except Exception as error:
        _write_terminal_report(
            terminal_path,
            {
                **base_report,
                "status": "error",
                "step": current_step,
                "input_count": input_count,
                "output_count": output_count,
                "output_path": str(output_path),
                "output_sha256": _sha256(output_path),
                "exit_code": 1,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc()[-12000:],
                },
            },
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


# /net/scratch mounts with acregmax=acdirmax=120 AND nocto, so close-to-open
# revalidation is off. A readdir bust clears a stale negative dentry most of the
# time, but not always, so recovery is bounded by something that clears the 120s
# TTL rather than by the bust alone. The wait is only ever paid when a chunk
# already looks failed, so a healthy run costs nothing -- and waiting out the
# TTL is strictly better than discarding work that is sitting on disk.
_RECOVERY_WAIT_S = 150.0
_RECOVERY_POLL_S = 2.0


def _bust_and_check(path: Path) -> bool:
    """Force a fresh readdir around *path*, then test existence.

    The stale entry is a *negative* dentry, so the parent must be re-read as
    well: listing only the leaf cannot help when the cached miss is on the
    leaf's own name.
    """
    for probe in (path.parent.parent, path.parent):
        try:
            os.listdir(probe)
        except OSError:
            pass
    return path.exists()


def recover_written_outputs(
    contexts, outputs, *, wait_s: float = _RECOVERY_WAIT_S, sleep=time.sleep
) -> int:
    """Accept chunks whose output parquet is on disk despite a failed future.

    submitit raises UncompletedJobError when the login node is still serving a
    directory listing that predates the compute node writing its result pickle.
    That is indistinguishable from a job which genuinely wrote nothing, so the
    filesystem is the tiebreaker: recovery requires readable Parquet and,
    when present, a successful terminal report with matching row count and digest.
    Without this the driver retries completed chunks
    and can fail the whole run with "N chunk(s) still failing" while every
    chunk's output.parquet is present.

    Mutates ``outputs`` in place and returns the number of chunks recovered.
    """
    pending = [i for i, out in enumerate(outputs) if out is None]
    if not pending:
        return 0

    deadline = time.monotonic() + wait_s
    recovered_count = 0
    while pending:
        still_pending = []
        for i in pending:
            parquet = contexts[i].steps[-1].chunk_dir / "output.parquet"
            terminal = parquet.with_name("terminal.json")
            report_present = True
            try:
                report = json.loads(terminal.read_text())
            except FileNotFoundError:
                report_present = False  # Older workers have no terminal reports.
                report = None
            except (OSError, ValueError):
                still_pending.append(i)
                continue
            if isinstance(report, dict) and report.get("status") == "error":
                continue
            if report_present and (
                not isinstance(report, dict) or report.get("status") != "success"
            ):
                still_pending.append(i)
                continue
            if not _bust_and_check(parquet):
                still_pending.append(i)
                continue
            try:
                # Read all pages, not just the footer: a partial/corrupt result
                # must remain eligible for retry instead of being called success.
                with parquet.open("rb") as handle:
                    frame = pl.read_parquet(handle)
                if report is not None:
                    if (
                        "output_count" in report
                        and report["output_count"] != frame.height
                    ):
                        still_pending.append(i)
                        continue
                    if report.get("output_sha256") and report[
                        "output_sha256"
                    ] != _sha256(parquet):
                        still_pending.append(i)
                        continue
            except (OSError, pl.exceptions.PolarsError):
                still_pending.append(i)
                continue
            logger.warning(
                "Chunk %s future failed but its output is readable and consistent "
                "with any available terminal report; recovering %s.",
                contexts[i].chunk_id,
                parquet,
            )
            outputs[i] = parquet
            recovered_count += 1
        pending = still_pending
        if not pending or time.monotonic() >= deadline:
            break
        sleep(min(_RECOVERY_POLL_S, max(0.0, deadline - time.monotonic())))

    if pending:
        logger.info(
            "%d chunk(s) had no recoverable output.parquet after %.0fs; treating as failed",
            len(pending),
            wait_s,
        )
    return recovered_count


def execute_chunks(
    blocks: List["BlockFunction"],
    handler_in,
    rundir,
    run_args,
) -> List[Optional[Path]]:
    """Execute block(s) across chunks with retry logic.

    Handles chunk creation, caching, dispatch, and SLURM retry with
    node exclusion. Single block or chain — same path.
    """
    from chaperone.engine.resolve import _create_runner

    block_names = [b.name for b in blocks]
    label = " → ".join(block_names) if len(blocks) > 1 else blocks[0].name

    # Pre-chunk hook: blocks may re-order the input handler before chunking.
    # AF3 uses this to sort by token length so each chunk slice covers a
    # narrow JIT-bucket range. Hook is opt-in per-block; default no-op.
    # Keyed off the first block since chunk slicing is shared across the chain.
    first_block = blocks[0]
    if hasattr(first_block, "pre_chunk_sort") and handler_in is not None:
        handler_in = first_block.pre_chunk_sort(handler_in)

    contexts = ChunkContext.create_chunks(
        handler_in,
        run_args.job_args.rows_per_job,
        block_names,
        rundir,
        num_chunks=run_args.job_args.num_chunks,
    )

    # Filter cached (check last step's output)
    contexts_to_run, cached_outputs = filter_cached_chunks(
        contexts, run_args.skip_chunk_cache or run_args.force_exec or run_args.recompile
    )
    if not contexts_to_run:
        console.print(
            f"[blue]⚡ All {len(cached_outputs)} chunks cached for {label}[/blue]"
        )
        return list(cached_outputs)

    ChunkContext.cache_blocks(blocks, contexts_to_run, rundir)

    log_folder = str(rundir / "logs" / blocks[0].name / "slurm")
    max_retries = run_args.max_retries
    excluded_nodes = run_args.job_args.excluded_nodes_set

    previous_failed_count = None
    for attempt in range(max_retries + 1):
        # Slurm failures must reach recovery even on the final attempt. The
        # explicit failure check below still rejects incomplete results.
        attempt_args = run_args.for_attempt(
            is_last=attempt == max_retries and run_args.is_local,
            excluded_nodes=excluded_nodes,
        )

        runner = _create_runner(attempt_args, log_folder)
        if (attempt < max_retries or not run_args.is_local) and hasattr(
            runner, "fail_on_error"
        ):
            runner.fail_on_error = False

        new_outputs = _dispatch_jobs(
            execute_chunk_job,
            contexts_to_run,
            runner,
            attempt_args,
            label,
            recompile=run_args.recompile,
            force_exec=run_args.force_exec,
            skip_chunk_cache=run_args.skip_chunk_cache,
            dedicated_submitit_worker=(run_args.mode == "slurm"),
        )
        # Recover only ordinary Slurm runs. Forced execution/recompilation may
        # still have an old parquet if the worker never started.
        if run_args.mode == "slurm" and not (
            run_args.force_exec or run_args.skip_chunk_cache or run_args.recompile
        ):
            recover_written_outputs(contexts_to_run, new_outputs)
        failed_count = sum(1 for p in new_outputs if p is None)

        if failed_count == 0 or attempt == max_retries:
            break

        # Query sacct for failed nodes and add to exclude list
        newly_excluded = (
            (extract_failed_nodes(run_args.job_args.job_name) - excluded_nodes)
            if run_args.mode == "slurm"
            else set()
        )

        if newly_excluded:
            excluded_nodes.update(newly_excluded)
            console.print(
                f"[yellow] Retry {attempt + 1}/{max_retries}: "
                f"{failed_count} chunk(s) failed, excluding nodes "
                f"{sorted(newly_excluded)}[/yellow]"
            )
        elif (
            previous_failed_count is not None and failed_count >= previous_failed_count
        ):
            console.print(
                f"[yellow]{failed_count} chunk(s) still failed after a retry "
                "with no progress; stopping[/yellow]"
            )
            break
        else:
            console.print(
                f"[yellow]Retry {attempt + 1}/{max_retries}: "
                f"{failed_count} chunk(s) failed with no node correlation; "
                "retrying on the same nodes[/yellow]"
            )
        previous_failed_count = failed_count

    # Both exhaustion and an early no-progress stop must fail explicitly.
    if failed_count:
        raise RuntimeError(
            f"{label}: {failed_count} chunk(s) still failing after "
            f"{attempt} retries (excluded nodes: {sorted(excluded_nodes)})"
        )

    # ── NFS attr-cache bust ────────────────────────────────────────────
    # filter_cached_chunks() stat()'d every output path, caching negative
    # DNLC entries ("doesn't exist") for files SLURM hadn't written yet.
    # The old code slept out the whole attr-cache TTL (up to ~120s of dead
    # wall-clock, per the /net/scratch acregmax=acdirmax=120 mount config)
    # before reading. Instead,
    # actively force a fresh readdir on each step's output directory: that
    # invalidates the stale negative entries so concat_outputs() sees the
    # freshly-written parquets immediately. concat_outputs() additionally
    # retries the actual reads (_read_parquet_with_nfs_retry) for any
    # residual staleness. Local execution never touches the shared NFS mount,
    # so skip the bust there.
    if not run_args.is_local:
        for name in block_names:
            try:
                os.listdir(rundir / name)
            except OSError:
                pass

    # Consolidate intermediate step outputs for chains
    if len(blocks) > 1:
        for name in block_names:
            step_parquet = rundir / name / "output.parquet"
            if not step_parquet.exists():
                chunk_outputs = sorted((rundir / name).glob("chunk_*/output.parquet"))
                if chunk_outputs:
                    mkdir_p(step_parquet.parent)
                    concat_outputs(chunk_outputs).write_parquet(step_parquet)

    return list(cached_outputs) + new_outputs


def filter_cached_chunks(
    contexts: List[ChunkContext], skip_cache: bool = False
) -> tuple[List[ChunkContext], List[Path]]:
    """Filter out chunks whose last step already has output.parquet."""
    if skip_cache:
        return contexts, []

    to_run = []
    cached = []
    for ctx in contexts:
        output_path = ctx.steps[-1].chunk_dir / "output.parquet"
        if output_path.exists():
            cached.append(output_path)
        else:
            to_run.append(ctx)

    if cached:
        console.print(
            f"[blue]⚡ Using {len(cached)} cached chunks, running {len(to_run)} new[/blue]"
        )

    return to_run, cached


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch & SLURM helpers
# ─────────────────────────────────────────────────────────────────────────────


def _dispatch_jobs(job_fn, contexts, runner, run_args, label, **extra_kwargs):
    """Run job_fn across contexts using GPU pool, SLURM runner, or sequential fallback."""
    from chaperone.orchestration.gpu_local_taskrunner import GPULocalTaskRunner

    n = len(contexts)
    if run_args.is_local and isinstance(runner, GPULocalTaskRunner) and n > 1:
        console.print(
            f"[bold]Running {n} jobs for {label} "
            f"(local, {len(runner.gpu_ids)} GPUs)[/bold]"
        )

        return run_with_cancel(
            lambda: _run_mapped_jobs(
                job_fn, contexts, runner, run_args.allow_errors, extra_kwargs
            ),
            label,
            runner,
        )
    elif (
        run_args.is_local
        and (local_workers := getattr(run_args.job_args, "local_workers", None))
        is not None
        and n > 1
    ):
        console.print(
            f"[bold]Running {n} jobs for {label} (local, {local_workers} workers)[/bold]"
        )
        log_folder = contexts[0].steps[0].chunk_dir.parent / "submitit_logs"
        executor = submitit.LocalExecutor(folder=str(log_folder))
        executor.update_parameters(timeout_min=60 * 24)
        # Submit in batches of local_workers to bound peak concurrency
        results = []
        for batch_start in range(0, n, local_workers):
            batch = contexts[batch_start : batch_start + local_workers]
            jobs = [
                executor.submit(
                    job_fn, ctx, allow_errors=run_args.allow_errors, **extra_kwargs
                )
                for ctx in batch
            ]
            for job in tqdm(
                jobs, desc=label, total=len(jobs), disable=progress_disabled()
            ):
                try:
                    results.append(job.result())
                except Exception:
                    if not run_args.allow_errors:
                        raise
                    results.append(None)
        return results
    elif run_args.is_local:
        console.print(f"[bold]Running {n} jobs for {label} (local)[/bold]")
        results = []
        for ctx in tqdm(contexts, desc=label, disable=progress_disabled()):
            try:
                results.append(
                    job_fn(ctx, allow_errors=run_args.allow_errors, **extra_kwargs)
                )
            except Exception:
                if not run_args.allow_errors:
                    raise
                results.append(None)
        return results
    else:
        console.print(
            f"[bold]Running {n} jobs for {label} ({run_args.mode.upper()})[/bold]"
        )

        return run_with_cancel(
            lambda: _run_mapped_jobs(
                job_fn, contexts, runner, run_args.allow_errors, extra_kwargs
            ),
            label,
            runner,
        )


def _run_mapped_jobs(job_fn, contexts, runner, allow_errors, extra_kwargs):
    """Map jobs through a task runner and collect results in input order."""
    n = len(contexts)
    with runner:
        futures = runner.map(
            job_fn,
            ctx=contexts,
            allow_errors=[allow_errors] * n,
            **{k: [v] * n for k, v in extra_kwargs.items()},
        )
        # Honour allow_errors at the job level too: a failed job yields None
        # instead of raising and aborting the whole pipeline.
        return [f.result(raise_on_failure=not allow_errors) for f in futures]


def run_with_cancel(flow_fn, label: str, runner):
    """Run SLURM flow with keyboard interrupt handling."""
    try:
        return flow_fn()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted![/yellow] Cancelling SLURM jobs...")
        if hasattr(runner, "_executor") and runner._executor:
            try:
                subprocess.run(
                    ["scancel", "-n", label], capture_output=True, timeout=10
                )
                console.print("[green]Jobs cancelled.[/green]")
            except Exception as e:
                console.print(f"[red]Cancel failed: {e}[/red]")
        raise


def extract_failed_nodes(job_name: str) -> set[str]:
    """Query sacct for nodes where SLURM jobs failed, then expand compressed nodelists.

    Returns an empty set on any failure (no sacct, timeout, etc.) so that
    retry logic degrades gracefully on systems without SLURM accounting.
    """
    try:
        result = subprocess.run(
            [
                "sacct",
                f"--name={job_name}",
                "--state=FAILED,NODE_FAIL,TIMEOUT,OUT_OF_MEMORY",
                "--starttime=now-2hours",
                "--format=NodeList%50",
                "--parsable2",
                "--noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.debug(f"sacct returned {result.returncode} for job {job_name}")
            return set()

        raw_nodes = {
            line.strip()
            for line in result.stdout.strip().splitlines()
            if line.strip() and line.strip() != "None assigned"
        }
        if not raw_nodes:
            return set()

        # Expand compressed nodelists (e.g. "node[01-03]" → node01, node02, node03)
        expanded: set[str] = set()
        for nodelist in raw_nodes:
            try:
                expand = subprocess.run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if expand.returncode == 0:
                    expanded.update(
                        h.strip()
                        for h in expand.stdout.strip().splitlines()
                        if h.strip()
                    )
                else:
                    expanded.add(nodelist)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                expanded.add(nodelist)

        return expanded

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"extract_failed_nodes failed for {job_name}: {e}")
        return set()


def _read_parquet_with_nfs_retry(
    path: Path, max_retries: int = 12, delay: float = 2.0
) -> pl.DataFrame:
    """Read parquet with retries to handle NFS attr-cache lag.

    Twelve attempts with linear backoff is ~156s.  We have observed valid
    chunk parquets become visible 85--100s after their Slurm job completed, so
    the previous eight-attempt/~72s window could misclassify successful GPU
    work as failed.  A large array finishing together is exactly when the attr
    cache is most stale and when giving up is most expensive.
    """
    import time

    for attempt in range(max_retries):
        # Refresh the directory that directly contains the file. Re-reading
        # only the step directory does not invalidate a negative dentry cached
        # for ``chunk_N/output.parquet`` on the submitting node.
        try:
            os.listdir(path.parent)
        except OSError:
            pass
        try:
            return pl.read_parquet(path)
        except FileNotFoundError:
            # Polars' long-lived IO runtime can retain a negative path lookup
            # after the Python process can already open the newly published
            # NFS file.  Reading through a fresh file descriptor bypasses that
            # path cache without another worker, copy, or driver restart.
            try:
                with path.open("rb") as parquet_file:
                    return pl.read_parquet(parquet_file)
            except FileNotFoundError:
                pass
            if attempt < max_retries - 1:
                # A mutation is required for stubborn negative dentries seen
                # through long-lived nested Apptainer/FUSE mounts; a readdir
                # alone may continue returning the cached miss.
                refresh = path.parent / f".nfs_parquet_refresh_{os.getpid()}_{attempt}"
                try:
                    refresh.touch()
                except OSError:
                    pass
                logger.debug(f"NFS retry {attempt + 1}/{max_retries} for {path}")
                time.sleep(delay * (attempt + 1))
            else:
                terminal = path.parent / "terminal.json"
                if terminal.is_file():
                    try:
                        outcome = json.loads(terminal.read_text())
                    except (OSError, json.JSONDecodeError):
                        outcome = None
                    if isinstance(outcome, dict) and outcome.get("status") == "error":
                        error = outcome.get("error") or {}
                        raise RuntimeError(
                            "Worker terminal report records failure before Parquet "
                            f"publication: {error.get('type', 'WorkerError')}: "
                            f"{error.get('message', 'unknown error')} "
                            f"(report: {terminal})"
                        )
                # The structures may well be on disk with only the collation
                # lost. Say so: campaigns have hand-written per-block recovery
                # scripts to rebuild these parquets, not knowing --recompile
                # already re-collects from the sandbox without re-running
                # inference.
                produced = list(path.parent.glob("**/*.cif")) + list(
                    path.parent.glob("**/*.pdb")
                )
                raise FileNotFoundError(
                    f"{path} did not appear after {max_retries} attempts "
                    f"(~{int(delay * max_retries * (max_retries + 1) / 2)}s). "
                    + (
                        f"The chunk sandbox still holds {len(produced)} structure "
                        "file(s), so re-run with --recompile to re-collect them "
                        "instead of re-running the compute."
                        if produced
                        else "No structure files were found in the chunk sandbox, "
                        "so the job itself likely failed — check its Slurm log."
                    )
                )


def concat_outputs(output_parquets: List[Optional[Path]]) -> PipelineDataHandler:
    """Concatenate chunk outputs into single handler."""
    valid = [p for p in output_parquets if p is not None]
    failed = len(output_parquets) - len(valid)

    if failed:
        logger.warning(f"Skipped {failed}/{len(output_parquets)} chunks due to errors")
        console.print(
            f"[yellow]⚠ Skipped {failed}/{len(output_parquets)} chunks[/yellow]"
        )

    if not valid:
        raise RuntimeError(f"All {len(output_parquets)} chunks failed")

    return PipelineDataHandler(
        pl.concat(
            [_read_parquet_with_nfs_retry(p) for p in valid],
            how="diagonal_relaxed",
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Local log tee
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def _tee_output(log_path: Path | None):
    """Context manager that tees stdout/stderr to a log file. No-op if log_path is None.

    Skips tee in non-main threads to avoid thread-safety issues with sys.stdout/stderr.
    """
    if log_path is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    mkdir_p(log_path.parent)
    log_file = open(log_path, "w")

    log_file.write(f"=== START {datetime.now().isoformat(timespec='seconds')} ===\n\n")
    log_file.flush()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeWriter(old_stdout, log_file)
    sys.stderr = _TeeWriter(old_stderr, log_file)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        log_file.write(
            f"\n=== END {datetime.now().isoformat(timespec='seconds')} | elapsed: {elapsed:.1f}s ===\n"
        )
        log_file.close()


def _get_local_log_path(chunk_dir: Path) -> Path | None:
    """Return log path for local execution, or None if logging is disabled."""
    from chaperone import get_settings

    if not get_settings().ENABLE_LOCAL_LOGS:
        return None
    rundir = chunk_dir.parent.parent
    block_name = chunk_dir.parent.name
    return rundir / "logs" / block_name / f"{chunk_dir.name}.log"
