"""
ExecutionEngine: Orchestrates block execution across local and SLURM runners.

BLOCK BEHAVIOR (configured via ClassVars):
    - _inline: True = Python execution, False = external command via manifest
    - _chunked: True = split into parallel chunks, False = single execution
    - _creates_new_step: True = new step, False = metric/update block (no new structures)

EXECUTION MODES:
    - inline + not chunked: Direct Python call (FilterBlock, IngestBlock)
    - inline + chunked: Python with parallel chunks (InlineBlockFunction)
    - not inline + chunked: External command with manifests (CompiledBlockFunction)

"""

import dataclasses
import hashlib
import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION, ALL_COMPLETED
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Type, Any
import polars as pl
from rich.panel import Panel

from chaperone.common import get_console

from chaperone.block_config.base import BlockFunction
from chaperone.data import PipelineDataHandler
from chaperone.manifest import PipelineManifest
from chaperone.orchestration import JobArgs

from chaperone.engine.chunking import (
    _tee_output,
    concat_outputs,
    execute_chunks,
    safe_call,
)

from chaperone.engine.resolve import (
    RuntimeArgs,
    _resolve_block,
    _resolve_input_data,
    _resolve_run_args,
    _resolve_job_args,
)

logger = logging.getLogger(__name__)
console = get_console()


def print_panel(message: str, style: str = "green"):
    console.print(Panel(message, border_style=style, width=50))


# ─────────────────────────────────────────────────────────────────────────────
# Submit helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class PreparedRun:
    """All resolved state needed to execute a run. Produced by _prepare()."""

    blocks: List[BlockFunction]
    handler_in: Optional[PipelineDataHandler]
    run_args: RuntimeArgs
    output_parquet: Path
    cached_handler: Optional[PipelineDataHandler]
    cache_fingerprint: str


def _cleanup_chunk_sandboxes(step_dir: Path) -> None:
    """Remove chunk workspaces after their catcif archives are durable."""
    for chunk_dir in sorted(step_dir.glob("chunk_*")):
        if chunk_dir.is_dir():
            shutil.rmtree(chunk_dir, ignore_errors=True)
    logger.info(f"Cleaned per-chunk sandboxes under {step_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionEngine:
    """Orchestrates block execution with manifest tracking and SLURM support."""

    def __init__(
        self,
        rundir: str | Path | None = None,
        reinit: bool = False,
        run_args_default: "RuntimeArgs | dict | None" = None,
        job_args_default: "JobArgs | dict | None" = None,
        auto_store_format: str | None = None,
        auto_store_cleanup: bool = False,
        cleanup_logs: bool = False,
        _parent: "ExecutionEngine" = None,
        _namespace: str = None,
    ):
        """Attach to *rundir*, loading its existing manifest by default.

        ``reinit=True`` starts an empty in-memory manifest; it does not disable
        output caching. Leave ``reinit=False`` to inspect or resume an existing
        run, including its recorded steps and execution summaries.
        """
        from chaperone import get_settings

        settings = get_settings()

        # CLI --rundir always wins
        from chaperone import get_runtime_overrides

        runtime_overrides = get_runtime_overrides()
        cli_rundir = runtime_overrides.get("rundir")
        if cli_rundir is not None:
            rundir = cli_rundir

        # CLI auto-store overrides (set by init_cli via --auto_store_format / --auto_store_cleanup;
        # --test forces both to the disabled state unless explicitly set).
        if "auto_store_format" in runtime_overrides:
            fmt = runtime_overrides["auto_store_format"]
            auto_store_format = None if fmt in (None, "", "none") else fmt
        if "auto_store_cleanup" in runtime_overrides:
            auto_store_cleanup = bool(runtime_overrides["auto_store_cleanup"])
        if "cleanup_logs" in runtime_overrides:
            cleanup_logs = bool(runtime_overrides["cleanup_logs"])

        if rundir is None:
            rundir = settings.RUNDIR
        if rundir is None:
            raise ValueError(
                "rundir required via argument, CHAPERONE_RUNDIR env var, or .env file"
            )

        rundir = Path(rundir)

        # Resolve relative paths against CHAPERONE_ROOT
        if not rundir.is_absolute():
            rundir = settings.CHAPERONE_ROOT / rundir
            logger.info(f"Resolved relative rundir to: {rundir}")
        elif settings.ENFORCE_RELATIVE_RUNDIR:
            # Enforce relative paths in testing mode
            raise ValueError(
                f"Absolute rundir '{rundir}' not allowed when ENFORCE_RELATIVE_RUNDIR=True. "
                f"Use a relative path to have it resolved under CHAPERONE_ROOT ({settings.CHAPERONE_ROOT})."
            )

        self.rundir = rundir.absolute()
        self.run_args_default = run_args_default
        self.job_args_default = job_args_default
        self._parent = _parent
        self._namespace = _namespace
        if auto_store_format is not None and auto_store_format != "catcif":
            raise ValueError(
                f"auto_store_format must be None or 'catcif'; got {auto_store_format!r}"
            )
        self.auto_store_format = auto_store_format
        self.auto_store_cleanup = auto_store_cleanup
        self.cleanup_logs = cleanup_logs
        self._input_cache: dict[Path, PipelineDataHandler] = {}
        self._lock = threading.Lock()  # Protects manifest and _input_cache

        from chaperone import _internally_set_settings

        _internally_set_settings({"RUNDIR": self.rundir})
        from chaperone.containers import capture_container_runtime_bind_settings

        self._container_bind_settings = capture_container_runtime_bind_settings(self.rundir)

        if reinit:
            self.manifest = PipelineManifest(self.rundir)
        else:
            self.manifest = PipelineManifest.load(self.rundir)
            logger.info(f"Loaded manifest with {len(self.manifest)} steps")

    def __repr__(self) -> str:
        ns = f", namespace={self._namespace}" if self._namespace else ""
        lines = [f"ExecutionEngine(rundir={self.rundir}{ns})"]
        if self._parent:
            lines.append(f"  parent: {self._parent.rundir}")
        for line in repr(self.manifest).split("\n"):
            lines.append(f"  {line}")
        return "\n".join(lines)

    def _apply_defaults(self, run_args, job_args):
        """Apply defaults, retaining runtime policy under partial dict overrides."""
        if run_args is None and self.run_args_default is not None:
            run_args = self.run_args_default
        elif isinstance(run_args, dict) and self.run_args_default is not None:
            defaults = self.run_args_default
            if dataclasses.is_dataclass(defaults):
                defaults = {
                    field.name: getattr(defaults, field.name)
                    for field in dataclasses.fields(defaults)
                }
            run_args = {**defaults, **run_args}
        if job_args is None and self.job_args_default is not None:
            job_args = self.job_args_default
        return run_args, job_args

    # ─────────────────────────────────────────────────────────────────────────
    # Compilation & Namespace API
    # ─────────────────────────────────────────────────────────────────────────

    def compile(self, steps: List[str] = None) -> Optional[PipelineDataHandler]:
        """Compile step parquets into merged handler with full lineage.

        Loads each step's parquet and merges them sequentially.
        """

        def _load_step(name: str) -> PipelineDataHandler:
            return PipelineDataHandler(
                pl.read_parquet(self.manifest.get_parquet_path(name))
            )

        if steps is None:
            return _load_step("last")

        assert steps, "Cannot compile: no steps in manifest."
        logger.info(f"Compiling steps: {', '.join(steps)}")

        result = _load_step(steps[0])
        for name in steps[1:]:
            result = PipelineDataHandler.merge(result, _load_step(name))
        return result

    def with_namespace(self, namespace: str) -> "ExecutionEngine":
        """Create engine view with namespace prefix for block names."""
        child = ExecutionEngine.__new__(ExecutionEngine)
        child.rundir = self.rundir
        child.manifest = PipelineManifest.load(self.rundir)
        child._parent = self._parent
        child._namespace = namespace
        child._input_cache = {}
        child._lock = threading.Lock()
        child.run_args_default = self.run_args_default
        child.job_args_default = self.job_args_default
        child.auto_store_format = getattr(self, "auto_store_format", None)
        child.auto_store_cleanup = getattr(self, "auto_store_cleanup", False)
        child.cleanup_logs = getattr(self, "cleanup_logs", False)
        child.manifest.clear_namespace(namespace)
        return child

    def cycle(self, name: str, n: int) -> Iterator["ExecutionEngine"]:
        """Yield n namespaced engines for iterative refinement.

        Clears the entire cycling block's namespace from the manifest up
        front (all `<name>/cycle_*` entries, including stale ones from
        prior runs that crashed mid-cycle). This guarantees the manifest
        records a linear, in-order trajectory of cycles for this run
        instead of an interleaved order where re-registered cycles
        appear after stranded entries from previous attempts. Cached
        per-step parquets on disk are untouched and will still cache-hit
        when each iteration re-runs.
        """
        base = f"{self._namespace}/{name}" if self._namespace else name
        self.manifest.clear_namespace(base)
        self.manifest.save()

        try:
            for i in range(n):
                ns = f"{base}/cycle_{i}"
                yield self.with_namespace(ns)
        finally:
            self.manifest = PipelineManifest.load(self.rundir)

    # ─────────────────────────────────────────────────────────────────────────
    # Block execution
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_auto_store(self, blocks: List[BlockFunction]) -> None:
        """Apply ``auto_store_format`` to any compiled block with a default ``store``.

        Blocks whose ``store`` was set explicitly (present in ``model_fields_set``)
        are left alone — the block author's choice always wins. Inline blocks
        are skipped since only compiled blocks honour ``self.store``.
        """
        if self.auto_store_format is None:
            return
        from chaperone.block_config.compiled import CompiledBlockFunction

        for b in blocks:
            if not isinstance(b, CompiledBlockFunction):
                continue
            if getattr(b, "output_mode", None) == "sequences":
                # Sequence-native blocks have no coordinates to archive.
                continue
            if "store" in b.model_fields_set:
                continue
            b.store = self.auto_store_format
            if "store_cleanup" not in b.model_fields_set:
                b.store_cleanup = self.auto_store_cleanup

    def _prepare(
        self,
        block: str | list | BlockFunction | Type[BlockFunction],
        run_args: RuntimeArgs | dict | None = None,
        data: Any = None,
        job_args: JobArgs | dict | None = None,
        **kwargs,
    ) -> PreparedRun:
        """Resolve arguments, blocks, input data, and check cache. Fast, synchronous."""
        # Resolve all arguments — resolution uses primary block (blocks[0])
        run_args, job_args = self._apply_defaults(run_args, job_args)
        blocks: List[BlockFunction] = _resolve_block(block, kwargs, self._namespace)
        self._apply_auto_store(blocks)
        self._configure_container_binds(blocks)
        block = blocks[0]
        run_args: RuntimeArgs = _resolve_run_args(run_args, block_name=block._name)
        from chaperone.block_config.compiled import CompiledBlockFunction

        # Legacy cloud=True requests Modal even on a Slurm-equipped host.
        if any(isinstance(b, CompiledBlockFunction) and b.cloud for b in blocks):
            if not run_args.recompile:
                run_args.mode = "modal"
        effective_job_args = (
            run_args.job_args if run_args.job_args is not None else job_args
        )
        run_args.job_args = _resolve_job_args(effective_job_args, block, run_args)
        # For multi-block chains, if the first block has no resources (e.g. inline
        # SelectMotif), resolve job_args from the first block that does.
        if len(blocks) > 1 and run_args.job_args is None:
            for b in blocks[1:]:
                run_args.job_args = _resolve_job_args(effective_job_args, b, run_args)
                if run_args.job_args is not None:
                    break
        # For multi-block chains, default job_name to joined block names
        if len(blocks) > 1 and run_args.job_args is not None:
            from dataclasses import replace as dc_replace

            # Only override if job_name was auto-set to the first block's name
            if run_args.job_args.job_name == block.name:
                chain_name = "-".join(b._name for b in blocks)
                run_args.job_args = dc_replace(run_args.job_args, job_name=chain_name)

        if run_args.mode == "modal":
            blocks = [b.model_copy(deep=True) for b in blocks]
            for b in blocks:
                if isinstance(b, CompiledBlockFunction):
                    b.cloud = True
                    b._modal_job_args = _resolve_job_args(effective_job_args, b, run_args)
            block = blocks[0]

        for b in blocks:
            if isinstance(b, CompiledBlockFunction):
                b._container_job_args = _resolve_job_args(effective_job_args, b, run_args)

        # Local mode never retries (no SLURM nodes to exclude)
        if run_args.mode == "local":
            run_args.max_retries = 0
        handler_in = _resolve_input_data(data, block)

        # Resolve output paths — use last block for chain output
        output_block = blocks[-1]
        output_parquet = self.rundir / output_block.name / "output.parquet"

        # ── Cache check ──
        cache_fingerprint = self._cache_fingerprint(
            blocks, handler_in=handler_in, run_args=run_args
        )
        cached_handler = self._check_cache(
            blocks, output_parquet, handler_in, run_args, cache_fingerprint
        )

        return PreparedRun(
            blocks=blocks,
            handler_in=handler_in,
            run_args=run_args,
            output_parquet=output_parquet,
            cached_handler=cached_handler,
            cache_fingerprint=cache_fingerprint,
        )

    def _execute_and_finalize(self, prep: PreparedRun) -> PipelineDataHandler:
        """Execute block(s) and register results. Potentially slow (SLURM/GPU)."""
        step_name = prep.blocks[-1].name

        if prep.cached_handler is not None:
            handler_out = prep.cached_handler
        else:
            handler_in = prep.handler_in
            if handler_in is None:
                handler_in = prep.blocks[0].init_handler()

            # Persist before dispatch so completed chunks from an interrupted
            # run are reusable only by the exact same block configuration.
            self._write_cache_fingerprint(
                prep.output_parquet, prep.cache_fingerprint
            )
            handler_out = self._run(
                prep.blocks,
                handler_in,
                prep.run_args,
            )

            prep.output_parquet.parent.mkdir(parents=True, exist_ok=True)
            handler_out.write_parquet(prep.output_parquet)
            with self._lock:
                self._input_cache[prep.output_parquet] = handler_out

        if prep.blocks[-1]._emits_sequence_table:
            handler_out.write_sequences_csv(
                prep.output_parquet.parent / "sequences.csv"
            )

        with self._lock:
            if prep.cached_handler is None:  # cache miss → execution happened
                self._invalidate_downstream_cache(prep.blocks[-1].name)
            # The chain's external parent is the step that produced the INPUT
            # handler — read from its lineage stamp (set by the producing step's
            # ``stamp_producing_step``). This is correct under branching (one
            # producer feeding several consumers) and passthrough blocks, where
            # the manifest tail (last-registered step) would be wrong. Falls
            # back to the manifest tail for raw/unstamped inputs (legacy paths).
            # For chained blocks (run([mpnn, af3])), block[i] is fed by block[i-1].
            chain_parents: list[str] = []
            if prep.handler_in is not None:
                chain_parents = list(getattr(prep.handler_in, "_producing_steps", []))
            if not chain_parents:
                tail = self.manifest.last_step_name()
                chain_parents = [tail] if tail else []
            for i, b in enumerate(prep.blocks):
                if i == 0:
                    parents = chain_parents
                else:
                    parents = [prep.blocks[i - 1].name]
                # Self-parent guards against an already-registered step
                # snapping to itself (cache-hit re-registration, or a stamp that
                # names this step). Drops blanks and de-duplicates.
                parents = list(dict.fromkeys(p for p in parents if p and p != b.name))
                # A full cache hit must not move an existing step to the end of
                # the ordered manifest. Reordering an upstream cached step can
                # make a later refresh classify that upstream artifact as
                # downstream and delete it during invalidation.
                if (
                    prep.cached_handler is not None
                    and self.manifest.get_step(b.name) is not None
                ):
                    continue
                self._register_step(
                    b,
                    self.rundir / b.name / "output.parquet",
                    parent_step_names=parents,
                )
        # Stamp the output with the producing step so a downstream
        # ``engine.run(next, data=handler_out)`` registers the true data parent
        # (this step) regardless of manifest registration order. This is what
        # makes branches and passthrough blocks resolve correctly.
        try:
            handler_out.stamp_producing_step(step_name)
        except Exception:  # noqa: BLE001
            pass

        # The step compiled — drop its .pipeline/ scratch (chunk input handler +
        # cached block pickle). They are only needed while chunk workers run.
        if prep.cached_handler is None:
            try:
                from chaperone.engine.cleanup import remove_pipeline_scratch

                remove_pipeline_scratch(self.rundir, [b.name for b in prep.blocks])
            except Exception:  # noqa: BLE001
                pass

        return handler_out

    def run(
        self,
        block: str | list | BlockFunction | Type[BlockFunction],
        run_args: RuntimeArgs | dict | None = None,
        data: Any = None,
        job_args: JobArgs | dict | None = None,
        **kwargs,
    ) -> PipelineDataHandler:
        """Execute a block or a chain of blocks.

        Block behavior is determined by ClassVars:
            - _inline: True = Python, False = external command
            - _chunked: True = parallel chunks, False = single execution
            - _creates_new_step: True = new step, False = metric/update block

        When block is a list, executes all blocks as a chain within the same
        distributed resources. Each chunk runs all blocks sequentially, feeding
        each block's output into the next.

        Args:
            block: Block name, class, instance, or list of these for chain execution.
            run_args: Runtime and job configuration. Accepts RuntimeArgs, dict, or None.
                      Use run_args.job_args for SLURM settings (partition, mem_gb, etc.).
            job_args: SLURM job configuration (deprecated, use run_args.job_args instead).
            data: Explicit input data for this block. Accepts DataFrame (packed
                  or unpacked), PipelineDataHandler, or None for source blocks
                  that generate their own inputs. Unpacked DataFrames (from
                  handler.unpack()) are automatically re-packed.
        """
        prep = self._prepare(
            block, run_args=run_args, data=data, job_args=job_args, **kwargs
        )
        return self._execute_and_finalize(prep)

    def run_parallel(
        self,
        blocks: List[str | BlockFunction | Type[BlockFunction]],
        run_args: RuntimeArgs | dict | None = None,
        data: Any = None,
        job_args: JobArgs | dict | None = None,
        **kwargs,
    ) -> Dict[str, PipelineDataHandler]:
        """Execute multiple overlay (update) blocks in parallel.

        Each block writes its own overlay parquet — no write contention.
        Only blocks with _creates_new_step=False are allowed.

        Args:
            blocks: List of block specs (strings, types, or instances)
            run_args: Runtime configuration shared by all blocks
            data: Explicit input data shared by all overlay blocks
            job_args: SLURM job configuration (shared)

        Returns:
            Dict of block_name → handler
        """
        for block_spec in blocks:
            resolved = _resolve_block(block_spec, kwargs, self._namespace)
            for b in resolved:
                if b._creates_new_step:
                    raise ValueError(
                        f"run_parallel only supports update blocks (_creates_new_step=False). "
                        f"Block '{b.name}' has _creates_new_step=True."
                    )

        results: Dict[str, PipelineDataHandler] = {}
        errors: Dict[str, Exception] = {}

        resolved_run_args = _resolve_run_args(run_args)
        fail_fast = not resolved_run_args.allow_errors

        with ThreadPoolExecutor(max_workers=len(blocks)) as executor:
            future_to_name = {}
            block_names: List[str] = []
            for block_spec in blocks:
                if isinstance(block_spec, str):
                    name = block_spec
                elif isinstance(block_spec, type):
                    name = getattr(block_spec, "_name", block_spec.__name__)
                elif isinstance(block_spec, BlockFunction):
                    name = block_spec.name
                else:
                    name = str(block_spec)
                block_names.append(name)

                future = executor.submit(
                    self.run,
                    block_spec,
                    run_args=run_args,
                    data=data,
                    job_args=job_args,
                    **kwargs,
                )
                future_to_name[future] = name

            pending = set(future_to_name.keys())
            while pending:
                return_when = FIRST_EXCEPTION if fail_fast else ALL_COMPLETED
                done, pending = wait(
                    pending,
                    return_when=return_when,
                )

                for future in done:
                    name = future_to_name[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        errors[name] = e

                if fail_fast and errors:
                    for future in pending:
                        future.cancel()
                    break

        if errors:
            msgs = [f"  {name}: {e}" for name, e in errors.items()]
            raise RuntimeError("run_parallel failures:\n" + "\n".join(msgs))

        # Merge all overlay results so downstream steps see every sibling metric,
        # not just whichever thread happened to register last.
        # Equivalent to running the overlays sequentially.
        successful = [n for n in block_names if n in results and results[n] is not None]
        if len(successful) > 1:
            merged = results[successful[0]]
            for name in successful[1:]:
                merged = PipelineDataHandler.merge(merged, results[name], how="left")

            last_name = successful[-1]
            merged_parquet = self.rundir / last_name / "output.parquet"
            merged.write_parquet(merged_parquet)
            self._input_cache[merged_parquet] = merged

            # Deterministic manifest order: pin the last-listed block as manifest's
            # last step regardless of thread completion order.
            with self._lock:
                self.manifest.set_last_step(last_name)
                self.manifest.save()

            results[last_name] = merged

        return results

    def _configure_container_binds(self, blocks: List[BlockFunction]) -> None:
        from chaperone.block_config.compiled import CompiledBlockFunction

        for block in blocks:
            if isinstance(block, CompiledBlockFunction):
                block._set_container_bind_settings(self._container_bind_settings)

    def _run(
        self,
        blocks: List[BlockFunction],
        handler_in: PipelineDataHandler,
        run_args: RuntimeArgs,
    ) -> PipelineDataHandler:
        """Run block(s) — direct call for single non-chunked, otherwise chunked dispatch."""
        self._configure_container_binds(blocks)
        block = blocks[0]
        if block._chunked or len(blocks) > 1:
            label = (
                " → ".join(b.name for b in blocks) if len(blocks) > 1 else block.name
            )
            output_parquets = execute_chunks(blocks, handler_in, self.rundir, run_args)
            handler_out = concat_outputs(output_parquets)
            stored_blocks = [b for b in blocks if b.store != "none"]
            # Each chained block has its own step dir, so clean intermediate
            # chunk sandboxes as well as the chain tail.
            for b in stored_blocks:
                if b.store_cleanup:
                    _cleanup_chunk_sandboxes(self.rundir / b.name)
            if getattr(self, "cleanup_logs", False):
                from chaperone.engine.cleanup import consolidate_step_logs

                for b in blocks:
                    consolidate_step_logs(b.name, self.rundir)
            print_panel(f"[bold green]✓ compiled outputs for {label}[/bold green]")
            return handler_out
        with _tee_output(None):
            handler_out = safe_call(
                block, handler=handler_in, sandbox_dir=self.rundir / block.name
            )
        if getattr(self, "cleanup_logs", False):
            from chaperone.engine.cleanup import consolidate_step_logs

            consolidate_step_logs(block.name, self.rundir)
        print_panel(f"[bold cyan]✓ {block.name} complete[/bold cyan]", style="cyan")
        return handler_out

    # ─────────────────────────────────────────────────────────────────────────
    # Caching
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_fingerprint(
        blocks: List[BlockFunction],
        handler_in: Optional[PipelineDataHandler] = None,
        run_args: Optional[RuntimeArgs] = None,
    ) -> str:
        """Stable hash of block parameters, input content, and chunk layout.

        Completed chunk paths are positional (``chunk_0``, ``chunk_1``, ...),
        so changing the rows per chunk changes which inputs those paths mean.
        A campaign may also replace rows or overwrite artifacts without changing
        their count or filenames, so include the handler's content fingerprint.
        Resource-only changes such as node exclusions remain cache-neutral.
        """
        payload = [
            {
                "class": f"{block.__class__.__module__}.{block.__class__.__qualname__}",
                "config": block.model_dump(mode="json"),
            }
            for block in blocks
        ]
        block = blocks[-1]
        if handler_in is not None:
            payload.append(
                {"engine_input_content": handler_in.content_fingerprint()}
            )
        if (
            block._chunked
            and handler_in is not None
            and run_args is not None
            and run_args.job_args is not None
        ):
            rows_per_chunk = run_args.job_args.effective_rows_per_job(
                len(handler_in), default=getattr(blocks[0], "_rows_per_job", None)
            )
            payload.append(
                {
                    "engine_chunk_layout": {
                        "input_rows": len(handler_in),
                        "rows_per_chunk": rows_per_chunk,
                    }
                }
            )
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _cache_fingerprint_path(output_parquet: Path) -> Path:
        return output_parquet.parent / ".cache_fingerprint.json"

    def _write_cache_fingerprint(
        self, output_parquet: Path, fingerprint: str
    ) -> None:
        path = self._cache_fingerprint_path(output_parquet)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"schema_version": 2, "fingerprint": fingerprint}))
        temporary.replace(path)

    def _cache_fingerprint_matches(
        self, output_parquet: Path, fingerprint: str
    ) -> bool:
        path = self._cache_fingerprint_path(output_parquet)
        try:
            payload = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        return (
            payload.get("schema_version") == 2
            and payload.get("fingerprint") == fingerprint
        )

    def _invalidate_chain_cache(self, step_dirs, handler_in) -> None:
        """Discard this chain's intermediate results before staging new inputs.

        Validate the entire deletion set first: names cannot escape the run,
        and a caller cannot overwrite a step whose files are its current input.
        """
        from chaperone.data.catcif import is_catcif_path, split_catcif_path

        root = self.rundir.resolve()
        resolved = {directory.resolve() for directory in step_dirs}
        for directory in resolved:
            if directory == root or not directory.is_relative_to(root):
                raise ValueError(f"Cache step directory must be inside {root}: {directory}")
        for directory in step_dirs:
            if directory.is_symlink():
                raise ValueError(f"Cannot invalidate a symlinked cache step: {directory}")
        for step in self.manifest.steps:
            directory = (self.rundir / step.name).resolve()
            if directory not in resolved and any(
                directory.is_relative_to(parent) for parent in resolved
            ):
                raise ValueError(
                    f"Cannot invalidate cache containing another step: {step.name}. "
                    "Use non-overlapping block names."
                )
        if handler_in is not None:
            for row in handler_in._df.iter_rows(named=True):
                for structure in row.get("structures") or []:
                    for key in ("path", "store_path"):
                        value = structure.get(key)
                        if not value:
                            continue
                        physical = str(value)
                        if is_catcif_path(physical):
                            physical, _ = split_catcif_path(physical)
                        path = Path(physical).resolve()
                        if any(path.is_relative_to(directory) for directory in resolved):
                            raise ValueError(
                                f"Cannot invalidate cache containing input artifact {path}. "
                                "Use a different block name or run directory."
                            )
        for directory in step_dirs:
            if directory.exists():
                shutil.rmtree(directory)
            self._input_cache.pop(directory / "output.parquet", None)

    def _check_cache(
        self,
        blocks,
        output_parquet,
        handler_in,
        run_args,
        cache_fingerprint,
    ):
        """Check if block output is cached. Returns handler if cached, None otherwise.

        For update blocks (_creates_new_step=False), caching is detected by
        checking whether the block's metric key already exists in the input
        handler's metrics column — blocks write metrics under ``self.name``.
        """
        block = blocks[-1]
        if run_args.recompile or run_args.force_exec:
            return None

        step_dirs = [self.rundir / item.name for item in blocks]
        # Fingerprint files guard disk caches. An update metric already carried
        # by the supplied handler is part of that handler's content, not a disk
        # cache owned by this rundir, and remains an intentional passthrough.
        # A failed chain may have completed intermediate chunks but no tail
        # output. Those positional caches are guarded by the same fingerprint.
        has_disk_cache_candidate = any(
            (directory / "output.parquet").exists()
            or any(directory.glob("chunk_*/output.parquet"))
            for directory in step_dirs
        )
        if has_disk_cache_candidate and not self._cache_fingerprint_matches(
            output_parquet, cache_fingerprint
        ):
            logger.warning(
                "Invalidating %s cache because its block configuration changed "
                "or has no fingerprint metadata.",
                block.name,
            )
            self._invalidate_chain_cache(step_dirs, handler_in)
            return None

        if run_args.skip_cache:
            return None

        # Metric-based cache for update blocks
        if (
            not block._creates_new_step
            and handler_in is not None
            and handler_in.has_metric(block.name)
        ):
            print_panel(
                f"[bold blue] {block.name} metrics already present — skipped[/bold blue]",
                style="blue",
            )
            return handler_in

        if (
            output_parquet.exists()
            and not run_args.recompile
            and not run_args.skip_cache
            and not run_args.force_exec
        ):
            cached = PipelineDataHandler(output_parquet)
            if len(cached) == 0:
                logger.warning(
                    f"Cached output for {block.name} is empty — re-executing."
                )
                return None
            print_panel(
                f"[bold blue] loaded {block.name} outputs from cache[/bold blue]",
                style="blue",
            )
            return cached

        # Check for completed chunks that haven't been consolidated yet.
        # Resolve rows-per-chunk through effective_rows_per_job so a run with
        # num_chunks= enumerates the SAME chunk_N dirs that dispatch wrote —
        # otherwise the fast path never matches and a completed (possibly
        # hours-long GPU) step is re-executed on resume.
        if block._chunked and handler_in is not None:
            rows_per_chunk = run_args.job_args.effective_rows_per_job(
                len(handler_in), default=getattr(block, "_rows_per_job", None)
            )
            chunk_slices = handler_in.get_slices(rows_per_chunk)
            chunk_dirs = [
                self.rundir / block.name / f"chunk_{cid}" for cid in chunk_slices
            ]
            chunk_outputs = [d / "output.parquet" for d in chunk_dirs]
            if chunk_outputs and all(p.exists() for p in chunk_outputs):
                print_panel(
                    f"[bold blue] {block.name} cached ({len(chunk_outputs)} chunks)[/bold blue]",
                    style="blue",
                )
                handler_out = concat_outputs(chunk_outputs)
                handler_out.write_parquet(output_parquet)
                self._input_cache[output_parquet] = handler_out
                return handler_out

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Manifest helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _register_step(
        self,
        block,
        output_parquet,
        parent_step_names: list[str] | None = None,
    ):
        """Register a step in the manifest.

        ``parent_step_names`` is the list of step names whose output fed this
        block (usually one; more than one for a fan-in concat). Recorded as the
        step's ``pipeline_input_names`` so the lineage DAG is exact.
        """
        pin = list(parent_step_names) if parent_step_names else []
        self.manifest.add_step(block._name, block.name, output_parquet, pin)
        self.manifest.refresh_step_execution_summary(block.name)
        self.manifest.save()

    def _invalidate_downstream_cache(self, step_name: str) -> None:
        """Delete cached outputs for all steps downstream of *step_name*.

        Called whenever a block actually executes (cache miss) to ensure
        downstream steps don't serve stale results from a previous run.
        Uses the manifest's ordered step list: everything registered after
        the given step is considered downstream.
        """
        removed = self.manifest.invalidate_downstream(step_name)
        if not removed:
            return

        names = [s.name for s in removed]
        logger.info(
            "Invalidating %d downstream step(s) after '%s': %s",
            len(removed),
            step_name,
            ", ".join(names),
        )
        print_panel(
            f"[bold yellow]Invalidating {len(removed)} stale downstream "
            f"step(s): {', '.join(names)}[/bold yellow]",
            style="yellow",
        )

        pipeline_dir = self.rundir / ".pipeline"
        for step_info in removed:
            step_dir = self.rundir / step_info.name
            if step_dir.exists():
                shutil.rmtree(step_dir)

            parquet_path = self.rundir / step_info.parquet
            self._input_cache.pop(parquet_path, None)

            if pipeline_dir.exists():
                for pattern in [
                    f"input.{step_info.name}.parquet",
                    f"input.{step_info.name}.csv",
                    f"block.{step_info.name}.pkl",
                ]:
                    for f in pipeline_dir.glob(pattern):
                        f.unlink()

        self.manifest.save()
