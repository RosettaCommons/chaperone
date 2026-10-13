"""Argument resolution helpers for ExecutionEngine.

Resolves blocks, RuntimeArgs, JobArgs, execution mode, and task runners
from the various input formats (str, dict, instances, CLI overrides, etc.).
"""

import logging
from dataclasses import asdict, dataclass, fields as dc_fields, replace
from typing import List, Literal

from chaperone.block_config.base import BlockFunction
from chaperone.common import get_console
from chaperone.data import PipelineDataHandler
from chaperone.orchestration import JobArgs, SlurmTaskRunner

logger = logging.getLogger(__name__)
console = get_console()


ExecutionMode = Literal["local", "slurm"]


@dataclass
class RuntimeArgs:
    """Runtime configuration for block execution.

    ``local`` runs chunks in-process or across visible local GPUs. ``slurm``
    submits chunk arrays. Preparation and final aggregation remain local.

    Cache override flags:
        - force_exec: Bypasses all caches (step + chunk) and forces full
          re-execution (prepare, execute, compile) while preserving the
          current execution mode.
        - recompile: Forces recompilation but also forces mode="local".
        - skip_cache: Skips step-level cache only.
        - skip_chunk_cache: Skips chunk-level cache only.
    """

    mode: ExecutionMode = "slurm"
    recompile: bool = False
    allow_errors: bool = False
    force_exec: bool = False
    skip_cache: bool = False
    skip_chunk_cache: bool = False
    use_local_gpus: bool = True
    max_retries: int = 2
    max_workers: int | None = None
    job_args: JobArgs | None = None

    def __post_init__(self):
        """Apply global settings overrides and handle deprecation."""
        from chaperone import get_settings

        if get_settings().ENGINE_RECOMPILE:
            self.recompile = True
        if self.recompile and self.mode != "local":
            self.mode = "local"

    @property
    def is_local(self) -> bool:
        """Check if running in local mode (for backward compatibility)."""
        return self.mode == "local"

    def for_attempt(self, is_last: bool, excluded_nodes: set[str]) -> "RuntimeArgs":
        """Build RuntimeArgs for a retry attempt.

        Non-final attempts set allow_errors=True so failed chunks return None.
        Accumulated excluded_nodes are merged into job_args.
        """
        args = self if is_last else replace(self, allow_errors=True)
        if excluded_nodes and self.job_args:
            args = replace(
                args, job_args=self.job_args.with_excluded_nodes(excluded_nodes)
            )
        return args


# ─────────────────────────────────────────────────────────────────────────────
# Block resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_block(block, params: dict, namespace: str | None) -> List[BlockFunction]:
    """Resolve block(s) from string/type/instance/list and apply namespace.

    Always returns a list of BlockFunction instances. For a single block input,
    returns a one-element list. For a list input, resolves each element.
    kwargs (params) are only supported for single-block inputs.
    """
    if isinstance(block, list):
        if params:
            raise ValueError(
                f"Cannot pass kwargs to chain execution. "
                f"Pre-instantiate blocks or use CLI overrides. Got: {set(params)}"
            )
        return [_resolve_single_block(b, {}, namespace) for b in block]
    return [_resolve_single_block(block, params, namespace)]


def _resolve_single_block(block, params: dict, namespace: str | None) -> BlockFunction:
    """Resolve a single block from string/type/instance and apply namespace.

    Priority order (highest wins):
        1. CLI flags (``--block.field=value``)
        2. CLI config (``--config=overrides.yaml``)
        3. Script kwargs (``engine.run("block", field=value)``)
        4. Script YAML defaults (``init_cli(defaults_yaml=...)``)
        5. Pydantic field defaults
    """
    from chaperone.block_config.base import RESERVED_FIELDS

    conflicts = RESERVED_FIELDS & set(params)
    if conflicts:
        raise ValueError(
            f"engine.run() received reserved kwargs: {conflicts}. "
            f"These are runtime/engine parameters, not block fields. "
            f"Pass them via run_args=RuntimeArgs(...) or job_args=JobArgs(...) instead."
        )

    if isinstance(block, str):
        from blocks import Blocks
        from chaperone import (
            get_block_overrides,
            get_cli_config_defaults,
            get_yaml_defaults,
        )

        script_yaml = get_yaml_defaults(block)
        cli_config = get_cli_config_defaults(block)
        cli = get_block_overrides(block)
        block = Blocks[block](**{**script_yaml, **params, **cli_config, **cli})
    elif isinstance(block, type):
        from chaperone import (
            get_block_overrides,
            get_cli_config_defaults,
            get_yaml_defaults,
        )

        block_name = getattr(block, "_name", "")
        script_yaml = get_yaml_defaults(block_name)
        cli_config = get_cli_config_defaults(block_name)
        cli = get_block_overrides(block_name)
        block = block(**{**script_yaml, **params, **cli_config, **cli})
    else:
        # Pre-instantiated block — apply CLI config and CLI flag overrides on top
        if params:
            raise ValueError(f"Cannot pass kwargs to instantiated block: {params}")
        from chaperone import get_block_overrides, get_cli_config_defaults

        # Resolve overrides by block type name (_name), then layer instance-name
        # overrides on top so that e.g. --config with "rfd3_ame1:" keys can
        # target a specific instance while "rfd3:" still applies to all RFD3s.
        type_name = getattr(block, "_name", "")
        instance_name = getattr(block, "name", type_name)

        cli_config = get_cli_config_defaults(type_name)
        cli = get_block_overrides(type_name)

        if instance_name != type_name:
            cli_config = {**cli_config, **get_cli_config_defaults(instance_name)}
            cli = {**cli, **get_block_overrides(instance_name)}

        overrides = {**cli_config, **cli}
        if overrides:
            block = block.model_copy(update=overrides)

    if namespace and not block.name.startswith(namespace):
        block = block.model_copy(update={"name": f"{namespace}/{block.name}"})
    return block


def _resolve_input_data(
    data,
    block: BlockFunction,
) -> PipelineDataHandler | None:
    """Resolve the input handler for a block.

    Args:
        data: Explicit input data (DataFrame, handler, path, or None).
        block: The block being executed.

    Returns:
        PipelineDataHandler or None for source blocks.

    Raises:
        ValueError: If block requires input but none was provided.
    """
    if data is not None:
        return PipelineDataHandler.load(
            data, allow_sequences=block._accepts_sequence_input
        )
    if block._requires_input:
        raise ValueError(
            f"Block '{block.name}' requires explicit input data. "
            f"Pass data=... to engine.run()."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeArgs resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_run_args(
    run_args: RuntimeArgs | dict | None,
    block_name: str | None = None,
) -> RuntimeArgs:
    """Resolve run_args from dict or use defaults.

    When the user has not explicitly set a mode (run_args is None or a dict
    without a ``"mode"`` key), the default mode is derived from
    ``COMPUTE_BACKEND`` via :func:`_resolve_default_mode`.

    Per-block CLI flags (e.g. ``--rmsd.force_exec``) override global flags.
    """
    from chaperone import get_settings

    settings = get_settings()

    needs_default_mode = False

    if run_args is None:
        resolved = RuntimeArgs()
        needs_default_mode = True
    elif isinstance(run_args, dict):
        run_args = run_args.copy()
        needs_default_mode = "mode" not in run_args
        if "job_args" in run_args and isinstance(run_args["job_args"], dict):
            run_args["job_args"] = JobArgs(**run_args["job_args"])
        resolved = RuntimeArgs(**run_args)
    else:
        resolved = replace(run_args)

    from chaperone import get_runtime_overrides

    cli_rt = get_runtime_overrides()
    if "recompile" in cli_rt:
        resolved.recompile = cli_rt["recompile"]
    if "force_exec" in cli_rt:
        resolved.force_exec = cli_rt["force_exec"]
    if "skip_cache" in cli_rt:
        resolved.skip_cache = cli_rt["skip_cache"]
    if "local" in cli_rt and cli_rt["local"]:
        resolved.mode = "local"
        needs_default_mode = False
    if "use_local_gpus" in cli_rt:
        resolved.use_local_gpus = cli_rt["use_local_gpus"]

    if block_name is not None:
        _apply_per_block_runtime_overrides(resolved, block_name)

    if needs_default_mode and not resolved.recompile:
        resolved.mode = _resolve_default_mode(
            settings.PIPELINES_COMPUTE_BACKEND, use_local_gpus=resolved.use_local_gpus
        )

    return resolved


def _apply_per_block_runtime_overrides(resolved: RuntimeArgs, block_name: str) -> None:
    """Apply per-block CLI flags like --rmsd.force_exec to RuntimeArgs."""
    try:
        from absl import flags

        FLAGS = flags.FLAGS
        for attr in ("force_exec", "recompile"):
            flag_name = f"{block_name}.{attr}"
            if hasattr(FLAGS, flag_name) and FLAGS[flag_name].present:
                setattr(resolved, attr, FLAGS[flag_name].value)
    except ImportError:
        pass


def _resolve_default_mode(backend: str, use_local_gpus: bool = True) -> ExecutionMode:
    """Resolve a backend, preferring local GPUs and then SLURM in auto mode."""
    import shutil

    if backend == "local":
        return "local"
    if backend == "slurm":
        return "slurm"
    if use_local_gpus:
        from chaperone.orchestration.gpu_local_taskrunner import detect_cuda_gpus

        if detect_cuda_gpus():
            return "local"
    # sbatch is what submission needs; sinfo is only an inspection tool and is
    # not always exposed to a container. Testing sinfo alone made a worker with
    # a perfectly good sbatch fall through to local execution, and agents had
    # to force Slurm by hand.
    if shutil.which("sbatch") or shutil.which("sinfo"):
        return "slurm"
    return "local"


# ─────────────────────────────────────────────────────────────────────────────
# JobArgs resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_job_args(
    job_args: JobArgs | dict | None, block: BlockFunction, run_args: RuntimeArgs
) -> JobArgs | None:
    """Merge user job_args with block's resource defaults.

    If a local CUDA GPU is detected and use_local_gpus=True (default),
    automatically routes to local execution instead of SLURM.
    Set use_local_gpus=False to force SLURM submission even when local GPUs exist.

    Args:
        job_args: User-provided job args or None
        block: The block being executed
        run_args: Runtime arguments (checked for use_local_gpus)
    """
    from chaperone.orchestration.gpu_local_taskrunner import detect_cuda_gpus

    block_resources = block.get_resources()  # JobArgs | None

    if block_resources is None:
        if job_args is None:
            if block._chunked:
                return JobArgs(job_name=block.name, rows_per_job=block._rows_per_job)
            return None
        elif isinstance(job_args, dict):
            # Campaigns commonly reuse one resource config across blocks.
            # Never leak generated per-block defaults back into that shared
            # caller-owned mapping.
            values = dict(job_args)
            values.setdefault("job_name", block.name)
            resolved_no_resources = JobArgs(**values)
            if resolved_no_resources.rows_per_job is None:
                resolved_no_resources.rows_per_job = block._rows_per_job
            return resolved_no_resources
        else:
            # Likewise, fill defaults on a copy of a shared JobArgs instance.
            # Mutating it would mislabel every later Slurm job as the first
            # block in broker and audit metadata.
            return replace(
                job_args,
                job_name=job_args.job_name or block.name,
                rows_per_job=(
                    job_args.rows_per_job
                    if job_args.rows_per_job is not None
                    else block._rows_per_job
                ),
            )

    defaults = asdict(block_resources)
    defaults["job_name"] = block.name

    # Apply global --partition flag to block defaults (only for GPU blocks)
    try:
        from absl import flags as absl_flags

        cli_partition = getattr(absl_flags.FLAGS, "partition", None)
        if cli_partition is not None and absl_flags.FLAGS.partition is not None:
            # Only override partition for blocks that already target a GPU partition;
            # CPU-only blocks (e.g. MPNN) should keep their default.
            if (
                "gpu" in defaults["partition"].lower()
                or defaults["partition"].lower() == "local"
            ):
                defaults["partition"] = absl_flags.FLAGS.partition
                # When --partition overrides, strip the GPU model from gres so
                # the job can run on any GPU in the target partition.  E.g.
                # "gpu:a4000:1" → "gpu:1".  An explicit --gres still wins.
                if defaults.get("gres") and ":" in (defaults["gres"] or ""):
                    parts = defaults["gres"].split(":")
                    if len(parts) == 3:  # "gpu:a4000:1" → "gpu:1"
                        defaults["gres"] = f"{parts[0]}:{parts[2]}"
        cli_gres = getattr(absl_flags.FLAGS, "gres", None)
        if cli_gres is not None and absl_flags.FLAGS.gres is not None:
            # Only GPU blocks take a gres. Applying it to a CPU-partition block
            # (e.g. MPNN) yields "--partition=cpu --gres=gpu:...", which SLURM
            # rejects with "Requested node configuration is not available".
            if "gpu" in defaults["partition"].lower():
                defaults["gres"] = absl_flags.FLAGS.gres
    except (ImportError, absl_flags.UnparsedFlagAccessError):
        pass

    if job_args is None:
        resolved = JobArgs(**defaults)
    elif isinstance(job_args, dict):
        resolved = JobArgs(**{**defaults, **job_args})
    else:
        # JobArgs instance — extract only user-set overrides, merge with defaults
        overrides = {
            f.name: getattr(job_args, f.name)
            for f in dc_fields(job_args)
            if getattr(job_args, f.name) != f.default
        }
        resolved = JobArgs(**{**defaults, **overrides})

    # Documented priority is "CLI flags > script kwargs", but the merges above
    # let an explicit ``job_args`` dict/instance shadow --partition/--gres. Only
    # flags actually present on the command line re-assert themselves here, so a
    # script's job_args still beats the flags' None defaults.
    try:
        from absl import flags as absl_flags

        for flag_name in ("partition", "gres"):
            flag = (
                absl_flags.FLAGS[flag_name] if flag_name in absl_flags.FLAGS else None
            )
            if flag is None or not flag.present or flag.value is None:
                continue
            # Never drag a CPU-only block (e.g. MPNN) onto a GPU partition.
            if flag_name == "partition" and "gpu" not in defaults["partition"].lower():
                continue
            if flag_name == "gres" and "gpu" not in resolved.partition.lower():
                continue
            setattr(resolved, flag_name, flag.value)
    except (ImportError, KeyError, absl_flags.UnparsedFlagAccessError):
        pass

    if resolved.rows_per_job is None:
        resolved.rows_per_job = block._rows_per_job

    if resolved.partition.lower() == "local":
        if run_args.mode != "local":
            logger.info("partition='local' — forcing mode='local'")
            run_args.mode = "local"
        return resolved

    if "gpu" in resolved.partition.lower():
        if run_args.use_local_gpus:
            local_gpus = detect_cuda_gpus()
            if local_gpus:
                console.print(
                    f"[cyan]LOCAL[/cyan] Detected {len(local_gpus)} local GPU(s), "
                    f"routing {block.name} to local execution"
                )
                resolved = JobArgs(
                    partition="local",
                    mem_gb=resolved.mem_gb,
                    time_limit=resolved.time_limit,
                    gres=resolved.gres,
                    constraint=resolved.constraint,
                    cpus_per_task=resolved.cpus_per_task,
                    job_name=resolved.job_name,
                    rows_per_job=resolved.rows_per_job,
                    num_chunks=resolved.num_chunks,
                    exclude=resolved.exclude,
                    exclude_nodes=resolved.exclude_nodes,
                    nice=resolved.nice,
                    slurm_array_parallelism=resolved.slurm_array_parallelism,
                )
                run_args.mode = "local"

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Task runner creation
# ─────────────────────────────────────────────────────────────────────────────


def _create_runner(run_args: RuntimeArgs, log_folder: str):
    """Create task runner based on execution mode.

    Local mode uses a GPU-aware runner when devices are visible and a thread
    runner otherwise. Slurm mode uses job arrays.
    """
    from chaperone.orchestration.thread_runner import ThreadPoolRunner

    if run_args.mode == "local":
        if run_args.use_local_gpus:
            from chaperone.orchestration.gpu_local_taskrunner import (
                GPULocalTaskRunner,
                detect_cuda_gpus,
            )

            gpu_ids = detect_cuda_gpus()
            if gpu_ids:
                logger.info(
                    f"Detected {len(gpu_ids)} CUDA GPUs, using GPULocalTaskRunner"
                )
                return GPULocalTaskRunner(gpu_ids=gpu_ids)
        max_workers = run_args.max_workers or 4
        logger.debug(f"Using ThreadPoolRunner with max_workers={max_workers}")
        return ThreadPoolRunner(max_workers=max_workers)
    return SlurmTaskRunner(
        job_args=run_args.job_args,
        log_folder=log_folder,
        slurm_array_parallelism=run_args.max_workers,
    )
