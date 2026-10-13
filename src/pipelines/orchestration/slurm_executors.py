"""SLURM-side execution functions.

These functions are serialized via submitit's pickling and executed on SLURM
compute nodes. They are NOT meant to be called directly - they are entry
points for SLURM jobs.

Functions are cloudpickled (by value) before submission to handle cases where
the original function isn't importable (e.g., defined in notebooks or test files).
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from typing import Any


# Environment variables that should not be forwarded to SLURM jobs because they
# conflict with libraries inside containers (e.g. foundry's environs reads DEBUG
# as a boolean, but the host may set DEBUG=release).
_ENV_DENYLIST = frozenset({"DEBUG"})

# Vars that must reflect the WORKER's allocation, not the submitter's.
# Stripped from the env dict shipped to the worker so they don't overwrite
# the worker's correctly-set values, but NOT scrubbed from the worker's
# os.environ (so the worker keeps its own).
_FORWARD_DENYLIST = (
    frozenset(
        {
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "SLURM_JOB_ID",
            "SLURM_JOBID",
            "SLURM_STEP_ID",
            "SLURM_STEPID",
            "SLURM_LOCALID",
            "SLURM_PROCID",
            "SLURM_NODEID",
            "SLURM_NODELIST",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "SLURM_GPUS",
            "SLURM_GPUS_ON_NODE",
            "SLURM_JOB_CPUS_PER_NODE",
            "SLURM_NTASKS",
            "SLURM_JOB_NUM_NODES",
        }
    )
    | _ENV_DENYLIST
)


def _sanitize_env(env: dict[str, str]) -> dict[str, str]:
    """Remove environment variables that cause issues inside containers.

    Strips both `_ENV_DENYLIST` (container-conflict vars like DEBUG) and
    `_FORWARD_DENYLIST` (allocation-local vars like CUDA_VISIBLE_DEVICES,
    which must reflect the worker's allocation, not the submitter's).
    """
    return {
        k: v
        for k, v in env.items()
        if k not in _FORWARD_DENYLIST
        and not k.startswith(("APPTAINER_", "SINGULARITY_"))
    }


def cloudpickle_fn(fn: Any) -> bytes:
    """Cloudpickle a function, forcing serialization by value.

    Temporarily registers the function's module for by-value pickling
    so that cloudpickle embeds the function's bytecode instead of just
    storing a module reference. This ensures the function can be
    unpickled in environments where the original module isn't importable
    (e.g., submitit subprocesses, SLURM nodes, notebooks).
    """
    import cloudpickle

    module = sys.modules.get(getattr(fn, "__module__", None))
    registered = False
    if module is not None:
        try:
            cloudpickle.register_pickle_by_value(module)
            registered = True
        except ValueError:
            pass  # Already registered or can't register

    try:
        return cloudpickle.dumps(fn)
    finally:
        if registered:
            try:
                cloudpickle.unregister_pickle_by_value(module)
            except ValueError:
                pass


def _scrub_env_denylist() -> None:
    """Remove denylisted variables from the process environment.

    SLURM inherits the full submitter environment (--export=ALL), so
    denylisted vars like DEBUG may already be present even though they
    were stripped from the explicit env dict.
    """
    for key in _ENV_DENYLIST:
        os.environ.pop(key, None)


def _run_fn_in_slurm(
    pickled_fn: bytes,
    parameters: dict[str, Any],
    env: dict[str, str] | None = None,
) -> Any:
    """Run a cloudpickled function inside a SLURM job.

    This function is submitted to SLURM and executes the target function
    directly with the given parameters.

    Args:
        pickled_fn: Cloudpickled function bytes.
        parameters: Keyword arguments to pass to the function.
        env: Environment variables to set before execution.
    """
    os.environ.update(env or {})
    _scrub_env_denylist()
    fn = pickle.loads(pickled_fn)
    return fn(**parameters)


def _run_batch_in_slurm(
    pickled_fn: bytes,
    batch_items: list[Any],
    param_name: str,
    static_params: dict[str, Any],
    env: dict[str, str] | None = None,
) -> list[Any]:
    """Execute a batch of items inside a SLURM job.

    Calls fn once per item, passing the item as param_name along with
    static_params.

    Args:
        pickled_fn: Cloudpickled function bytes.
        batch_items: List of items to process.
        param_name: Parameter name for each item.
        static_params: Static parameters passed to every call.
        env: Environment variables to set.

    Returns:
        List of results from processing each item.
    """
    logger = logging.getLogger("pipelines.slurm.batch")

    os.environ.update(env or {})
    _scrub_env_denylist()
    fn = pickle.loads(pickled_fn)

    results = []
    for i, item in enumerate(batch_items):
        try:
            logger.debug(
                f"Processing item {i + 1}/{len(batch_items)}: {_item_repr(item)}"
            )
            call_params = {param_name: item}
            call_params.update(static_params)
            result = fn(**call_params)
            results.append(result)
            logger.debug(f"Item {i + 1}/{len(batch_items)} completed successfully")
        except Exception as e:
            logger.error(
                f"Item {i + 1}/{len(batch_items)} failed: {type(e).__name__}: {e}\n"
                f"Item details: {_item_repr(item)}"
            )
            raise  # Fail entire batch

    return results


def _item_repr(item: Any, max_len: int = 100) -> str:
    """Safe string representation of an item for logging."""
    try:
        r = repr(item)
        return r if len(r) <= max_len else r[:max_len] + "..."
    except Exception:
        return f"<{type(item).__name__}>"
