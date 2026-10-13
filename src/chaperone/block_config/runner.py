"""Runner for the low-memory execvp finisher chain.

A single ``runner.py`` handles both the wait and finish phases, dispatching on
``sys.argv[1]``:

  Phase 2 (``wait``)
      Supervises ``Manifest.exec`` for the entire apptainer/SLURM wait. Loaded
      by FILE PATH (``python <path>/runner.py wait <state.json>``) so that
      ``chaperone/__init__.py`` does NOT run — which would pull in polars +
      the block registry and blow the ~16 MiB memory budget. Heavy imports
      (``submitit``, ``PipelineDataHandler``) live inside ``_phase_finish``
      and are deferred until after ``Manifest.exec`` returns.

      On success, ``_phase_finish`` is called **in-process**

  Phase 3 (``finish``)
      Loads the cached block from disk, runs ``block.compile`` + ``compress``,
      merges with the input handler (lazy-loaded from parquet), writes
      ``output.parquet``, and writes submitit's result pickle. Normally
      executed in-process after Phase 2 succeeds. Also exposed as a standalone
      ``runner.py finish <state.json>`` entry point for debugging and tests.

During normal operation, this script is invoked with "wait" which automatically calls
  finish. The distinction is here both for testing and because during "finish" the memory
  spikes considerably

When Brian originally developed this technique, 2.7GB was shaved from phase1 by implementing
  this method. Phase 2 is tiny, and then phase 3 uses 600 MB but by that point the
  underlying program has finished so that's fine

State is one JSON file (``_runner_state.json``) shared between both phases.
The block itself is NOT in the state — it is loaded from the path written by
``ChunkContext.cache_blocks`` (``.pipeline/block.<name>.pkl``), avoiding a
duplicate cloudpickle of a heavy object.
"""

# stdlib-only at module load — keep the wait-mode footprint tiny.
import importlib.util
import hashlib
import json
import logging
import os
import sys
import traceback
import types
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent  # src/chaperone/block_config/
_CHAPERONE_ROOT = _HERE.parent  # src/chaperone/


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _terminal(state: dict, *, status: str, error: dict | None = None) -> None:
    path = Path(state["terminal_path"])
    # exist_ok re-checks with a stat after EEXIST; on NFS that can come back stale and
    # raise, so concurrent jobs creating one shared directory tolerate FileExistsError.
    with suppress(FileExistsError):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(state["output_path"])
    report = {
        "schema_version": 1,
        "status": status,
        "chunk_id": state["chunk_id"],
        "step": state["block_name"],
        "blocks": [state["block_name"]],
        "block_sha256": {state["block_name"]: _sha256(Path(state["block_path"]))},
        "input_count": state.get("input_count", 0),
        "input_row_ids": state.get("input_row_ids", []),
        "output_count": state.get("output_count", 0),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_at": state.get("started_at"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": 0 if status == "success" else 1,
        "error": error,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


# ── Module loading ───────────────────────────────────────────────────────


def _load_standalone(module_name: str, path: Path) -> types.ModuleType:
    """Import a single .py file WITHOUT triggering its parent package __init__.

    Critical for keeping wait mode's resident set tiny — if we let Python
    walk up the package tree, ``chaperone/__init__.py`` runs and pulls in
    polars + the block registry + everything else.

    The module must be registered in ``sys.modules`` before ``exec_module``
    runs, otherwise decorators like ``@dataclass`` fail when they try to
    introspect ``sys.modules.get(cls.__module__)``.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Chunk-log tee ────────────────────────────────────────────────────────
# Restores the legacy per-chunk log capture that ``chunking._tee_output``
# provides for the in-process path. After ``os.execvp`` from the harness,
# that context manager is gone — without this tee, all of Manifest.exec's
# subprocess output (and runner.py's own INFO lines) would only land in
# submitit's job-level *.out/*.err. Tailing the chunk log file under
# ``{rundir}/logs/<block>/chunk_<id>/log.txt`` would show nothing.
#
# The ``_TeeWriter`` class itself lives in the stdlib-only ``_log_tee``
# module so both this file (in wait mode and finish mode) and
# ``chunking._tee_output`` can share a single implementation. In wait mode
# we load it via importlib so no package init runs.


def _get_tee_writer_class():
    """Return ``_TeeWriter`` from the stdlib-only ``_log_tee`` module.

    Always loaded by FILE PATH via importlib — a normal
    ``from chaperone.block_config._log_tee import _TeeWriter`` would trigger
    ``chaperone/__init__.py`` (polars + block registry) in wait mode, which
    is exactly the bloat we are avoiding. ``_log_tee`` is tiny and
    stdlib-only, so standalone-loading it in finish mode too is harmless and
    keeps this single-path.
    """
    mod = _load_standalone("_log_tee_standalone", _HERE / "_log_tee.py")
    return mod._TeeWriter


def _install_chunk_log_tee(log_path: Path, *, role: str, truncate: bool) -> None:
    """Redirect sys.stdout/sys.stderr through a tee that also writes
    ``log_path``, AND attach a dedicated ``logging.FileHandler`` to the
    root logger so logger.* calls land in the same file.

    Args:
        log_path: Per-chunk log file (e.g. ``{rundir}/logs/<block>/chunk_<id>.log``).
        role: ``"wait"`` or ``"finish"`` — embedded in the log formatter so
            lines from each phase are distinguishable in the combined file.
        truncate: If True, truncate ``log_path`` before opening (wait mode,
            so each chunk attempt starts fresh). If False, append (finish
            mode, so runner_wait's earlier subprocess output is preserved).

    The dual mechanism (Tee + FileHandler) is necessary because
    ``logging.StreamHandler`` binds its stream at handler-creation time. If
    anything in the import chain already configured logging, replacing
    ``sys.stderr`` after that point would NOT redirect existing handlers;
    the FileHandler we add here bypasses that problem.

    CRITICAL: both the Tee's file object and the logging FileHandler must
    use append mode — having two file descriptors at different positions on
    the same file causes one writer to silently overwrite bytes from the
    other. We do a one-shot truncate (``open("w")``) up front in wait mode
    instead.

    Best-effort: if anything goes wrong opening the file we leave stdio
    alone and let the phase proceed without per-chunk log capture, rather
    than aborting the whole job.
    """
    try:
        with suppress(FileExistsError):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            open(log_path, "w").close()
        log_file = open(log_path, "a", buffering=1)
    except OSError:
        return

    tee_cls = _get_tee_writer_class()
    sys.stdout = tee_cls(sys.__stdout__, log_file)
    sys.stderr = tee_cls(sys.__stderr__, log_file)

    handler = logging.FileHandler(str(log_path), mode="a")
    handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s [runner_{role}:%(process)d] %(levelname)s %(message)s"
        )
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


# ── Submitit result pickle (shared) ───────────────────────────────────────


def _write_result_pickle(result_pickle_path: str, payload: tuple) -> None:
    """Write submitit's result pickle in the same format process_job uses.

    Mirrors submitit/core/submission.py — cloudpickle dump of
    ``("success", result)`` or ``("error", traceback_str)`` via
    ``temporary_save_path`` so partial writes don't corrupt the file.

    Imports submitit lazily so wait mode's happy path never pays for it.
    """
    from submitit.core import utils as submitit_utils

    with submitit_utils.temporary_save_path(result_pickle_path) as tmppath:
        submitit_utils.cloudpickle_dump(payload, tmppath)


# ── Phase 2: wait ────────────────────────────────────────────────────────


def _phase_wait(state: dict) -> int:
    """Supervise Manifest.exec, then execvp into Phase 3 on success.

    On Phase-2 failure (Manifest.exec raises, or execvp into finish itself
    fails), write submitit's error pickle and exit — there is nothing for
    Phase 3 to do in that case.
    """
    sandbox_dir = Path(state["sandbox_dir"])
    step_dir = Path(state["step_dir"])
    chunk_id = state["chunk_id"]
    block_name = state["block_name"]
    result_pickle = state["result_pickle"]
    # state_path = state["_state_path"]  # for the execvp handoff

    logger.info(
        "runner_wait: block=%s chunk=%s sandbox=%s",
        block_name,
        chunk_id,
        sandbox_dir,
    )

    try:
        # Standalone-load Manifest and track_execution: importing them via
        # ``from chaperone.X import Y`` would trigger ``chaperone/__init__.py``
        # (which pulls in polars + block registry). Loading the .py files
        # directly via importlib bypasses the package machinery — these two
        # source files are stdlib-only at module load and stay <10 MiB
        # combined.
        manifest_mod = _load_standalone("_manifest_standalone", _HERE / "manifest.py")
        tracking_mod = _load_standalone(
            "_execution_tracking_standalone", _CHAPERONE_ROOT / "execution_tracking.py"
        )
        Manifest = manifest_mod.Manifest
        track_execution = tracking_mod.track_execution

        manifest = Manifest.read(sandbox_dir)
        with track_execution(
            step_dir=step_dir,
            sandbox_dir=sandbox_dir,
            block_name=block_name,
            source="compiled_block_finisher",
            chunk_id=chunk_id,
            command=manifest.command_str or manifest.command,
        ):
            # Don't pass log_path — our sys.stdout tee already captures
            # Manifest.exec's per-line writes. Passing log_path would open
            # the log a second time in "w" mode and clobber what we wrote.
            Manifest.exec(sandbox_dir)
    except Exception:
        tb = traceback.format_exc()
        logger.error("runner_wait: Manifest.exec failed:\n%s", tb)
        _terminal(
            state,
            status="error",
            error={
                "type": "ManifestExecutionError",
                "message": tb.splitlines()[-1] if tb else "manifest failed",
                "traceback": tb[-12000:],
            },
        )
        try:
            _write_result_pickle(result_pickle, ("error", tb))
        except Exception as dump_exc:  # noqa: BLE001
            logger.error("runner_wait: could not write error pickle: %s", dump_exc)
        return 1

    # Phase 2 done. Phase 3 runs in-process: the ~16 MiB waiter footprint is
    # negligible alongside Phase 3's hundreds of MiB, so replacing the process
    # image via execvp buys nothing.
    logger.info("runner_wait: Manifest.exec done; running finish in-process")
    return _phase_finish(state)


# ── Phase 3: finish ──────────────────────────────────────────────────────


def _phase_finish(state: dict) -> int:
    """Load the cached block, run compile+merge, write output.parquet + result pickle.

    All heavy imports live inside this function so module load stays light
    for Phase 2.
    """
    sandbox_dir = Path(state["sandbox_dir"])
    chunk_id = state["chunk_id"]
    output_path = Path(state["output_path"])
    block_path = state["block_path"]
    input_handler_path = state["input_handler_path"]
    result_pickle = state["result_pickle"]
    # step_dir kept for parity / future tracking hooks
    _ = state.get("step_dir")

    logger.info("runner_finish: loading block from %s", block_path)

    try:
        # Heavy imports deferred to the body so Phase 2 never pays for them.
        import cloudpickle
        from chaperone.data import PipelineDataHandler

        with open(block_path, "rb") as f:
            block = cloudpickle.load(f)

        logger.info(
            "runner_finish: block=%s chunk=%s sandbox=%s output=%s",
            block.name,
            chunk_id,
            sandbox_dir,
            output_path,
        )

        output = block.compile(str(sandbox_dir))
        output = block.compress(output, str(sandbox_dir))
        # Lazy-load the input handler from disk; deferred past Phase 2 to
        # keep the supervisor's RSS low during the long subprocess wait
        # (see stage_then_exec_finisher in compiled.py).
        if block._requires_input:
            input_handler = PipelineDataHandler(input_handler_path)
            handler = PipelineDataHandler.merge(input_handler, output)
        else:
            # The persisted input for a source block contains scheduling-only
            # rows created by init_handler(); compiled output is authoritative.
            handler = output

        # Persist the merged handler at the same path execute_chunk_job
        # would have written it (chunk_dir/output.parquet).
        with suppress(FileExistsError):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.stem}.{os.getpid()}.tmp.parquet"
        )
        handler.write_parquet(temporary)
        os.replace(temporary, output_path)
        state["output_count"] = len(handler)

        _write_result_pickle(result_pickle, ("success", output_path))
        _terminal(state, status="success")
        logger.info("runner_finish: success; output.parquet at %s", output_path)
        return 0

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("runner_finish: failed: %s\n%s", exc, tb)
        _terminal(
            state,
            status="error",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb[-12000:],
            },
        )
        try:
            _write_result_pickle(result_pickle, ("error", tb))
        except Exception as dump_exc:  # noqa: BLE001
            logger.error("runner_finish: could not write error pickle: %s", dump_exc)
        return 1


# ── Entry point ──────────────────────────────────────────────────────────


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("wait", "finish"):
        print(
            f"usage: {sys.argv[0]} {{wait|finish}} <state_json>",
            file=sys.stderr,
        )
        return 2

    mode = sys.argv[1]
    state_path = sys.argv[2]

    with open(state_path) as f:
        state = json.load(f)
    state["_state_path"] = state_path  # so wait can pass the same file to finish

    log_path = Path(state["log_path"])
    # Truncate the chunk log on wait entry so each attempt starts fresh;
    # finish appends to preserve what wait wrote.
    _install_chunk_log_tee(log_path, role=mode, truncate=(mode == "wait"))

    return _phase_wait(state) if mode == "wait" else _phase_finish(state)


if __name__ == "__main__":
    sys.exit(main())
