"""Cleanup helpers for consolidating step directories."""

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


_CHUNK_LOG_RE = re.compile(r"chunk_(\d+)\.log$")


def _chunk_log_sort_key(p: Path) -> tuple[int, str]:
    m = _CHUNK_LOG_RE.search(p.name)
    return (int(m.group(1)) if m else 1 << 30, p.name)


def consolidate_step_logs(step_name: str, rundir: Path) -> None:
    """Cat per-chunk logs into a single step log, then remove the step log dir.

    Concatenates ``logs/<step>/chunk_*.log`` into ``logs/<step>.log`` (in chunk
    order), removes ``logs/<step>/slurm/``, and removes the now-empty
    ``logs/<step>/`` directory. Idempotent: no-op if the dir is missing.
    """
    log_dir = rundir / "logs" / step_name
    if not log_dir.is_dir():
        return

    chunk_logs = sorted(log_dir.glob("chunk_*.log"), key=_chunk_log_sort_key)
    if chunk_logs:
        out_path = rundir / "logs" / f"{step_name}.log"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as out:
            for p in chunk_logs:
                out.write(f"=== {p.name} ===\n")
                try:
                    out.write(p.read_text())
                except Exception as e:
                    out.write(f"<failed to read {p}: {e}>\n")
                out.write("\n")
        logger.debug(f"Consolidated {len(chunk_logs)} chunk log(s) → {out_path}")

    shutil.rmtree(log_dir, ignore_errors=True)
    logger.debug(f"Removed step log dir: {log_dir}")


def _safe_block_name(name: str) -> str:
    """Mirror ``ChunkContext.cache_blocks`` filename sanitization."""
    return name.replace("/", "--")


def remove_pipeline_scratch(rundir: Path, block_names: list[str]) -> None:
    """Delete a step's ``.pipeline/`` scratch once it has compiled.

    The chunk input handler (``input.<step>.parquet``) and the cached block
    pickle (``block.<name>.pkl``) are the inter-process channel that hands each
    chunk worker its row-slice + block during execution. They are load-bearing
    *while the step runs*, but nothing references them afterward — the step's
    ``output.parquet`` and the manifest carry everything downstream needs. So we
    remove them as soon as the step finalizes, leaving no ``.pipeline/`` clutter.

    This only touches ``.pipeline/`` — never the step's ``chunk_*`` sandboxes,
    which may still hold the structure files the output parquet points at.
    """
    pipeline_dir = rundir / ".pipeline"
    if not pipeline_dir.exists() or not block_names:
        return

    # The chunk input parquet is keyed on the chain's first block name.
    patterns = [
        f"input.{block_names[0]}.parquet",
        f"input.{block_names[0]}.csv",
    ]
    patterns += [f"block.{_safe_block_name(n)}.pkl" for n in block_names]

    for pattern in patterns:
        for f in pipeline_dir.glob(pattern):
            try:
                f.unlink()
                logger.debug(f"Removed pipeline scratch: {f}")
            except OSError:
                pass

    # Drop the directory entirely once it is empty (last step cleaned).
    try:
        next(pipeline_dir.iterdir())
    except StopIteration:
        shutil.rmtree(pipeline_dir, ignore_errors=True)
    except OSError:
        pass
