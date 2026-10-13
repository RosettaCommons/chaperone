#!/usr/bin/env python3
"""Controller-owned submission, transcript, and compute audit metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


_SINGLE_CHAIN_TASKS = {
    "biotin_binder_design",
    "dna_binder",
    "nanoparticle_c3",
    "peptide_binder",
    "ppi_pdl1",
    "rna_binder",
}
_ENZYME_TASKS = {"enzyme_glycosidase", "enzyme_protease"}
_TERMINAL_EXECUTION_STATUSES = {"succeeded", "failed"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _submission_contract(
    designs: Path, task_name: str, required_n: int
) -> dict[str, object]:
    """Validate only the universal controller contract.

    Scientific sequence, motif, fixed-region, uniqueness, and diversity checks
    remain owned by the task scorer.  The controller checks the packaging
    boundary: a readable CSV, the requested row count, and task-essential
    columns only.
    """

    errors: list[str] = []
    columns: list[str] = []
    row_count = 0
    try:
        with designs.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            if not columns:
                errors.append("submission CSV has no header")
            row_count = sum(1 for _ in reader)
    except (OSError, csv.Error, UnicodeError) as error:
        errors.append(f"submission CSV is unreadable: {error}")

    column_set = set(columns)
    if len(column_set) != len(columns):
        errors.append("submission CSV contains duplicate column names")

    if task_name == "antibody_cdr":
        required = {"sequence_H", "sequence_L"}
        allowed = required | {"design_id"}
        missing = sorted(required - column_set)
        if missing:
            errors.append(f"missing required columns: {missing}")
    else:
        sequence_columns = {"sequence", "sequence_A"}
        present = sequence_columns & column_set
        allowed = sequence_columns | {"design_id"}
        if task_name in _ENZYME_TASKS:
            allowed.add("motif_index_map")
            if "motif_index_map" not in column_set:
                errors.append("missing required column: motif_index_map")
        if not present:
            errors.append("missing required sequence column: sequence")
        elif len(present) > 1:
            errors.append("supply either sequence or sequence_A, not both")

    unexpected = sorted(column_set - allowed)
    if unexpected:
        errors.append(f"unexpected columns: {unexpected}")
    if row_count != required_n:
        errors.append(f"expected {required_n} designs, got {row_count}")

    return {
        "path": str(designs.absolute()),
        "sha256": _sha256(designs) if designs.is_file() else None,
        "bytes": designs.stat().st_size if designs.is_file() else None,
        "row_count": row_count,
        "columns": columns,
        "required_n": required_n,
        "contract_ok": not errors,
        "contract_errors": errors,
    }


def _last_turn_usage(transcript: Path) -> dict[str, int] | None:
    """Read the final Codex turn usage without loading a large transcript."""

    try:
        size = transcript.stat().st_size
        with transcript.open("rb") as handle:
            handle.seek(max(0, size - 4 * 1024 * 1024))
            payload = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(payload.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage") if event.get("type") == "turn.completed" else None
        if not isinstance(usage, dict):
            continue
        normalized = {
            key: int(value)
            for key, value in usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        return normalized or None
    return None


def _transcript_metadata(trial: Path) -> tuple[dict[str, object], list[str]]:
    transcript = trial / "agent" / "codex.txt"
    warnings: list[str] = []
    if not transcript.is_file():
        warnings.append(f"canonical Harbor transcript is missing: {transcript}")
        return {
            "path": str(transcript.absolute()),
            "present": False,
            "sha256": None,
            "bytes": None,
            "final_usage": None,
        }, warnings

    usage = _last_turn_usage(transcript)
    if usage is None:
        # Harbor also writes aggregate agent usage into result.json. It is a
        # reconciliation fallback, not a replacement for the raw transcript.
        result_path = trial / "result.json"
        try:
            result = json.loads(result_path.read_text())
            agent_result = result.get("agent_result") or {}
            usage = {
                "input_tokens": int(agent_result["n_input_tokens"]),
                "cached_input_tokens": int(agent_result.get("n_cache_tokens", 0)),
                "output_tokens": int(agent_result["n_output_tokens"]),
            }
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            warnings.append(
                "final token usage is unavailable from transcript and result"
            )
            usage = None
    return {
        "path": str(transcript.absolute()),
        "present": True,
        "sha256": _sha256(transcript),
        "bytes": transcript.stat().st_size,
        "final_usage": usage,
    }, warnings


def _registry_paths(trial: Path) -> tuple[Path, ...]:
    # The first path is the current contract. The second accepts trials created
    # by the immediately preceding controller without trusting model telemetry.
    return (
        trial / ".runtime" / "slurm-broker" / "submissions.jsonl",
        trial / ".baseline-runtime" / "submissions.jsonl",
    )


def _load_registered_jobs(trial: Path) -> tuple[list[str], list[str], Path]:
    warnings: list[str] = []
    primary = _registry_paths(trial)[0]
    registry = next(
        (path for path in _registry_paths(trial) if path.is_file()), primary
    )
    if not registry.is_file():
        warnings.append(f"guarded Slurm submission registry is missing: {primary}")
        return [], warnings, primary

    jobs: list[str] = []
    seen: set[str] = set()
    try:
        for line_number, line in enumerate(registry.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                job_id = str(record["job_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                warnings.append(
                    f"invalid guarded Slurm registry record {registry}:{line_number}"
                )
                continue
            if not re.fullmatch(r"[0-9]+", job_id):
                warnings.append(f"invalid guarded Slurm job ID in registry: {job_id!r}")
                continue
            if job_id not in seen:
                jobs.append(job_id)
                seen.add(job_id)
    except OSError as error:
        warnings.append(f"guarded Slurm registry is unreadable: {error}")
    return jobs, warnings, registry


def _tres_count(raw: str, resource: str) -> float:
    values: dict[str, float] = {}
    for item in raw.split(","):
        key, separator, value = item.strip().partition("=")
        if not separator:
            continue
        try:
            values[key] = float(value)
        except ValueError:
            continue
    if resource == "gpu":
        if "gres/gpu" in values:
            return values["gres/gpu"]
        return sum(
            value for key, value in values.items() if key.startswith("gres/gpu:")
        )
    return values.get(resource, 0.0)


def parse_sacct_allocations(
    output: str, registered_jobs: list[str]
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Parse allocation rows, preferring array tasks over their parent row."""

    by_root: dict[str, dict[str, dict[str, object]]] = {
        job_id: {} for job_id in registered_jobs
    }
    warnings: list[str] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|")
        if len(fields) < 5:
            warnings.append(f"unparseable sacct allocation row {line_number}")
            continue
        job_id, state, elapsed_raw, alloc_tres, alloc_cpus = fields[:5]
        if "." in job_id:
            continue
        root_match = re.match(r"^([0-9]+)", job_id)
        if root_match is None or root_match.group(1) not in by_root:
            continue
        try:
            elapsed_seconds = max(0.0, float(elapsed_raw or 0))
            cpus = max(float(alloc_cpus or 0), _tres_count(alloc_tres, "cpu"))
        except ValueError:
            warnings.append(f"invalid sacct duration/CPU count for {job_id}")
            continue
        gpus = max(0.0, _tres_count(alloc_tres, "gpu"))
        by_root[root_match.group(1)][job_id] = {
            "job_id": job_id,
            "state": state.split()[0].rstrip("+").upper(),
            "elapsed_seconds": elapsed_seconds,
            "allocated_gpus": gpus,
            "allocated_cpus": cpus,
            "gpu_seconds": elapsed_seconds * gpus,
            "cpu_seconds": elapsed_seconds * cpus,
        }

    totals: dict[str, dict[str, object]] = {}
    for root, rows_by_id in by_root.items():
        rows = list(rows_by_id.values())
        array_rows = [row for row in rows if "_" in str(row["job_id"])]
        selected = array_rows or [row for row in rows if row["job_id"] == root]
        if not selected and rows:
            # Covers uncommon allocation identifiers such as heterogeneous
            # components without ever counting both a parent and its children.
            selected = rows
        if selected:
            totals[root] = {
                "source": "sacct",
                "rows": selected,
                "gpu_seconds": sum(float(row["gpu_seconds"]) for row in selected),
                "cpu_seconds": sum(float(row["cpu_seconds"]) for row in selected),
            }
    return totals, warnings


def _query_sacct(
    job_ids: list[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    if not job_ids:
        return {}, []
    command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--allocations",
        "--jobs",
        ",".join(job_ids),
        "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,AllocCPUS",
    ]
    try:
        completed = runner(command, text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {}, [f"sacct accounting query failed: {error}"]
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip() or "unknown error"
        return {}, [f"sacct accounting query failed: {detail}"]
    return parse_sacct_allocations(completed.stdout, job_ids)


def _execution_fallback(
    trial: Path, registered_jobs: list[str]
) -> tuple[dict[str, dict[str, object]], list[str]]:
    grouped: dict[str, list[dict[str, object]]] = {
        job_id: [] for job_id in registered_jobs
    }
    warnings: list[str] = []
    for path in trial.rglob("*.json"):
        if path.parent.name != ".execution" or "scoring" in path.parts:
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("backend") != "slurm"
            or record.get("status") not in _TERMINAL_EXECUTION_STATUSES
        ):
            continue
        job_id = str(record.get("job_id") or "")
        root_match = re.match(r"^([0-9]+)", job_id)
        if root_match is None or root_match.group(1) not in grouped:
            continue
        try:
            gpu_seconds = float(
                record.get("gpu_seconds")
                or float(record.get("wall_seconds", 0))
                * int(record.get("allocated_gpus", 0))
            )
        except (TypeError, ValueError):
            warnings.append(f"invalid Pipelines execution record: {path}")
            continue
        grouped[root_match.group(1)].append(
            {
                "path": str(path.absolute()),
                "job_id": job_id,
                "status": record["status"],
                "gpu_seconds": max(0.0, gpu_seconds),
            }
        )
    return {
        root: {
            "source": "pipelines_execution",
            "rows": rows,
            "gpu_seconds": sum(float(row["gpu_seconds"]) for row in rows),
            # Execution records intentionally do not claim whole-allocation CPU cost.
            "cpu_seconds": 0.0,
        }
        for root, rows in grouped.items()
        if rows
    }, warnings


def reconcile_compute(
    trial: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, object], list[str]]:
    job_ids, warnings, registry = _load_registered_jobs(trial)
    sacct, sacct_warnings = _query_sacct(job_ids, runner)
    fallback, fallback_warnings = _execution_fallback(trial, job_ids)
    warnings.extend(sacct_warnings)
    warnings.extend(fallback_warnings)

    jobs: dict[str, dict[str, object]] = {}
    uncovered: list[str] = []
    for job_id in job_ids:
        if job_id in sacct:
            jobs[job_id] = sacct[job_id]
        elif job_id in fallback:
            jobs[job_id] = fallback[job_id]
            warnings.append(
                f"Slurm job {job_id} used Pipelines .execution fallback; "
                "scheduler allocation accounting was unavailable"
            )
        else:
            uncovered.append(job_id)
    registry_present = registry.is_file()
    accounting_complete = registry_present and not uncovered
    if uncovered:
        warnings.append(
            "GPU accounting is incomplete; no sacct allocation or Pipelines "
            f"execution record covers jobs: {', '.join(uncovered)}"
        )
    gpu_seconds = sum(float(job["gpu_seconds"]) for job in jobs.values())
    cpu_seconds = sum(float(job["cpu_seconds"]) for job in jobs.values())
    return {
        "registry_path": str(registry.absolute()),
        "registry_present": registry_present,
        "registered_job_ids": job_ids,
        "jobs": jobs,
        "uncovered_job_ids": uncovered,
        "accounting_complete": accounting_complete,
        "gpu_hours": gpu_seconds / 3600.0,
        "cpu_hours": cpu_seconds / 3600.0,
    }, warnings


def _agent_wallclock_minutes(trial: Path) -> float | None:
    try:
        result = json.loads((trial / "result.json").read_text())
        execution = result["agent_execution"]
        started = datetime.fromisoformat(
            str(execution["started_at"]).replace("Z", "+00:00")
        )
        finished = datetime.fromisoformat(
            str(execution["finished_at"]).replace("Z", "+00:00")
        )
        return max(0.0, (finished - started).total_seconds() / 60.0)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_trial_audit(
    trial: Path,
    designs: Path,
    task_name: str,
    required_n: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[Path, Path, dict[str, object]]:
    """Write the immutable controller view used by held-out scoring."""

    controller = trial / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    submission = _submission_contract(designs, task_name, required_n)
    transcript, transcript_warnings = _transcript_metadata(trial)
    compute, compute_warnings = reconcile_compute(trial, runner=runner)
    warnings = [*transcript_warnings, *compute_warnings]
    audit_ok = (
        bool(transcript["present"])
        and transcript["final_usage"] is not None
        and bool(compute["accounting_complete"])
    )
    audit = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "task": task_name,
        "submission": submission,
        "transcript": transcript,
        "compute": compute,
        "contract_ok": submission["contract_ok"],
        "audit_ok": audit_ok,
        "warnings": warnings,
    }
    audit_path = controller / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    run_meta = {
        "schema_version": 1,
        "source": "harbor-controller",
        "gpu_hours": compute["gpu_hours"],
        "cpu_hours": compute["cpu_hours"],
        "wallclock_min": _agent_wallclock_minutes(trial),
        "gpu_accounting_complete": compute["accounting_complete"],
    }
    run_meta_path = controller / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")
    return audit_path, run_meta_path, audit
