"""Compact, read-only inspection of ordinary Pipelines campaign artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INSPECT_VIEWS = ("summary", "failures", "funnel", "candidates")
MAX_ITEMS = 20
_MAX_RELEVANT_FILES = 100_000
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_LOGS_SCANNED = 50
_LOG_EXCERPT_BYTES = 1_024
_MAX_RESPONSE_BYTES = 32 * 1024
_SKIP_DIRECTORIES = frozenset(
    {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
)


class InspectError(ValueError):
    """The campaign cannot be inspected safely or compactly."""


def _display_path(path: Path, campaign: Path) -> str:
    relative = path.relative_to(campaign)
    return "." if not relative.parts else relative.as_posix()


def _campaign_files(campaign: Path) -> list[Path]:
    """Return relevant artifacts without retaining raw design payload files."""
    files: list[Path] = []
    seen = 0
    for directory, names, filenames in os.walk(campaign, followlinks=False):
        names[:] = sorted(
            name
            for name in names
            if name not in _SKIP_DIRECTORIES
            and not (Path(directory) / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = Path(directory) / filename
            name = path.name.lower()
            relevant = (
                name
                in {
                    "pipeline.json",
                    "score.json",
                    "score_report.json",
                    "designs.csv",
                    "score_per_design.csv",
                }
                or name.endswith("_score.json")
                or (path.suffix == ".json" and path.parent.name == ".execution")
                or (path.suffix.lower() == ".csv" and "per_design" in path.stem)
                or _is_log(path)
            )
            if not relevant:
                continue
            if not path.is_file() or path.is_symlink():
                continue
            seen += 1
            if seen > _MAX_RELEVANT_FILES:
                raise InspectError(
                    "campaign contains more than "
                    f"{_MAX_RELEVANT_FILES:,} monitoring artifacts; inspect a "
                    "narrower campaign directory"
                )
            files.append(path)
    return files


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _trim_text(value: object, limit: int = 400) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact_scalar(value: object, limit: int = 160) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return _trim_text(value, limit)


def _count_csv_rows(path: Path) -> int | None:
    try:
        with path.open(newline="", errors="replace") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except (OSError, csv.Error):
        return None


def _execution_files(files: Iterable[Path]) -> list[Path]:
    return [
        path
        for path in files
        if path.suffix == ".json"
        and path.parent.name == ".execution"
        and not path.name.startswith(".tmp_")
    ]


def _execution_summary(files: Iterable[Path]) -> dict[str, object]:
    statuses: Counter[str] = Counter()
    wall_seconds = 0.0
    gpu_seconds = 0.0
    for path in _execution_files(files):
        record = _load_json(path)
        if record is None:
            continue
        statuses[_trim_text(record.get("status") or "unknown", 80)] += 1
        try:
            wall_seconds += float(record.get("wall_seconds") or 0.0)
            gpu_seconds += float(record.get("gpu_seconds") or 0.0)
        except (TypeError, ValueError):
            pass
    return {
        "attempts": sum(statuses.values()),
        "statuses": dict(sorted(statuses.items())),
        "wall_hours": round(wall_seconds / 3600.0, 3),
        "gpu_hours": round(gpu_seconds / 3600.0, 3),
    }


def _manifest_entries(
    campaign: Path, files: Iterable[Path]
) -> list[tuple[Path, dict[str, Any]]]:
    entries = []
    for path in files:
        if path.name != "pipeline.json":
            continue
        payload = _load_json(path)
        if payload is not None and isinstance(payload.get("steps", []), list):
            entries.append((path, payload))
    return sorted(entries, key=lambda item: _display_path(item[0], campaign))


def _score_summaries(
    campaign: Path, files: Iterable[Path], max_items: int
) -> tuple[int, list[dict[str, object]]]:
    candidates = []
    for path in files:
        name = path.name.lower()
        if name not in {"score.json", "score_report.json"} and not name.endswith(
            "_score.json"
        ):
            continue
        payload = _load_json(path)
        if payload is None or not {
            "n_reported",
            "required_n",
            "contract_ok",
        }.issubset(payload):
            continue
        criterion = payload.get("per_criterion")
        criterion_items = (
            [
                (_trim_text(name, 128), _compact_scalar(value))
                for name, value in sorted(criterion.items())[:max_items]
            ]
            if isinstance(criterion, dict)
            else []
        )
        errors = payload.get("contract_errors")
        summary: dict[str, object] = {
            "path": _display_path(path, campaign),
            "eval": _compact_scalar(payload.get("eval")),
            "contract_ok": _compact_scalar(payload.get("contract_ok")),
            "n_reported": _compact_scalar(payload.get("n_reported")),
            "required_n": _compact_scalar(payload.get("required_n")),
            "n_pass": _compact_scalar(payload.get("n_pass")),
            "pass_rate": _compact_scalar(payload.get("pass_rate")),
            "per_criterion": dict(criterion_items),
        }
        if isinstance(criterion, dict) and len(criterion) > len(criterion_items):
            summary["criteria_truncated"] = True
        if isinstance(errors, list) and errors:
            summary["contract_errors"] = [
                _trim_text(error) for error in errors[:max_items]
            ]
        candidates.append((path, summary))
    candidates.sort(key=lambda item: _mtime(item[0]), reverse=True)
    return len(candidates), [summary for _, summary in candidates[:max_items]]


def _submission_summaries(
    campaign: Path, files: Iterable[Path], max_items: int
) -> tuple[int, list[dict[str, object]]]:
    candidates = [path for path in files if path.name == "designs.csv"]
    candidates.sort(
        key=lambda path: ("submission" in path.parts, _mtime(path)), reverse=True
    )
    return len(candidates), [
        {"path": _display_path(path, campaign), "designs": _count_csv_rows(path)}
        for path in candidates[:max_items]
    ]


def _summary_view(
    campaign: Path, files: list[Path], max_items: int
) -> dict[str, object]:
    manifests = _manifest_entries(campaign, files)
    runs = []
    completed_steps = 0
    for path, payload in manifests[:max_items]:
        steps = [step for step in payload.get("steps", []) if isinstance(step, dict)]
        completed_steps += len(steps)
        runs.append(
            {
                "rundir": _display_path(path.parent, campaign),
                "completed_steps": len(steps),
                "last_step": _compact_scalar(steps[-1].get("name")) if steps else None,
            }
        )
    # Account for manifests omitted from the sample as well.
    for _, payload in manifests[max_items:]:
        completed_steps += sum(
            isinstance(step, dict) for step in payload.get("steps", [])
        )

    submission_count, submissions = _submission_summaries(campaign, files, max_items)
    score_count, scores = _score_summaries(campaign, files, max_items)
    return {
        "view": "summary",
        "campaign": str(campaign),
        "pipelines": {
            "runs": len(manifests),
            "completed_steps": completed_steps,
            "execution": _execution_summary(files),
            "sample": runs,
            "sample_truncated": len(manifests) > len(runs),
        },
        "submissions": {
            "artifacts": submission_count,
            "sample": submissions,
            "sample_truncated": submission_count > len(submissions),
        },
        "scores": {
            "artifacts": score_count,
            "sample": scores,
            "sample_truncated": score_count > len(scores),
        },
    }


def _failed_execution_records(
    campaign: Path, files: Iterable[Path]
) -> list[dict[str, object]]:
    failures = []
    for path in _execution_files(files):
        record = _load_json(path)
        if record is None or record.get("status") != "failed":
            continue
        failures.append(
            {
                "path": _display_path(path, campaign),
                "block": _compact_scalar(record.get("block_name")),
                "chunk": _compact_scalar(record.get("chunk_id")),
                "attempt": _compact_scalar(record.get("attempt")),
                "job_id": _compact_scalar(record.get("job_id")),
                "ended_at": _compact_scalar(record.get("ended_at")),
                "error": _trim_text(record.get("error") or "unknown failure"),
            }
        )
    failures.sort(key=lambda item: str(item.get("ended_at") or ""), reverse=True)
    return failures


def _is_log(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in {".log", ".err", ".out"} or name in {
        "stderr",
        "stdout",
        "log.txt",
    }


def _log_evidence(
    campaign: Path, files: Iterable[Path], max_items: int
) -> list[dict[str, object]]:
    # Import lazily so wait/launch remain lightweight and do not initialize the
    # main Pipelines package unless inspection actually needs a log excerpt.
    from chaperone.artifacts import read_excerpt

    logs = sorted((path for path in files if _is_log(path)), key=_mtime, reverse=True)[
        :_MAX_LOGS_SCANNED
    ]
    evidence = []
    for path in logs:
        try:
            excerpt = read_excerpt(
                path,
                tail=True,
                max_lines=4,
                max_bytes=_LOG_EXCERPT_BYTES,
                view="errors",
            )
        except (OSError, ValueError):
            continue
        if not excerpt.text.strip():
            continue
        evidence.append(
            {
                "path": _display_path(path, campaign),
                "excerpt": excerpt.text.strip(),
                "truncated": excerpt.truncated,
            }
        )
        if len(evidence) >= max_items:
            break
    return evidence


def _failures_view(
    campaign: Path, files: list[Path], max_items: int
) -> dict[str, object]:
    failures = _failed_execution_records(campaign, files)
    return {
        "view": "failures",
        "campaign": str(campaign),
        "failed_attempts": len(failures),
        "failures": failures[:max_items],
        "failures_truncated": len(failures) > max_items,
        "log_evidence": _log_evidence(campaign, files, max_items),
    }


def _parquet_rows(path: Path) -> int | None:
    try:
        import polars as pl

        return int(pl.scan_parquet(path).select(pl.len()).collect().item())
    except Exception:  # A broken/incomplete output is represented as unavailable.
        return None


def _funnel_view(
    campaign: Path, files: list[Path], max_items: int
) -> dict[str, object]:
    manifests = _manifest_entries(campaign, files)
    entries: list[dict[str, object]] = []
    total_steps = 0
    for manifest_path, payload in manifests:
        rundir = manifest_path.parent
        for step in payload.get("steps", []):
            if not isinstance(step, dict):
                continue
            total_steps += 1
            if len(entries) >= max_items:
                continue
            parquet_value = step.get("parquet")
            parquet = rundir / str(parquet_value) if parquet_value else None
            if parquet is not None:
                parquet = parquet.resolve(strict=False)
            rows = (
                _parquet_rows(parquet)
                if parquet is not None
                and parquet.is_relative_to(campaign)
                and parquet.is_file()
                else None
            )
            entries.append(
                {
                    "rundir": _display_path(rundir, campaign),
                    "step": _compact_scalar(step.get("name")),
                    "parents": [
                        _compact_scalar(parent)
                        for parent in (step.get("pipeline_input_names") or [])[:5]
                    ]
                    if isinstance(step.get("pipeline_input_names") or [], list)
                    else [],
                    "rows": rows,
                }
            )
    return {
        "view": "funnel",
        "campaign": str(campaign),
        "runs": len(manifests),
        "steps": entries,
        "total_steps": total_steps,
        "truncated": total_steps > len(entries),
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _candidate_file(files: Iterable[Path]) -> Path | None:
    candidates = [
        path
        for path in files
        if path.suffix.lower() == ".csv"
        and (path.name == "score_per_design.csv" or "per_design" in path.stem)
    ]
    return max(candidates, key=_mtime) if candidates else None


def _candidates_view(
    campaign: Path,
    files: list[Path],
    max_items: int,
    rank_by: str | None,
    direction: str,
) -> dict[str, object]:
    path = _candidate_file(files)
    if path is None:
        return {
            "view": "candidates",
            "campaign": str(campaign),
            "status": "unavailable",
            "reason": "no score_per_design.csv artifact was found",
        }

    criterion_counts: Counter[str] = Counter()
    selected: list[tuple[float, int, dict[str, object]]] = []
    eligible = 0
    total = 0
    try:
        with path.open(newline="", errors="replace") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if rank_by is not None and rank_by not in columns:
                raise InspectError(
                    f"rank_by column {rank_by!r} is absent from "
                    f"{_display_path(path, campaign)}"
                )
            verdict_columns = [
                column
                for column in columns
                if column == "pass"
                or column == "soft_pass"
                or column.startswith("pass_")
            ]
            verdict_columns.sort(
                key=lambda column: (
                    {"pass": 0, "soft_pass": 1}.get(column, 2),
                    column,
                )
            )
            reported_verdict_columns = verdict_columns[:MAX_ITEMS]
            identity_column = next(
                (column for column in ("design_id", "name", "id") if column in columns),
                None,
            )
            has_pass = "pass" in columns
            for index, row in enumerate(reader):
                total += 1
                for column in reported_verdict_columns:
                    if _truthy(row.get(column)):
                        criterion_counts[column] += 1
                passed = _truthy(row.get("pass")) if has_pass else True
                if not passed:
                    continue
                eligible += 1
                candidate: dict[str, object] = {
                    "design_id": _compact_scalar(
                        row.get(identity_column) if identity_column else index, 200
                    ),
                }
                for column in ("pass", "soft_pass"):
                    if column in columns:
                        candidate[column] = _truthy(row.get(column))
                if rank_by is None:
                    score = -float(index)
                else:
                    try:
                        score = float(row.get(rank_by, ""))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(score):
                        continue
                    candidate[rank_by] = score
                    if direction == "min":
                        score = -score
                selected.append((score, index, candidate))
                selected.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                if len(selected) > max_items:
                    selected.pop()
    except (OSError, csv.Error) as error:
        raise InspectError(
            f"cannot read {_display_path(path, campaign)}: {error}"
        ) from error

    return {
        "view": "candidates",
        "campaign": str(campaign),
        "artifact": _display_path(path, campaign),
        "total": total,
        "eligible": eligible,
        "criterion_pass_counts": dict(sorted(criterion_counts.items())),
        "criteria_truncated": len(verdict_columns) > len(reported_verdict_columns),
        "rank_by": rank_by,
        "direction": direction if rank_by else None,
        "candidates": [candidate for _, _, candidate in selected],
        "truncated": eligible > len(selected),
    }


def inspect_campaign(
    *,
    campaign: Path,
    view: str = "summary",
    max_items: int = 5,
    rank_by: str | None = None,
    direction: str = "max",
) -> dict[str, object]:
    """Inspect a campaign without creating state or returning unbounded artifacts."""
    campaign = Path(campaign).resolve(strict=False)
    if not campaign.is_dir():
        raise InspectError(f"campaign directory does not exist: {campaign}")
    if view not in INSPECT_VIEWS:
        raise InspectError(f"view must be one of: {', '.join(INSPECT_VIEWS)}")
    if isinstance(max_items, bool) or not isinstance(max_items, int):
        raise InspectError("max_items must be an integer")
    if not 1 <= max_items <= MAX_ITEMS:
        raise InspectError(f"max_items must be between 1 and {MAX_ITEMS}")
    if direction not in {"min", "max"}:
        raise InspectError("direction must be 'min' or 'max'")
    if rank_by is not None and (not isinstance(rank_by, str) or len(rank_by) > 128):
        raise InspectError("rank_by must be a column name of at most 128 characters")

    files = _campaign_files(campaign)
    if view == "summary":
        result = _summary_view(campaign, files, max_items)
    elif view == "failures":
        result = _failures_view(campaign, files, max_items)
    elif view == "funnel":
        result = _funnel_view(campaign, files, max_items)
    else:
        result = _candidates_view(campaign, files, max_items, rank_by, direction)
    if len(json.dumps(result, separators=(",", ":")).encode()) > _MAX_RESPONSE_BYTES:
        raise InspectError(
            "inspection exceeded the 32 KiB response limit; reduce max_items or "
            "inspect a narrower campaign directory"
        )
    return result
