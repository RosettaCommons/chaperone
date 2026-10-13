"""A provenance record written at launch, outside everything the agent can read.

The manifest records what was launched: the controller commit, the container
images, the sanitized bundle, the rendered prompt and instruction, the public
inputs, the resolved arguments and the scorer's hash. It lives in
``<jobs root>/manifests/`` and is never copied into ``task-definitions/`` or a
trial workspace, both of which the agent can read.

It deliberately holds no path into the private tasks tree and no path to the
scorer, only the scorer's hash.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task_paths import EVALS_ROOT, _digests, task_scorer

SCHEMA_VERSION = 1
_CHUNK = 8 * 1024 * 1024
_REDACTED = "<redacted>"
_CACHE_NAME = ".asset-cache.json"


def _identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str | None:
    """SHA-256 of a small file, or None if it does not exist."""
    return _sha256(path) if path.is_file() else None


def asset_record(
    path: Path, cache_path: Path, *, digest: bool = True
) -> dict[str, Any]:
    """Describe a large file, hashing it at most once while it is unchanged.

    The digest is cached under the file's identity (device, inode, size, mtime,
    ctime), so a multi-gigabyte image is read once, not on every launch. The file
    is re-checked after hashing, and a file that changed underneath is refused.
    """
    path = Path(os.path.abspath(path))
    if not path.is_file():
        return {"path": str(path), "present": False}
    before = _identity(path)
    record: dict[str, Any] = {"path": str(path), "present": True, **before}
    if not digest:
        record["sha256"] = None
        return record
    key = ":".join(str(before[name]) for name in sorted(before))
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    cached = cache.get(key)
    if cached and cached.get("path") == str(path):
        record["sha256"] = cached["sha256"]
        record["digest_cached"] = True
        return record
    print(f"Hashing {path} once; the digest is cached.", flush=True)
    value = _sha256(path)
    if _identity(path) != before:
        raise SystemExit(f"{path} changed while it was being hashed; retry.")
    cache[key] = {"path": str(path), "sha256": value}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, indent=1, sort_keys=True))
    temporary.replace(cache_path)
    record["sha256"] = value
    record["digest_cached"] = False
    return record


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(EVALS_ROOT), *arguments],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def controller_state() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--", ".")
    try:
        harbor = importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError:
        harbor = None
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "harbor_version": harbor,
        "python": sys.version.split()[0],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def resolved_arguments(args: Any) -> dict[str, Any]:
    """The parsed arguments, without the location of the private tree."""
    return {
        name: _REDACTED if "private" in name else _jsonable(value)
        for name, value in sorted(vars(args).items())
    }


def redacted_argv(argv: list[str]) -> list[str]:
    """The command line, with the value of any ``--private-*`` option removed."""
    cleaned: list[str] = []
    hide_next = False
    for item in argv:
        if hide_next:
            cleaned.append(_REDACTED)
            hide_next = False
        elif item.startswith("--private") and "=" in item:
            cleaned.append(item.split("=", 1)[0] + "=" + _REDACTED)
        elif item.startswith("--private"):
            cleaned.append(item)
            hide_next = True
        else:
            cleaned.append(item)
    return cleaned


def build_manifest(
    args: Any,
    task: Path,
    harbor_task: Path,
    run_name: str,
    jobs_root: Path,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Collect the launch record for ``run_name``."""
    cache = jobs_root / _CACHE_NAME
    digest = not getattr(args, "skip_image_digests", False)
    scorer = None if args.no_score else sha256_file(task_scorer(task.name))
    seed = getattr(args, "workspace_seed", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "run_name": run_name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": args.arm,
        "argv": redacted_argv(list(sys.argv[1:] if argv is None else argv)),
        "args": resolved_arguments(args),
        "controller": controller_state(),
        "task": {
            "name": task.name,
            "public_inputs": _digests(task / "inputs"),
            "rendered_prompt_sha256": sha256_file(harbor_task / "prompt.md"),
            "rendered_instruction_sha256": sha256_file(harbor_task / "instruction.md"),
        },
        "bundle_sha256": Path(seed).name if args.arm == "harness" and seed else None,
        "images": {
            "agent": asset_record(args.image, cache, digest=digest),
            "worker": asset_record(args.worker_image, cache, digest=digest),
        },
        "scorer_sha256": scorer,
    }


def _write(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_manifest(manifest: dict[str, Any], jobs_root: Path) -> Path:
    """Write the manifest under ``<jobs root>/manifests`` and return its path."""
    stamp = manifest["created"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = jobs_root / "manifests" / f"{manifest['run_name']}-{stamp}.json"
    _write(manifest, path)
    return path


def mark_submitted(path: Path, job_id: str, result_dir: Path) -> None:
    """Record the Slurm job id once the job has been accepted."""
    manifest = json.loads(path.read_text())
    manifest.update(
        {
            "status": "submitted",
            "slurm_job_id": str(job_id),
            "result_dir": str(result_dir),
        }
    )
    _write(manifest, path)
