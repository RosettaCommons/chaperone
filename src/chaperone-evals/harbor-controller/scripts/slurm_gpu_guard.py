#!/usr/bin/env python3
"""Guarded ``sbatch`` client and host-side broker.

The agent container receives only the client entry point.  The real host
``sbatch`` binary remains outside the container and is invoked by the broker
after every GPU request has been checked.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOCKET = "/run/harbor-slurm/sbatch.sock"
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
# These are task-provided scientific/runtime settings, not arbitrary shell
# state.  HOME, PATH, TMPDIR, and XDG paths are deliberately absent: the broker
# supplies fixed workspace-local values for those security-sensitive names.
FORWARDED_ENV_NAMES = {
    "AF3_JAX_CACHE_DIR",
    "ALPHAFOLD3_WEIGHTS_DIR",
    "BOLTZ_CACHE_DIR",
    "CCD_MIRROR_PATH",
    "FOLDSEEK_EXEC",
    "FOUNDRY_CHECKPOINT_DIRS",
    "HBPLUS_COPY",
    "HF_HUB_CACHE",
    "OPENFOLD3_CACHE_DIR",
    "PDB_MIRROR_PATH",
    "PYROSETTA_PATH",
    "X3DNA_BIN_DIR",
    "X3DNA_DIR",
}
FORWARDED_ENV_PREFIXES = ("CHAPERONE_",)
# Controller-side settings that share the CHAPERONE_ prefix but are never
# worker configuration (the held-out task tree location, bootstrap options).
NEVER_FORWARDED_ENV_NAMES = {"CHAPERONE_EVALS_PRIVATE", "CHAPERONE_PYTHON_VERSION"}


def _is_forwarded_name(name: str) -> bool:
    if name in NEVER_FORWARDED_ENV_NAMES:
        return False
    return name in FORWARDED_ENV_NAMES or name.startswith(FORWARDED_ENV_PREFIXES)


def _forwarded_environment(request_env: dict[Any, Any]) -> dict[str, str]:
    """Return only safe, string-valued worker environment entries."""
    forwarded: dict[str, str] = {}
    for name, value in request_env.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise GuardError("sbatch environment names and values must be strings")
        if "\x00" in name or "\x00" in value or "\n" in name or "\r" in name:
            raise GuardError("sbatch environment contains an invalid name or value")
        if _is_forwarded_name(name):
            forwarded[name] = value
    return forwarded


def _explicit_export(environment: dict[str, str]) -> str:
    """Build an sbatch export option without embedding attacker-controlled values."""
    names = sorted(environment)
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names):
        raise GuardError("sbatch environment contains an invalid variable name")
    return "--export=" + (",".join(names) if names else "NONE")


def _merged_excluded_nodes(existing: str | None, required: str) -> str:
    """Merge a campaign-required node exclusion with a validated job request."""
    merged = ",".join(value for value in (existing, required) if value)
    if merged and not re.fullmatch(r"[A-Za-z0-9_,.\[\]-]+", merged):
        raise GuardError("unsupported --exclude value")
    return merged


_SAFE_SUBMITIT_DIRECTIVES = {
    "--array",
    "--cpus-per-task",
    "--error",
    "--exclude",
    "--gres",
    "--gpus-per-node",
    "--job-name",
    "--mem",
    "--nice",
    "--nodes",
    "--open-mode",
    "--output",
    "--partition",
    "--signal",
    "--time",
    "--wckey",
}
_REQUIRED_SUBMITIT_DIRECTIVES = {
    "--error",
    "--job-name",
    "--nodes",
    "--open-mode",
    "--output",
    "--partition",
    "--signal",
    "--time",
    "--wckey",
}
_SAFE_GENERIC_DIRECTIVES = {
    "--array",
    "--cpus-per-task",
    "--error",
    "--gres",
    "--job-name",
    "--mem",
    "--nodes",
    "--output",
    "--partition",
    "--time",
}
_REQUIRED_GENERIC_DIRECTIVES = {
    "--cpus-per-task",
    "--error",
    "--job-name",
    "--mem",
    "--output",
    "--partition",
    "--time",
}
_SLURM_PATH_PLACEHOLDER = "harbor-slurm-placeholder"


class GuardError(ValueError):
    """An sbatch request cannot be proven to satisfy the GPU limit."""


def _positive_max(value: str, option: str) -> int:
    """Return the largest value from a Slurm integer or integer range."""
    text = value.strip()
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", text):
        raise GuardError(f"unsupported {option} value {value!r}")
    values = [int(part) for part in text.split("-")]
    return max(values)


def _gpu_spec(
    value: str, option: str, *, gres: bool = False
) -> tuple[int, set[str | None]]:
    """Parse Slurm GPU/GRES syntax into count and requested GPU types."""
    total = 0
    matched = False
    types: set[str | None] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item or item.lower() == "none":
            continue
        parts = item.split(":")
        resource = parts[0].lower()
        if gres and resource not in {"gpu", "gres/gpu"}:
            continue
        matched = True
        count_text = parts[-1]
        if count_text.isdigit():
            count = int(count_text)
            if gres:
                gpu_type = parts[-2] if len(parts) >= 3 else None
            else:
                if len(parts) > 2:
                    raise GuardError(f"unsupported {option} GPU value {value!r}")
                gpu_type = parts[0] if len(parts) == 2 else None
        elif gres and len(parts) == 2:
            count = 1
            gpu_type = parts[1]
        elif len(parts) == 1 and not gres:
            raise GuardError(f"unsupported {option} GPU value {value!r}")
        else:
            raise GuardError(f"unsupported {option} GPU value {value!r}")
        total += count
        if count > 0:
            types.add(gpu_type.lower() if gpu_type else None)
    if not matched and not gres:
        raise GuardError(f"unsupported {option} GPU value {value!r}")
    return total, types


def _gpu_count(value: str, option: str, *, gres: bool = False) -> int:
    return _gpu_spec(value, option, gres=gres)[0]


def _option_values(tokens: list[str]) -> dict[str, list[str]]:
    """Extract only resource options needed to bound a request."""
    long_with_values = {
        "--gres",
        "--gpus",
        "--gpus-per-node",
        "--gpus-per-socket",
        "--gpus-per-task",
        "--nodes",
        "--ntasks",
        "--ntasks-per-node",
        "--sockets-per-node",
        "--tres-per-task",
    }
    gpu_resource_options = {
        "--gres",
        "--gpus",
        "--gpus-per-node",
        "--gpus-per-socket",
        "--gpus-per-task",
        "--tres-per-task",
    }
    short_names = {"-G": "--gpus", "-N": "--nodes", "-n": "--ntasks"}
    values: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name: str | None = None
        value: str | None = None
        if token.startswith("--"):
            name, separator, attached = token.partition("=")
            if name not in long_with_values and any(
                option.startswith(name) for option in gpu_resource_options
            ):
                raise GuardError(
                    f"unsupported abbreviated or unknown GPU option {name!r}"
                )
            if name in long_with_values:
                if separator:
                    value = attached
                else:
                    index += 1
                    if index >= len(tokens):
                        raise GuardError(f"{name} requires a value")
                    value = tokens[index]
        else:
            for short, long_name in short_names.items():
                if token == short:
                    index += 1
                    if index >= len(tokens):
                        raise GuardError(f"{short} requires a value")
                    name, value = long_name, tokens[index]
                    break
                if token.startswith(short) and len(token) > len(short):
                    name, value = long_name, token[len(short) :]
                    break
        if name is not None and value is not None:
            values.setdefault(name, []).append(value)
        index += 1
    return values


def _directive_tokens(script: str) -> list[str]:
    tokens: list[str] = []
    for line in script.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#SBATCH"):
            try:
                tokens.extend(shlex.split(stripped[len("#SBATCH") :].strip()))
            except ValueError as exc:
                raise GuardError(f"invalid #SBATCH directive: {line!r}: {exc}") from exc
            continue
        if not stripped or stripped.startswith("#"):
            continue
        # Slurm ignores directives after the first executable line.
        break
    return tokens


def _requested_gpu_details(
    argv: list[str], scripts: list[str] | None = None
) -> tuple[int, set[str | None]]:
    """Return a conservative GPU count and every requested GPU type."""
    tokens = list(argv)
    for script in scripts or []:
        tokens.extend(_directive_tokens(script))
    options = _option_values(tokens)

    nodes = max(
        (_positive_max(value, "--nodes") for value in options.get("--nodes", [])),
        default=1,
    )
    ntasks_values = options.get("--ntasks", [])
    ntasks_per_node_values = options.get("--ntasks-per-node", [])
    if ntasks_values:
        ntasks = max(_positive_max(value, "--ntasks") for value in ntasks_values)
    elif ntasks_per_node_values:
        ntasks = nodes * max(
            _positive_max(value, "--ntasks-per-node")
            for value in ntasks_per_node_values
        )
    else:
        ntasks = 1

    candidates = [0]
    gpu_types: set[str | None] = set()

    def add(value: str, option: str, multiplier: int = 1, *, gres: bool = False):
        count, types = _gpu_spec(value, option, gres=gres)
        candidates.append(multiplier * count)
        gpu_types.update(types)

    for value in options.get("--gpus", []):
        add(value, "--gpus")
    for value in options.get("--gres", []):
        add(value, "--gres", nodes, gres=True)
    for value in options.get("--gpus-per-node", []):
        add(value, "--gpus-per-node", nodes)
    for value in options.get("--gpus-per-task", []):
        add(value, "--gpus-per-task", ntasks)

    if options.get("--gpus-per-socket"):
        socket_values = options.get("--sockets-per-node", [])
        if not socket_values:
            raise GuardError(
                "--gpus-per-socket requires --sockets-per-node so the guard can "
                "bound the total; use --gres or --gpus-per-node instead"
            )
        sockets = max(
            _positive_max(value, "--sockets-per-node") for value in socket_values
        )
        for value in options["--gpus-per-socket"]:
            add(value, "--gpus-per-socket", nodes * sockets)

    for value in options.get("--tres-per-task", []):
        add(value, "--tres-per-task", ntasks, gres=True)
    return max(candidates), gpu_types


def requested_gpus(argv: list[str], scripts: list[str] | None = None) -> int:
    """Return a conservative upper bound on GPUs in one sbatch allocation."""
    return _requested_gpu_details(argv, scripts)[0]


DEFAULT_MAX_REQUESTED_GPUS_PER_SUBMISSION = 30


def _array_task_counts(value: str | None) -> tuple[int, int]:
    """Return total and simultaneously runnable task counts for an array."""
    if value is None:
        return 1, 1
    match = re.fullmatch(
        r"(?P<first>[0-9]+)(?:-(?P<last>[0-9]+))?(?:%(?P<throttle>[1-9][0-9]*))?",
        value,
    )
    if match is None:
        raise GuardError(f"unsupported --array value {value!r}")
    first = int(match.group("first"))
    last = int(match.group("last") or first)
    if last < first:
        raise GuardError(f"unsupported descending --array value {value!r}")
    task_count = last - first + 1
    throttle = int(match.group("throttle") or task_count)
    return task_count, min(task_count, throttle)


def _submission_gpu_reservation(
    directives: dict[str, str],
    gpu_count: int,
    max_total_gpus: int,
    max_requested_gpus_per_submission: int = (
        DEFAULT_MAX_REQUESTED_GPUS_PER_SUBMISSION
    ),
) -> int:
    """Return and enforce this submission's maximum concurrent GPU use."""
    task_count, array_concurrency = _array_task_counts(directives.get("--array"))
    requested_total = gpu_count * task_count
    if requested_total > max_requested_gpus_per_submission:
        raise GuardError(
            "sbatch request rejected: this submission requests "
            f"{requested_total} GPUs across all array tasks ({task_count} task(s) x "
            f"{gpu_count} GPU(s)); each submission may request at most "
            f"{max_requested_gpus_per_submission} GPUs regardless of its array "
            "concurrency throttle"
        )
    total = gpu_count * array_concurrency
    if total > max_total_gpus:
        raise GuardError(
            "sbatch request rejected: this submission could reserve "
            f"{total} GPUs ({array_concurrency} concurrent task(s) x "
            f"{gpu_count} GPU(s)); the entire trial may reserve at most "
            f"{max_total_gpus} GPUs total across all Slurm jobs and array tasks"
        )
    return total


def validate_request(
    argv: list[str],
    scripts: list[str],
    max_total_gpus: int,
    allowed_gpu_type: str = "small",
) -> int:
    count, gpu_types = _requested_gpu_details(argv, scripts)
    if count > max_total_gpus:
        raise GuardError(
            f"sbatch request rejected: requested {count} GPUs in one allocation; "
            f"the entire trial may reserve at most {max_total_gpus} GPUs total"
        )
    normalized_allowed = allowed_gpu_type.lower()
    if count > 0 and gpu_types != {normalized_allowed}:
        rendered = ", ".join(
            "untyped" if value is None else value
            for value in sorted(gpu_types, key=lambda item: item or "")
        )
        raise GuardError(
            "sbatch request rejected: GPU allocations must explicitly request "
            f"only gpu:{normalized_allowed}:N; found {rendered or 'untyped'}"
        )
    return count


def _send_message(connection: socket.socket, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload).encode()
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise GuardError("sbatch broker message is too large")
    connection.sendall(struct.pack("!I", len(encoded)) + encoded)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise GuardError("sbatch broker connection closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_message(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise GuardError("sbatch broker message is too large")
    value = json.loads(_receive_exact(connection, size))
    if not isinstance(value, dict):
        raise GuardError("invalid sbatch broker request")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _workspace_path(
    value: str,
    workspace: Path,
    option: str,
    workspace_alias: Path | None = None,
) -> Path:
    """Resolve a Slurm path template without letting substitutions escape."""
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise GuardError(f"invalid {option} path {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_./%+-]+", value):
        raise GuardError(f"unsafe characters in {option} path {value!r}")
    normalized = re.sub(r"%[A-Za-z%]", _SLURM_PATH_PLACEHOLDER, value)
    candidate = Path(normalized)
    if candidate.is_absolute() and workspace_alias is not None:
        try:
            relative = candidate.relative_to(workspace_alias)
        except ValueError:
            pass
        else:
            candidate = workspace / relative
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve(strict=False)
    if not _inside(candidate, workspace):
        raise GuardError(f"{option} path must stay inside {workspace}: {value}")
    return candidate


def _generic_directives(
    script: str, workspace: Path, workspace_alias: Path
) -> dict[str, str]:
    """Parse the deliberately small baseline sbatch directive language."""
    lines = script.splitlines()
    if not lines or lines[0] not in {"#!/bin/bash", "#!/usr/bin/env bash"}:
        raise GuardError("baseline script must use a supported bash shebang")
    directives: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or (
            stripped.startswith("#") and not stripped.startswith("#SBATCH")
        ):
            continue
        if not stripped.startswith("#SBATCH"):
            break
        try:
            tokens = shlex.split(stripped[len("#SBATCH") :].strip())
        except ValueError as exc:
            raise GuardError(f"invalid #SBATCH directive {line!r}: {exc}") from exc
        if len(tokens) != 1 or not tokens[0].startswith("--"):
            raise GuardError(f"unsupported #SBATCH directive {line!r}")
        name, separator, value = tokens[0].partition("=")
        if name not in _SAFE_GENERIC_DIRECTIVES:
            raise GuardError(f"unsafe or unsupported #SBATCH directive {name}")
        if not separator or not value:
            raise GuardError(f"{name} must use a non-empty --option=value form")
        if name in directives:
            raise GuardError(f"duplicate #SBATCH directive {name}")
        directives[name] = value

    missing = sorted(_REQUIRED_GENERIC_DIRECTIVES - directives.keys())
    if missing:
        raise GuardError("baseline script is missing directives: " + ", ".join(missing))
    for name in ("--output", "--error"):
        resolved_log = _workspace_path(
            directives[name], workspace, name, workspace_alias
        )
        if not _inside(resolved_log, workspace / "logs"):
            raise GuardError(f"{name} must stay inside /workspace/logs")
        unsupported_templates = re.findall(r"%[A-Za-z%]", directives[name])
        if any(
            value not in {"%A", "%a", "%j", "%x", "%%"}
            for value in unsupported_templates
        ):
            raise GuardError(f"unsupported Slurm path template in {name}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", directives["--job-name"]):
        raise GuardError("unsafe --job-name in baseline script")
    if directives.get("--nodes", "1") != "1":
        raise GuardError("baseline jobs must request exactly one node")
    if directives["--partition"] not in {"cpu", "gpu"}:
        raise GuardError("baseline partition must be cpu or gpu")
    if not re.fullmatch(
        r"[0-9]+(?:-[0-9]+)?(?:%[1-9][0-9]*)?", directives.get("--array", "0")
    ):
        raise GuardError("unsupported --array value")
    if not re.fullmatch(
        r"[1-9][0-9]*(?:[KMGT]B?)?", directives["--mem"], re.IGNORECASE
    ):
        raise GuardError("unsupported --mem value")
    if not re.fullmatch(r"[1-9][0-9]*", directives["--cpus-per-task"]):
        raise GuardError("unsupported --cpus-per-task value")
    if not re.fullmatch(
        r"[0-9]+(?:-[0-9]+){0,2}(?::[0-9]+){0,2}", directives["--time"]
    ):
        raise GuardError("unsupported --time value")
    return directives


def prepare_generic_submission(
    argv: list[str],
    cwd: Path,
    workspace: Path,
    workspace_alias: Path,
    scheduler_log_dir: Path,
    max_total_gpus: int = 5,
    allowed_gpu_type: str = "small",
    max_requested_gpus_per_submission: int = (
        DEFAULT_MAX_REQUESTED_GPUS_PER_SUBMISSION
    ),
) -> tuple[list[str], int, str, dict[str, str]]:
    """Validate a baseline payload and build canonical sbatch arguments."""
    if len(argv) != 1 or argv[0].startswith("-"):
        raise GuardError("sbatch requires exactly one baseline batch script")
    candidate = Path(argv[0])
    if candidate.is_absolute():
        try:
            candidate = workspace / candidate.relative_to(workspace_alias)
        except ValueError:
            pass
    else:
        candidate = cwd / candidate
    try:
        script_path = candidate.resolve(strict=True)
    except OSError as exc:
        raise GuardError(
            f"cannot resolve baseline batch script {candidate}: {exc}"
        ) from exc
    if not script_path.is_file() or not _inside(script_path, workspace):
        raise GuardError(
            "baseline batch script must stay inside the assigned workspace: "
            f"{script_path}"
        )
    try:
        script = script_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise GuardError(
            f"cannot inspect baseline batch script {script_path}: {exc}"
        ) from exc
    directives = _generic_directives(script, workspace, workspace_alias)
    gpu_count = validate_request([], [script], max_total_gpus, allowed_gpu_type)
    _submission_gpu_reservation(
        directives,
        gpu_count,
        max_total_gpus,
        max_requested_gpus_per_submission,
    )
    partition = directives["--partition"]
    gres = directives.get("--gres")
    scheduler_log_dir.mkdir(parents=True, exist_ok=True)
    scheduler_log_dir = scheduler_log_dir.resolve(strict=True)
    log_stem = directives["--job-name"] + "-%A_%a"
    canonical = [
        f"--chdir={workspace}",
        "--export=NONE",
        "--nodes=1",
        f"--job-name={directives['--job-name']}",
        f"--output={scheduler_log_dir / (log_stem + '.out')}",
        f"--error={scheduler_log_dir / (log_stem + '.err')}",
        f"--partition={partition}",
        f"--time={directives['--time']}",
        f"--mem={directives['--mem']}",
        f"--cpus-per-task={directives['--cpus-per-task']}",
    ]
    if "--array" in directives:
        canonical.append(f"--array={directives['--array']}")
    if gpu_count:
        expected_gres = f"gpu:{allowed_gpu_type.lower()}:{gpu_count}"
        if partition != "gpu" or gres is None or gres.lower() != expected_gres:
            raise GuardError(
                f"GPU baseline jobs must use --partition=gpu and --gres={expected_gres}"
            )
        canonical.extend(
            [
                f"--gres={expected_gres}",
                "--constraint=A4000",
            ]
        )
    elif partition != "cpu" or gres is not None:
        raise GuardError("CPU baseline jobs must use --partition=cpu without --gres")
    return canonical, gpu_count, script, directives


def _submitit_directives(script: str, workspace: Path) -> dict[str, str]:
    if not script.startswith("#!/bin/bash\n"):
        raise GuardError("Submitit script must use the exact #!/bin/bash shebang")
    directives: dict[str, str] = {}
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        try:
            tokens = shlex.split(stripped[len("#SBATCH") :].strip())
        except ValueError as exc:
            raise GuardError(f"invalid #SBATCH directive {line!r}: {exc}") from exc
        if len(tokens) != 1 or not tokens[0].startswith("--"):
            raise GuardError(f"unsupported #SBATCH directive {line!r}")
        name, separator, value = tokens[0].partition("=")
        if name not in _SAFE_SUBMITIT_DIRECTIVES:
            raise GuardError(f"unsafe or unsupported #SBATCH directive {name}")
        if not separator or not value:
            raise GuardError(f"{name} must use a non-empty --option=value form")
        if name in directives:
            raise GuardError(f"duplicate #SBATCH directive {name}")
        directives[name] = value

    missing = sorted(_REQUIRED_SUBMITIT_DIRECTIVES - directives.keys())
    if missing:
        raise GuardError("Submitit script is missing directives: " + ", ".join(missing))

    for name in ("--output", "--error"):
        _workspace_path(directives[name], workspace, name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", directives["--job-name"]):
        raise GuardError("unsafe --job-name in Submitit script")
    if directives["--nodes"] != "1":
        raise GuardError("Submitit jobs must request exactly one node")
    if directives["--open-mode"] != "append":
        raise GuardError("Submitit jobs must use --open-mode=append")
    if directives["--partition"] not in {"cpu", "gpu"}:
        raise GuardError("Submitit partition must be cpu or gpu")
    if not re.fullmatch(r"USR2@[1-9][0-9]*", directives["--signal"]):
        raise GuardError("Submitit jobs must use a bounded USR2 signal delay")
    if directives["--wckey"] != "submitit":
        raise GuardError("Submitit jobs must use --wckey=submitit")
    if not re.fullmatch(
        r"[0-9]+(?:-[0-9]+)?(?:%[1-9][0-9]*)?", directives.get("--array", "0")
    ):
        raise GuardError("unsupported --array value")
    if not re.fullmatch(
        r"[1-9][0-9]*(?:[KMGT]B?)?", directives.get("--mem", "1"), re.IGNORECASE
    ):
        raise GuardError("unsupported --mem value")
    if not re.fullmatch(r"[1-9][0-9]*", directives.get("--cpus-per-task", "1")):
        raise GuardError("unsupported --cpus-per-task value")
    if not re.fullmatch(
        r"[0-9]+(?:-[0-9]+){0,2}(?::[0-9]+){0,2}", directives["--time"]
    ):
        raise GuardError("unsupported --time value")
    if "--nice" in directives and not re.fullmatch(r"[0-9]+", directives["--nice"]):
        raise GuardError("unsupported --nice value")
    if "--exclude" in directives and not re.fullmatch(
        r"[A-Za-z0-9_,.\[\]-]+", directives["--exclude"]
    ):
        raise GuardError("unsupported --exclude value")
    if "--gpus-per-node" in directives and directives["--gpus-per-node"] != "0":
        raise GuardError(
            "Submitit GPU jobs must use typed --gres; only --gpus-per-node=0 "
            "is accepted for CPU compatibility"
        )
    return directives


def _submitit_body(
    script: str, workspace: Path, slurm_python: Path
) -> tuple[Path, Path]:
    body = [
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(body) != 2 or body[0] != "export SUBMITIT_EXECUTOR=slurm":
        raise GuardError(
            "Submitit script body must contain only its executor export and srun command"
        )
    try:
        tokens = shlex.split(body[1])
    except ValueError as exc:
        raise GuardError(f"invalid Submitit srun command: {exc}") from exc
    if not tokens or tokens.pop(0) != "srun":
        raise GuardError("Submitit script must execute through srun")
    if not tokens or tokens.pop(0) != "--unbuffered":
        raise GuardError("Submitit srun must begin with --unbuffered")

    output: Path | None = None
    error: Path | None = None
    saw_export_all = False
    while tokens and tokens[0].startswith("--"):
        option = tokens.pop(0)
        if option in {"--output", "--error"}:
            if not tokens:
                raise GuardError(f"{option} requires a path")
            resolved = _workspace_path(tokens.pop(0), workspace, f"srun {option}")
            if option == "--output" and output is None:
                output = resolved
            elif option == "--error" and error is None:
                error = resolved
            else:
                raise GuardError(f"duplicate srun {option}")
        elif option == "--export=ALL" and not saw_export_all:
            # The batch allocation is forced to --export=NONE. Submitit uses this
            # fixed flag only to pass that already-minimal environment to its step.
            saw_export_all = True
        else:
            raise GuardError(f"unsafe or unsupported srun option {option}")
    if output is None or error is None:
        raise GuardError("Submitit srun requires workspace-contained output and error")

    try:
        invoked_python = Path(tokens[0]).resolve(strict=True)
        expected_python = slurm_python.resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise GuardError(
            "Submitit srun must invoke the controller-provided slurm_python "
            "and submitit.core._submit exactly"
        ) from exc
    expected_suffix = ["-u", "-m", "submitit.core._submit"]
    if (
        invoked_python != expected_python
        or tokens[1:4] != expected_suffix
        or len(tokens) != 5
    ):
        raise GuardError(
            "Submitit srun must invoke the controller-provided slurm_python "
            "and submitit.core._submit exactly"
        )
    pickle_dir = _workspace_path(tokens[4], workspace, "Submitit pickle")
    return output, pickle_dir


def prepare_submission(
    argv: list[str],
    cwd: Path,
    workspace: Path,
    slurm_python: Path,
    stdin: str = "",
    max_total_gpus: int = 5,
    allowed_gpu_type: str = "small",
    max_requested_gpus_per_submission: int = (
        DEFAULT_MAX_REQUESTED_GPUS_PER_SUBMISSION
    ),
) -> tuple[list[str], int, str | None]:
    """Validate one broker request and return canonical real-sbatch arguments."""
    if stdin:
        raise GuardError("sbatch scripts must be passed as one workspace file")
    if argv == ["--version"]:
        return ["--version"], 0, None
    if "--wrap" in argv or any(token.startswith("--wrap=") for token in argv):
        raise GuardError("--wrap is not accepted by the guarded sbatch client")
    if ":" in argv:
        raise GuardError("heterogeneous sbatch requests are not accepted")
    if len(argv) != 1 or argv[0].startswith("-"):
        raise GuardError("sbatch requires exactly one Submitit batch script")

    candidate = Path(argv[0])
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        script_path = candidate.resolve(strict=True)
    except OSError as exc:
        raise GuardError(
            f"cannot resolve Submitit batch script {candidate}: {exc}"
        ) from exc
    if not script_path.is_file() or not _inside(script_path, workspace):
        raise GuardError(
            f"Submitit batch script must stay inside the assigned workspace: {script_path}"
        )
    try:
        script = script_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise GuardError(
            f"cannot inspect Submitit batch script {script_path}: {exc}"
        ) from exc

    directives = _submitit_directives(script, workspace)
    _submitit_body(script, workspace, slurm_python)
    gpu_count = validate_request([], [script], max_total_gpus, allowed_gpu_type)
    _submission_gpu_reservation(
        directives,
        gpu_count,
        max_total_gpus,
        max_requested_gpus_per_submission,
    )
    partition = directives["--partition"]
    gres = directives.get("--gres")
    gpus_per_node = directives.get("--gpus-per-node")
    canonical = [
        f"--chdir={cwd}",
        "--export=NONE",
    ]
    if gpu_count:
        expected_gres = f"gpu:{allowed_gpu_type.lower()}:{gpu_count}"
        if partition != "gpu" or gres is None or gres.lower() != expected_gres:
            raise GuardError(
                f"GPU Submitit jobs must use --partition=gpu and --gres={expected_gres}"
            )
        canonical.extend(
            [
                f"--gres={expected_gres}",
                "--constraint=A4000",
            ]
        )
    elif partition != "cpu" or gres is not None:
        raise GuardError("CPU Submitit jobs must use --partition=cpu without --gres")
    if gpus_per_node is not None and partition != "cpu":
        raise GuardError("--gpus-per-node=0 is accepted only for CPU Submitit jobs")
    # Submit the exact bytes that were validated instead of asking real sbatch
    # to reopen an agent-writable file after validation (a TOCTOU escape).
    return canonical, gpu_count, script


class _BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, _BrokerServer)
        try:
            request = _receive_message(self.connection)
            command = request.get("command", "sbatch")
            argv = request.get("argv")
            cwd_text = request.get("cwd")
            stdin = request.get("stdin", "")
            request_env = request.get("env", {})
            sbatch_env = request.get("sbatch_env", {})
            if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
                raise GuardError("invalid sbatch argument list")
            if command not in {"sbatch", "scancel", "scontrol", "squeue", "srun"}:
                raise GuardError(f"unsupported guarded Slurm command {command!r}")
            if server.policy == "generic" and command not in {
                "sbatch",
                "scancel",
                "squeue",
            }:
                raise GuardError(f"generic baseline broker does not permit {command!r}")
            if not isinstance(cwd_text, str) or not isinstance(stdin, str):
                raise GuardError("invalid sbatch cwd or stdin")
            if not isinstance(request_env, dict):
                raise GuardError("invalid sbatch environment")
            if not isinstance(sbatch_env, dict):
                raise GuardError("invalid SBATCH_* environment")
            if sbatch_env:
                names = ", ".join(sorted(str(name) for name in sbatch_env))
                raise GuardError(
                    "SBATCH_* environment options are not accepted by the guard; "
                    f"pass explicit sbatch flags instead (found: {names})"
                )

            cwd = server.resolve_client_path(cwd_text)
            if not _inside(cwd, server.workspace):
                raise GuardError(
                    f"sbatch working directory must stay inside {server.workspace}"
                )
            environment = {
                "HOME": str(server.workspace / ".home"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "TMPDIR": str(server.workspace / ".tmp"),
            }
            environment.update(_forwarded_environment(request_env))
            executable = server.real_sbatch
            canonical_argv: list[str]
            gpu_count = 0
            script: str | None = None
            directives: dict[str, str] | None = None
            if command == "sbatch":
                if server.policy == "generic" and argv != ["--version"]:
                    if stdin:
                        raise GuardError(
                            "sbatch scripts must be passed as one workspace file"
                        )
                    canonical_argv, gpu_count, payload, directives = (
                        prepare_generic_submission(
                            argv,
                            cwd,
                            server.workspace,
                            server.workspace_alias,
                            server.scheduler_log_dir,
                            server.max_total_gpus,
                            server.allowed_gpu_type,
                            server.max_requested_gpus_per_submission,
                        )
                    )
                    script = server.snapshot_generic_payload(payload, directives)
                elif server.policy == "generic":
                    canonical_argv = ["--version"]
                else:
                    canonical_argv, gpu_count, script = prepare_submission(
                        argv,
                        cwd,
                        server.workspace,
                        server.slurm_python,
                        stdin,
                        server.max_total_gpus,
                        server.allowed_gpu_type,
                        server.max_requested_gpus_per_submission,
                    )
            elif command == "scancel":
                if stdin:
                    raise GuardError("scancel does not accept standard input")
                canonical_argv = server.validate_scancel(
                    argv, allow_name=server.policy != "generic"
                )
                executable = server.real_scancel
            elif command == "scontrol":
                if stdin:
                    raise GuardError("scontrol does not accept standard input")
                canonical_argv = server.validate_scontrol(argv)
                executable = server.real_scontrol
            elif command == "squeue":
                if stdin:
                    raise GuardError("squeue does not accept standard input")
                canonical_argv = server.validate_squeue(argv)
                executable = server.real_squeue
            else:
                if stdin or argv != ["--version"]:
                    raise GuardError(
                        "direct srun execution is disabled; Chaperone may use "
                        "srun only inside a broker-approved allocation"
                    )
                canonical_argv = ["--version"]
                executable = server.real_srun
            if command == "sbatch" and script is not None:
                # prepare_* always emits --export=NONE. Replace it only after
                # validating request environment names, so the allocation gets
                # the broker's fixed HOME/PATH plus the narrow scientific
                # allowlist and nothing else from the agent process.
                canonical_argv[canonical_argv.index("--export=NONE")] = (
                    _explicit_export(environment)
                )
                if directives is None:
                    directives = _submitit_directives(script, server.workspace)
                if gpu_count:
                    excluded_nodes = _merged_excluded_nodes(
                        directives.get("--exclude"), server.exclude_nodes
                    )
                    if excluded_nodes:
                        canonical_argv.append(f"--exclude={excluded_nodes}")
                gpu_reservation = _submission_gpu_reservation(
                    directives,
                    gpu_count,
                    server.max_total_gpus,
                    server.max_requested_gpus_per_submission,
                )
                completed = server.submit(
                    canonical_argv,
                    cwd,
                    script,
                    environment,
                    directives["--job-name"],
                    gpu_reservation,
                )
            else:
                completed = subprocess.run(
                    [executable, *canonical_argv],
                    cwd=cwd,
                    input=script,
                    text=True,
                    capture_output=True,
                    env=environment,
                    timeout=30,
                )
                if (
                    command == "squeue"
                    and completed.returncode == 1
                    and "Invalid job id specified" in completed.stderr
                ):
                    completed = subprocess.CompletedProcess(
                        completed.args, 0, stdout="", stderr=""
                    )
            _send_message(
                self.connection,
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "validated_gpus": gpu_count,
                },
            )
        except Exception as exc:
            _send_message(
                self.connection,
                {"returncode": 2, "stdout": "", "stderr": f"{exc}\n"},
            )


class _BrokerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(
        self,
        socket_path: str,
        real_sbatch: str,
        real_scancel: str,
        real_scontrol: str,
        real_srun: str,
        real_squeue: str,
        workspace: Path,
        slurm_python: Path | None,
        max_total_gpus: int,
        allowed_gpu_type: str,
        policy: str = "submitit",
        workspace_alias: Path | None = None,
        generic_runner: Path | None = None,
        payload_dir: Path | None = None,
        submission_registry: Path | None = None,
        max_requested_gpus_per_submission: int = (
            DEFAULT_MAX_REQUESTED_GPUS_PER_SUBMISSION
        ),
        exclude_nodes: str = "",
    ) -> None:
        self.real_sbatch = real_sbatch
        self.real_scancel = real_scancel
        self.real_scontrol = real_scontrol
        self.real_srun = real_srun
        self.real_squeue = real_squeue
        self.workspace = workspace.resolve(strict=True)
        # Slurm may consult the submitted user's passwd home before starting
        # the batch script. Ensure both fixed writable locations exist even
        # when the broker is constructed outside the normal Harbor bootstrap.
        (self.workspace / ".home").mkdir(parents=True, exist_ok=True)
        (self.workspace / ".tmp").mkdir(parents=True, exist_ok=True)
        self.slurm_python = (
            slurm_python.resolve(strict=True) if slurm_python is not None else None
        )
        self.max_total_gpus = max_total_gpus
        self.max_requested_gpus_per_submission = max_requested_gpus_per_submission
        self.allowed_gpu_type = allowed_gpu_type.lower()
        self.exclude_nodes = _merged_excluded_nodes(None, exclude_nodes)
        if policy not in {"submitit", "generic"}:
            raise GuardError(f"unsupported broker policy {policy!r}")
        self.policy = policy
        self.workspace_alias = (workspace_alias or self.workspace).resolve(strict=False)
        self.generic_runner = (
            generic_runner.resolve(strict=True) if generic_runner is not None else None
        )
        self.payload_dir = (
            payload_dir.resolve(strict=True) if payload_dir is not None else None
        )
        self.scheduler_log_dir = (
            self.payload_dir.parent / "scheduler-logs"
            if self.payload_dir is not None
            else self.workspace / ".slurm-logs"
        )
        self.submission_registry = submission_registry
        if self.submission_registry is not None:
            self.submission_registry.parent.mkdir(parents=True, exist_ok=True)
            self.submission_registry.touch(exist_ok=True)
            self.submission_registry.chmod(0o600)
        if self.policy == "submitit" and self.slurm_python is None:
            raise GuardError("submitit broker requires slurm_python")
        if self.policy == "generic" and (
            self.generic_runner is None or self.payload_dir is None
        ):
            raise GuardError("generic broker requires runner and payload directory")
        self._job_ids: set[str] = set()
        self._job_names: set[str] = set()
        self._gpu_reservations: dict[str, int] = {}
        self._jobs_lock = threading.RLock()
        self._load_submission_registry()
        super().__init__(socket_path, _BrokerHandler)

    def _load_submission_registry(self) -> None:
        """Restore trial ownership and reservations after a broker restart."""
        if self.submission_registry is None:
            return
        try:
            lines = self.submission_registry.read_text().splitlines()
        except OSError as error:
            raise GuardError(
                f"cannot read guarded Slurm submission registry: {error}"
            ) from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                job_id = str(record["job_id"])
                job_name = str(record["job_name"])
                reservation = int(record.get("gpu_reservation", 0))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise GuardError(
                    "invalid guarded Slurm submission registry record "
                    f"{self.submission_registry}:{line_number}: {error}"
                ) from error
            if not re.fullmatch(r"[0-9]+", job_id) or reservation < 0:
                raise GuardError(
                    "invalid guarded Slurm submission registry record "
                    f"{self.submission_registry}:{line_number}"
                )
            self._job_ids.add(job_id)
            self._job_names.add(job_name)
            if reservation:
                self._gpu_reservations[job_id] = reservation

    def _append_submission_registry(
        self, job_id: str, job_name: str, gpu_reservation: int
    ) -> None:
        """Append one crash-durable JSONL record under an interprocess lock."""
        if self.submission_registry is None:
            return
        payload = (
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "job_name": job_name,
                    "policy": self.policy,
                    "gpu_reservation": gpu_reservation,
                    "submitted_at": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        descriptor = os.open(
            self.submission_registry,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            remaining = memoryview(payload.encode())
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short write to guarded Slurm registry")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def resolve_client_path(self, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                candidate = self.workspace / candidate.relative_to(self.workspace_alias)
            except ValueError:
                pass
        else:
            candidate = self.workspace / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise GuardError(f"cannot resolve workspace path {value!r}: {exc}") from exc
        if not _inside(resolved, self.workspace):
            raise GuardError(
                f"sbatch working directory must stay inside {self.workspace}"
            )
        return resolved

    def snapshot_generic_payload(self, payload: str, directives: dict[str, str]) -> str:
        """Freeze agent bytes and return a host-only canonical batch script."""
        if self.generic_runner is None or self.payload_dir is None:
            raise GuardError("generic broker runtime is unavailable")
        payload_path = self.payload_dir / f"payload-{uuid.uuid4().hex}.sh"
        partial = payload_path.with_suffix(".partial")
        partial.write_text(payload)
        partial.chmod(0o400)
        partial.replace(payload_path)

        def contained_log(value: str) -> Path:
            candidate = Path(value)
            return (
                candidate
                if candidate.is_absolute()
                else self.workspace_alias / candidate
            )

        output = contained_log(directives["--output"])
        error = contained_log(directives["--error"])
        return "\n".join(
            (
                "#!/bin/bash",
                "set -euo pipefail",
                f"exec {shlex.quote(str(self.generic_runner))} "
                f"{shlex.quote(str(payload_path))} "
                f"{shlex.quote(str(output))} {shlex.quote(str(error))}",
                "",
            )
        )

    def register_submission(
        self, stdout: str, job_name: str, gpu_reservation: int = 0
    ) -> None:
        match = re.search(r"(?:^|\n)Submitted batch job ([0-9]+)(?:\n|$)", stdout)
        if match is None:
            raise GuardError("sbatch succeeded without returning a Slurm job ID")
        with self._jobs_lock:
            job_id = match.group(1)
            if job_id in self._job_ids:
                return
            self._append_submission_registry(job_id, job_name, gpu_reservation)
            self._job_ids.add(job_id)
            self._job_names.add(job_name)
            if gpu_reservation:
                self._gpu_reservations[job_id] = gpu_reservation

    def _refresh_gpu_reservations_locked(self) -> None:
        if not self._gpu_reservations:
            return
        job_ids = sorted(self._gpu_reservations, key=int)
        completed = subprocess.run(
            [
                self.real_squeue,
                "--noheader",
                f"--jobs={','.join(job_ids)}",
                "--format=%F",
            ],
            cwd=self.workspace,
            text=True,
            capture_output=True,
            env={
                "HOME": str(self.workspace / ".home"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            timeout=30,
        )
        if completed.returncode != 0:
            if (
                completed.returncode == 1
                and not completed.stdout.strip()
                and "Invalid job id specified" in completed.stderr
            ):
                active_job_ids: set[str] = set()
            else:
                raise GuardError(
                    "cannot verify the trial-wide GPU total with squeue: "
                    f"{completed.stderr.strip() or 'unknown squeue error'}"
                )
        else:
            active_job_ids = {
                line.strip()
                for line in completed.stdout.splitlines()
                if re.fullmatch(r"[0-9]+", line.strip())
            }
        self._gpu_reservations = {
            job_id: count
            for job_id, count in self._gpu_reservations.items()
            if job_id in active_job_ids
        }

    def submit(
        self,
        canonical_argv: list[str],
        cwd: Path,
        script: str,
        environment: dict[str, str],
        job_name: str,
        gpu_reservation: int,
    ) -> subprocess.CompletedProcess[str]:
        """Atomically enforce the trial total, submit, and reserve GPU slots."""
        with self._jobs_lock:
            self._refresh_gpu_reservations_locked()
            reserved = sum(self._gpu_reservations.values())
            requested_total = reserved + gpu_reservation
            if requested_total > self.max_total_gpus:
                raise GuardError(
                    "sbatch request rejected: this trial already reserves "
                    f"{reserved} GPU(s), and this submission could reserve "
                    f"{gpu_reservation} more; the entire trial may reserve at "
                    f"most {self.max_total_gpus} GPUs total across all Slurm "
                    "jobs and array tasks"
                )
            completed = subprocess.run(
                [self.real_sbatch, *canonical_argv],
                cwd=cwd,
                input=script,
                text=True,
                capture_output=True,
                env=environment,
                timeout=30,
            )
            if completed.returncode == 0:
                self.register_submission(completed.stdout, job_name, gpu_reservation)
            return completed

    def validate_scancel(
        self, argv: list[str], *, allow_name: bool = True
    ) -> list[str]:
        with self._jobs_lock:
            job_ids = set(self._job_ids)
            job_names = set(self._job_names)
        if len(argv) == 2 and argv[0] in {"-n", "--name"}:
            if not allow_name:
                raise GuardError(
                    "generic scancel requires a broker-registered numeric job ID"
                )
            if argv[1] not in job_names:
                raise GuardError("scancel may target only jobs created by this trial")
            return argv
        if not argv:
            raise GuardError("scancel requires a trial-owned Slurm job ID")
        for value in argv:
            match = re.fullmatch(r"([0-9]+)(?:_[0-9]+)?", value)
            if match is None or match.group(1) not in job_ids:
                raise GuardError("scancel may target only jobs created by this trial")
        return argv

    def validate_squeue(self, argv: list[str]) -> list[str]:
        """Return a query restricted to jobs registered by this broker."""
        with self._jobs_lock:
            owned_job_ids = set(self._job_ids)
        if (
            len(argv) == 5
            and argv[:2] == ["--noheader", "--job"]
            and argv[3:]
            == [
                "--format",
                "%T",
            ]
        ):
            match = re.fullmatch(r"([0-9]+)(?:_[0-9]+)?", argv[2])
            if match is None or match.group(1) not in owned_job_ids:
                raise GuardError("squeue may query only jobs created by this trial")
            return ["--noheader", f"--jobs={argv[2]}", "--format", "%T"]
        if argv not in ([], ["--noheader"]):
            raise GuardError(
                "squeue accepts only an optional --noheader flag or an exact "
                "trial-owned job state query"
            )
        job_ids = sorted(owned_job_ids, key=int)
        # Slurm treats a syntactically valid nonexistent ID as an empty scoped
        # result on this cluster; never fall back to an unscoped query.
        scoped = ",".join(job_ids) if job_ids else "0"
        return [*argv, f"--jobs={scoped}"]

    @staticmethod
    def validate_scontrol(argv: list[str]) -> list[str]:
        if argv == ["show", "config"]:
            return argv
        if (
            len(argv) == 3
            and argv[:2] == ["show", "hostnames"]
            and re.fullmatch(r"[A-Za-z0-9_,.\[\]-]+", argv[2])
        ):
            return argv
        raise GuardError(
            "scontrol is query-only; allowed forms are 'show config' and "
            "'show hostnames NODELIST'"
        )

    def cancel_registered_jobs(self) -> None:
        with self._jobs_lock:
            job_ids = sorted(self._job_ids, key=int)
            self._job_ids.clear()
            self._job_names.clear()
            self._gpu_reservations.clear()
        if not job_ids:
            return
        environment = {
            "HOME": str(self.workspace / ".home"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        }
        try:
            # The durable ownership registry is append-only, so it contains
            # completed jobs as well as live ones.  Passing the entire history
            # to one scancel call lets an expired/invalid ID abort cleanup of
            # still-running children.  Restrict shutdown to the active subset
            # reported by Slurm first; %F maps array tasks back to the parent
            # ID recorded by the broker.
            active = subprocess.run(
                [
                    self.real_squeue,
                    "--noheader",
                    f"--jobs={','.join(job_ids)}",
                    "--format=%F",
                ],
                cwd=self.workspace,
                text=True,
                capture_output=True,
                env=environment,
                timeout=30,
            )
            if active.returncode == 0:
                registered = set(job_ids)
                job_ids = sorted(
                    {
                        line.strip()
                        for line in active.stdout.splitlines()
                        if line.strip() in registered
                    },
                    key=int,
                )
            elif (
                active.returncode == 1
                and not active.stdout.strip()
                and "Invalid job id specified" in active.stderr
            ):
                job_ids = []
            else:
                print(
                    "Guarded Slurm cleanup could not determine active registered "
                    f"jobs; falling back to the full registry: {active.stderr.strip()}",
                    file=sys.stderr,
                    flush=True,
                )
            if not job_ids:
                return
            completed = subprocess.run(
                [self.real_scancel, *job_ids],
                cwd=self.workspace,
                text=True,
                capture_output=True,
                env=environment,
                timeout=30,
            )
            if completed.returncode != 0:
                print(
                    "Guarded Slurm cleanup could not cancel registered jobs "
                    f"{', '.join(job_ids)}: {completed.stderr.strip()}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            print(
                f"Guarded Slurm cleanup failed for {', '.join(job_ids)}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def client(socket_path: str, argv: list[str]) -> int:
    stdin = ""
    if not sys.stdin.isatty():
        stdin = sys.stdin.read(MAX_MESSAGE_BYTES + 1)
        if len(stdin.encode()) > MAX_MESSAGE_BYTES:
            print("sbatch input script is too large", file=sys.stderr)
            return 2
    forwarded_env = {
        name: value for name, value in os.environ.items() if _is_forwarded_name(name)
    }
    request = {
        "command": Path(sys.argv[0]).name,
        "argv": argv,
        "cwd": os.getcwd(),
        "stdin": stdin,
        "env": forwarded_env,
        "sbatch_env": {
            name: value
            for name, value in os.environ.items()
            if name.startswith("SBATCH_")
        },
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(30)
            connection.connect(socket_path)
            _send_message(connection, request)
            response = _receive_message(connection)
    except (OSError, GuardError, json.JSONDecodeError) as exc:
        print(f"guarded sbatch broker unavailable: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(str(response.get("stdout", "")))
    sys.stderr.write(str(response.get("stderr", "")))
    return int(response.get("returncode", 2))


def serve(
    socket_path: Path,
    real_sbatch: Path,
    real_scancel: Path,
    real_scontrol: Path,
    real_srun: Path,
    real_squeue: Path,
    workspace: Path,
    slurm_python: Path | None,
    max_total_gpus: int,
    allowed_gpu_type: str,
    policy: str = "submitit",
    workspace_alias: Path | None = None,
    generic_runner: Path | None = None,
    payload_dir: Path | None = None,
    submission_registry: Path | None = None,
    max_requested_gpus_per_submission: int = (
        DEFAULT_MAX_REQUESTED_GPUS_PER_SUBMISSION
    ),
    exclude_nodes: str = "",
) -> int:
    if max_total_gpus < 1:
        raise SystemExit("--max-total-gpus must be positive")
    if max_requested_gpus_per_submission < 1:
        raise SystemExit("--max-requested-gpus-per-submission must be positive")
    for name, executable in {
        "sbatch": real_sbatch,
        "scancel": real_scancel,
        "scontrol": real_scontrol,
        "srun": real_srun,
        "squeue": real_squeue,
    }.items():
        if not executable.is_file():
            raise SystemExit(f"real {name} binary not found: {executable}")
    if policy == "submitit" and (slurm_python is None or not slurm_python.is_file()):
        raise SystemExit(f"controller slurm_python not found: {slurm_python}")
    if policy == "generic":
        if generic_runner is None or not generic_runner.is_file():
            raise SystemExit(f"controller generic runner not found: {generic_runner}")
        if payload_dir is None or not payload_dir.is_dir():
            raise SystemExit(f"controller payload directory not found: {payload_dir}")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = _BrokerServer(
        str(socket_path),
        str(real_sbatch),
        str(real_scancel),
        str(real_scontrol),
        str(real_srun),
        str(real_squeue),
        workspace,
        slurm_python,
        max_total_gpus,
        allowed_gpu_type,
        policy,
        workspace_alias,
        generic_runner,
        payload_dir,
        submission_registry,
        max_requested_gpus_per_submission,
        exclude_nodes,
    )
    socket_path.chmod(0o600)

    def stop(_signum: int, _frame: Any) -> None:
        # BaseServer.shutdown() must run from a different thread than
        # serve_forever(), which owns this signal handler.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.cancel_registered_jobs()
        server.server_close()
        socket_path.unlink(missing_ok=True)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        parser = argparse.ArgumentParser()
        parser.add_argument("serve")
        parser.add_argument("--socket", type=Path, required=True)
        parser.add_argument("--real-sbatch", type=Path, required=True)
        parser.add_argument("--real-scancel", type=Path, required=True)
        parser.add_argument("--real-scontrol", type=Path, required=True)
        parser.add_argument("--real-srun", type=Path, required=True)
        parser.add_argument("--real-squeue", type=Path, required=True)
        parser.add_argument("--workspace", type=Path, required=True)
        parser.add_argument("--slurm-python", type=Path)
        parser.add_argument("--max-total-gpus", type=int, default=5)
        parser.add_argument(
            "--max-requested-gpus-per-submission",
            type=int,
            default=DEFAULT_MAX_REQUESTED_GPUS_PER_SUBMISSION,
        )
        parser.add_argument("--allowed-gpu-type", default="small")
        parser.add_argument(
            "--exclude-nodes",
            default="",
            help="Validated Slurm node list added to every brokered GPU job.",
        )
        parser.add_argument(
            "--policy", choices=("submitit", "generic"), default="submitit"
        )
        parser.add_argument("--workspace-alias", type=Path)
        parser.add_argument("--generic-runner", type=Path)
        parser.add_argument("--payload-dir", type=Path)
        parser.add_argument("--submission-registry", type=Path)
        args = parser.parse_args()
        return serve(
            args.socket,
            args.real_sbatch,
            args.real_scancel,
            args.real_scontrol,
            args.real_srun,
            args.real_squeue,
            args.workspace,
            args.slurm_python,
            args.max_total_gpus,
            args.allowed_gpu_type,
            args.policy,
            args.workspace_alias,
            args.generic_runner,
            args.payload_dir,
            args.submission_registry,
            args.max_requested_gpus_per_submission,
            args.exclude_nodes,
        )
    return client(os.environ.get("HARBOR_SBATCH_SOCKET", DEFAULT_SOCKET), sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
