"""Stateless monitoring for detached work and ordinary campaign artifacts.

The caller blocks in this module, outside the model loop.  No registration,
monitor thread, receipt, cursor, or other persistent monitor state is created.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence


SLURM_SUCCESS = {"COMPLETED"}
SLURM_FAILURE = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}

# Scientific workers can see the same shared filesystems through either spelling,
# depending on the container mount namespace.  Keep this list deliberately narrow:
# accepting an arbitrary /mnt path would weaken workspace confinement.
_SHARED_PREFIX_ALIASES = (
    (Path("/net"), Path("/mnt/net")),
    (Path("/projects"), Path("/mnt/projects")),
)


class WaitError(ValueError):
    """The requested wait cannot be performed."""


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise WaitError(f"cannot run {command[0]}: {error}") from error


def _slurm_state(value: str) -> str:
    return value.strip().split()[0].split("+")[0].upper() if value.strip() else ""


def active_slurm_state(job_id: str) -> str | None:
    """Return the current squeue state, or None once the job leaves squeue."""
    result = _run(["squeue", "--noheader", "--job", job_id, "--format", "%T"])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise WaitError(f"squeue failed for job {job_id}: {detail}")
    states = [_slurm_state(line) for line in result.stdout.splitlines()]
    states = [state for state in states if state]
    return states[0] if states else None


def accounted_slurm_state(job_id: str) -> str | None:
    """Return the terminal state recorded by sacct, if it is available yet."""
    result = _run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            job_id,
            "--format",
            "JobIDRaw,State",
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise WaitError(f"sacct failed for job {job_id}: {detail}")
    fallback = None
    for line in result.stdout.splitlines():
        fields = line.split("|", 2)
        if len(fields) < 2:
            continue
        row_id, state = fields[0].strip(), _slurm_state(fields[1])
        if not state:
            continue
        if fallback is None:
            fallback = state
        if row_id == job_id or row_id.split(".", 1)[0] == job_id:
            return state
    return fallback


def process_alive(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_wait_request(
    *,
    job_ids: Sequence[str],
    process_ids: Sequence[int],
    completion_markers: Sequence[Path],
    failure_markers: Sequence[Path],
) -> None:
    if not (job_ids or process_ids or completion_markers or failure_markers):
        raise WaitError("provide a Slurm job ID, process ID, or marker path")
    if not job_ids and not process_ids and not completion_markers:
        raise WaitError("a marker-only wait requires at least one completion marker")
    if process_ids and not completion_markers:
        raise WaitError(
            "process waits require at least one completion marker because a detached "
            "process exit code cannot be recovered statelessly"
        )
    if any(process_id <= 0 for process_id in process_ids):
        raise WaitError("process IDs must be positive")


def wait_for_completion(
    *,
    job_ids: Sequence[str] = (),
    process_ids: Sequence[int] = (),
    completion_markers: Sequence[Path] = (),
    failure_markers: Sequence[Path] = (),
    poll_seconds: float = 30.0,
    registration_grace_seconds: float = 30.0,
    timeout_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Block until all work succeeds, any work fails, or an explicit timeout expires."""
    jobs = tuple(str(item).strip() for item in job_ids if str(item).strip())
    processes = tuple(int(item) for item in process_ids)
    complete = tuple(Path(item) for item in completion_markers)
    failed = tuple(Path(item) for item in failure_markers)
    validate_wait_request(
        job_ids=jobs,
        process_ids=processes,
        completion_markers=complete,
        failure_markers=failed,
    )
    if poll_seconds <= 0:
        raise WaitError("poll_seconds must be positive")
    if registration_grace_seconds < 0:
        raise WaitError("registration_grace_seconds cannot be negative")
    if timeout_seconds is not None and timeout_seconds < 0:
        raise WaitError("timeout_seconds cannot be negative")

    started = clock()
    seen_jobs: set[str] = set()
    completed_jobs: dict[str, str] = {}

    def event(status: str, reason: str, **extra: object) -> dict[str, object]:
        return {
            "status": status,
            "reason": reason,
            "elapsed_seconds": round(max(0.0, clock() - started), 3),
            **extra,
        }

    while True:
        present_failures = [str(path) for path in failed if path.exists()]
        if present_failures:
            return event("failed", "failure marker appeared", markers=present_failures)

        markers_done = bool(complete) and all(path.exists() for path in complete)
        if markers_done:
            return event(
                "completed",
                "all completion markers appeared",
                markers=[str(path) for path in complete],
            )

        now = clock()
        for job_id in jobs:
            if job_id in completed_jobs:
                continue
            active = active_slurm_state(job_id)
            if active is not None:
                seen_jobs.add(job_id)
                if active in SLURM_FAILURE:
                    return event(
                        "failed", "Slurm job failed", job_id=job_id, slurm_state=active
                    )
                if active in SLURM_SUCCESS:
                    completed_jobs[job_id] = active
                continue

            accounted = accounted_slurm_state(job_id)
            if accounted is not None:
                seen_jobs.add(job_id)
                if accounted in SLURM_SUCCESS:
                    completed_jobs[job_id] = accounted
                elif accounted in SLURM_FAILURE:
                    return event(
                        "failed",
                        "Slurm job failed",
                        job_id=job_id,
                        slurm_state=accounted,
                    )
            elif job_id not in seen_jobs and now - started >= registration_grace_seconds:
                return event("failed", "Slurm job was not found", job_id=job_id)

        exited = [process_id for process_id in processes if not process_alive(process_id)]
        if exited:
            # Check markers one last time to avoid racing a process that writes its
            # success marker immediately before exiting.
            present_failures = [str(path) for path in failed if path.exists()]
            if present_failures:
                return event("failed", "failure marker appeared", markers=present_failures)
            if complete and all(path.exists() for path in complete):
                return event(
                    "completed",
                    "all completion markers appeared",
                    markers=[str(path) for path in complete],
                )
            return event(
                "failed",
                "detached process exited before writing completion markers",
                process_ids=exited,
            )

        jobs_done = len(completed_jobs) == len(jobs)
        processes_done = not processes
        if jobs and jobs_done and processes_done:
            if complete:
                return event(
                    "failed",
                    "Slurm jobs completed without required completion markers",
                    jobs=completed_jobs,
                )
            return event("completed", "all Slurm jobs completed", jobs=completed_jobs)

        if timeout_seconds is not None and now - started >= timeout_seconds:
            return event("timeout", "explicit wait timeout expired")
        sleep(poll_seconds)


WAIT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "job_ids": {"type": "array", "items": {"type": "string"}},
        "process_ids": {"type": "array", "items": {"type": "integer"}},
        "completion_markers": {"type": "array", "items": {"type": "string"}},
        "failure_markers": {"type": "array", "items": {"type": "string"}},
        "poll_seconds": {"type": "number", "exclusiveMinimum": 0},
        "registration_grace_seconds": {"type": "number", "minimum": 0},
        "timeout_seconds": {"type": "number", "minimum": 0},
    },
}


def _workspace_root(workspace: Path | None) -> Path:
    configured = workspace
    if configured is None:
        raw = os.environ.get("HARBOR_HOST_WORKSPACE")
        configured = Path(raw).expanduser() if raw else Path.cwd()
    if not configured.is_absolute():
        raise WaitError("monitor workspace must be absolute")
    return configured.resolve(strict=False)


def _workspace_spellings(root: Path) -> tuple[Path, ...]:
    """Return trusted mount-namespace spellings for one workspace root."""
    spellings = [root]
    for left, right in _SHARED_PREFIX_ALIASES:
        for source, target in ((left, right), (right, left)):
            try:
                relative = root.relative_to(source)
            except ValueError:
                continue
            spelling = target / relative
            if spelling not in spellings:
                spellings.append(spelling)
    return tuple(spellings)


def _translate_workspace_path(path: Path, root: Path) -> Path:
    """Translate a trusted container/shared-filesystem spelling to ``root``."""
    if not path.is_absolute():
        return root / path
    for spelling in (Path("/workspace"), *_workspace_spellings(root)):
        try:
            return root / path.relative_to(spelling)
        except ValueError:
            continue
    return path


def _workspace_path(raw: str, workspace: Path | None) -> Path:
    path = Path(raw).expanduser()
    root = _workspace_root(workspace)
    path = _translate_workspace_path(path, root)
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise WaitError(f"marker path is outside the workspace: {raw}")
    return resolved


def _allowed_runtime_executables() -> tuple[set[Path], set[Path]]:
    python = Path(sys.executable)
    python_executables = {python, Path("/opt/pipelines/.venv/bin/python")}
    cli_executables = {
        python.parent / "chaperone",
        Path("/opt/pipelines/.venv/bin/chaperone"),
    }
    return python_executables, cli_executables


def _validated_argv(arguments: object, root: Path, cwd: Path) -> list[str]:
    if not isinstance(arguments, list) or not arguments:
        raise WaitError("argv must be a non-empty array")
    if not all(
        isinstance(item, str) and item and "\x00" not in item for item in arguments
    ):
        raise WaitError("argv entries must be non-empty strings without NUL bytes")
    argv = list(arguments)
    executable = Path(argv[0])
    if not executable.is_absolute():
        raise WaitError("the launch executable must be an absolute path")

    python_executables, cli_executables = _allowed_runtime_executables()
    runtime_executable = executable in python_executables | cli_executables
    if runtime_executable:
        resolved = executable
    else:
        resolved = _translate_workspace_path(executable, root).resolve(strict=False)
    inside_workspace = resolved.is_relative_to(root)
    if not runtime_executable and not inside_workspace:
        raise WaitError(
            "launch executable must be the active Chaperone Python/CLI or a "
            "workspace executable"
        )
    if inside_workspace and (
        not resolved.is_file() or not os.access(resolved, os.X_OK)
    ):
        raise WaitError("workspace launch executable is missing or not executable")
    argv[0] = str(resolved)

    if resolved in python_executables:
        script_index = 1
        if len(argv) > 1 and argv[1] == "-u":
            script_index = 2
        if len(argv) <= script_index or argv[script_index].startswith("-"):
            raise WaitError(
                "Chaperone Python requires a workspace script; -c and -m are forbidden"
            )
        script = Path(argv[script_index]).expanduser()
        if script.is_absolute():
            script = _translate_workspace_path(script, root)
        else:
            script = cwd / script
        script = script.resolve(strict=False)
        if not script.is_relative_to(root):
            raise WaitError("Python launch script is outside the workspace")
        if not script.is_file():
            raise WaitError("Python launch script is missing")
        argv[script_index] = str(script)
    return argv


def launch_and_wait(
    *,
    argv: object,
    cwd: str = ".",
    completion_markers: Sequence[str],
    failure_markers: Sequence[str] = (),
    stdout_path: str = "logs/detached-driver.stdout.log",
    stderr_path: str = "logs/detached-driver.stderr.log",
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
    workspace: Path | None = None,
) -> dict[str, object]:
    """Safely launch a detached driver and wait for its terminal marker."""
    if poll_seconds <= 0:
        raise WaitError("poll_seconds must be positive")
    if timeout_seconds is not None and timeout_seconds < 0:
        raise WaitError("timeout_seconds cannot be negative")
    root = _workspace_root(workspace)
    workdir = _workspace_path(cwd, root)
    if not workdir.is_dir():
        raise WaitError(f"launch cwd is not a directory: {cwd}")
    command = _validated_argv(argv, root, workdir)
    complete = [_workspace_path(item, root) for item in completion_markers]
    failed = [_workspace_path(item, root) for item in failure_markers]
    if not complete:
        raise WaitError("launch_and_wait requires a completion marker")
    stdout = _workspace_path(stdout_path, root)
    stderr = _workspace_path(stderr_path, root)
    marker_paths = {*complete, *failed}
    if stdout in marker_paths or stderr in marker_paths:
        raise WaitError("log paths must be distinct from marker paths")
    preexisting = [str(path) for path in marker_paths if path.exists()]
    if preexisting:
        raise WaitError(
            "launch marker paths must not already exist: " + ", ".join(preexisting)
        )
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with stdout.open("ab", buffering=0) as stdout_stream, stderr.open(
        "ab", buffering=0
    ) as stderr_stream:
        try:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=True,
                shell=False,
            )
        except OSError as error:
            raise WaitError(f"could not launch driver: {error}") from error

        while True:
            present_failures = [str(path) for path in failed if path.exists()]
            if present_failures:
                return {
                    "status": "failed",
                    "reason": "failure marker appeared",
                    "process_id": process.pid,
                    "markers": present_failures,
                }
            if all(path.exists() for path in complete):
                return {
                    "status": "completed",
                    "reason": "all completion markers appeared",
                    "process_id": process.pid,
                    "markers": [str(path) for path in complete],
                }
            returncode = process.poll()
            if returncode is not None:
                return {
                    "status": "failed",
                    "reason": "driver exited before writing completion markers",
                    "process_id": process.pid,
                    "returncode": returncode,
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                }
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                return {
                    "status": "timeout",
                    "reason": "explicit wait timeout expired; detached driver remains running",
                    "process_id": process.pid,
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                }
            time.sleep(poll_seconds)


LAUNCH_INPUT_SCHEMA = {
    "type": "object",
    "required": ["argv", "completion_markers"],
    "properties": {
        "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "cwd": {"type": "string"},
        "completion_markers": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "failure_markers": {"type": "array", "items": {"type": "string"}},
        "stdout_path": {"type": "string"},
        "stderr_path": {"type": "string"},
        "poll_seconds": {"type": "number", "exclusiveMinimum": 0},
        "timeout_seconds": {"type": "number", "minimum": 0},
    },
}


INSPECT_INPUT_SCHEMA = {
    "type": "object",
    "required": ["campaign"],
    "additionalProperties": False,
    "properties": {
        "campaign": {
            "type": "string",
            "minLength": 1,
            "description": "Campaign directory inside the monitor workspace.",
        },
        "view": {
            "type": "string",
            "enum": ["summary", "failures", "funnel", "candidates"],
            "default": "summary",
        },
        "max_items": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "default": 5,
        },
        "rank_by": {
            "type": "string",
            "maxLength": 128,
            "description": (
                "Public metric column used only by the candidates view. "
                "Omit to return passing design IDs in file order."
            ),
        },
        "direction": {
            "type": "string",
            "enum": ["min", "max"],
            "default": "max",
        },
    },
}


def inspect_campaign(
    *,
    campaign: object,
    view: object = "summary",
    max_items: object = 5,
    rank_by: object = None,
    direction: object = "max",
    workspace: Path | None = None,
) -> dict[str, object]:
    """Return a bounded, read-only view over existing campaign artifacts."""
    if not isinstance(campaign, str) or not campaign:
        raise WaitError("campaign must be a non-empty workspace path")
    if not isinstance(view, str):
        raise WaitError("view must be a string")
    if rank_by is not None and not isinstance(rank_by, str):
        raise WaitError("rank_by must be a column name")
    if not isinstance(direction, str):
        raise WaitError("direction must be a string")

    from chaperone_cli.campaign_inspect import inspect_campaign as inspect

    return inspect(
        campaign=_workspace_path(campaign, workspace),
        view=view,
        max_items=max_items,
        rank_by=rank_by,
        direction=direction,
    )


def _write_response(request_id: object, *, result=None, error=None) -> None:
    response: dict[str, object] = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        response["result"] = result
    else:
        response["error"] = {"code": -32000, "message": str(error)}
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def serve_mcp(
    *, server_name: str = "chaperone-monitor", workspace: Path | None = None
) -> int:
    """Serve stateless Chaperone monitoring over line-delimited MCP stdio."""
    for raw in sys.stdin:
        request_id = None
        try:
            request = json.loads(raw)
            method = request.get("method")
            request_id = request.get("id")
            if method == "initialize":
                _write_response(
                    request_id,
                    result={
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": server_name, "version": "2.1"},
                    },
                )
            elif method == "tools/list":
                _write_response(
                    request_id,
                    result={
                        "tools": [
                            {
                                "name": "wait",
                                "description": (
                                    "Block outside the model loop until submitted Slurm jobs, "
                                    "detached processes, or marker files finish. This call is "
                                    "stateless and writes no monitor records."
                                ),
                                "inputSchema": WAIT_INPUT_SCHEMA,
                            },
                            {
                                "name": "launch_and_wait",
                                "description": (
                                    "Safely launch a detached workspace driver without a "
                                    "shell and block until its completion marker, failure, "
                                    "exit, or timeout."
                                ),
                                "inputSchema": LAUNCH_INPUT_SCHEMA,
                            },
                            {
                                "name": "inspect",
                                "description": (
                                    "Read a compact campaign summary, failure evidence, "
                                    "pipeline funnel, or public candidate shortlist from "
                                    "ordinary Chaperone artifacts. This call is read-only, "
                                    "bounded, and creates no campaign ledger."
                                ),
                                "inputSchema": INSPECT_INPUT_SCHEMA,
                            },
                        ]
                    },
                )
            elif method == "tools/call":
                params = request.get("params") or {}
                arguments = params.get("arguments") or {}
                name = params.get("name")
                if name == "wait":
                    result = wait_for_completion(
                        job_ids=arguments.get("job_ids", ()),
                        process_ids=arguments.get("process_ids", ()),
                        completion_markers=[
                            _workspace_path(item, workspace)
                            for item in arguments.get("completion_markers", ())
                        ],
                        failure_markers=[
                            _workspace_path(item, workspace)
                            for item in arguments.get("failure_markers", ())
                        ],
                        poll_seconds=float(arguments.get("poll_seconds", 30)),
                        registration_grace_seconds=float(
                            arguments.get("registration_grace_seconds", 30)
                        ),
                        timeout_seconds=(
                            float(arguments["timeout_seconds"])
                            if arguments.get("timeout_seconds") is not None
                            else None
                        ),
                    )
                elif name == "launch_and_wait":
                    result = launch_and_wait(
                        argv=arguments.get("argv"),
                        cwd=arguments.get("cwd", "."),
                        completion_markers=arguments.get("completion_markers", ()),
                        failure_markers=arguments.get("failure_markers", ()),
                        stdout_path=arguments.get(
                            "stdout_path", "logs/detached-driver.stdout.log"
                        ),
                        stderr_path=arguments.get(
                            "stderr_path", "logs/detached-driver.stderr.log"
                        ),
                        poll_seconds=float(arguments.get("poll_seconds", 30)),
                        timeout_seconds=(
                            float(arguments["timeout_seconds"])
                            if arguments.get("timeout_seconds") is not None
                            else None
                        ),
                        workspace=workspace,
                    )
                elif name == "inspect":
                    result = inspect_campaign(
                        campaign=arguments.get("campaign"),
                        view=arguments.get("view", "summary"),
                        max_items=arguments.get("max_items", 5),
                        rank_by=arguments.get("rank_by"),
                        direction=arguments.get("direction", "max"),
                        workspace=workspace,
                    )
                else:
                    raise WaitError(f"unknown tool: {name}")
                _write_response(
                    request_id,
                    result={"content": [{"type": "text", "text": json.dumps(result)}]},
                )
            elif request_id is not None and method != "notifications/initialized":
                raise WaitError(f"unsupported method: {method}")
        except Exception as error:  # Keep serving after malformed calls.
            if request_id is not None:
                _write_response(request_id, error=error)
    return 0


def wait_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chaperone wait",
        description="Wait outside the model loop for detached work to finish.",
    )
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--process-id", action="append", type=int, default=[])
    parser.add_argument("--completion-marker", action="append", type=Path, default=[])
    parser.add_argument("--failure-marker", action="append", type=Path, default=[])
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--registration-grace-seconds", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args(argv)
    try:
        result = wait_for_completion(
            job_ids=args.job_id,
            process_ids=args.process_id,
            completion_markers=[path.expanduser().resolve() for path in args.completion_marker],
            failure_markers=[path.expanduser().resolve() for path in args.failure_marker],
            poll_seconds=args.poll_seconds,
            registration_grace_seconds=args.registration_grace_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except WaitError as error:
        print(json.dumps({"status": "error", "reason": str(error)}))
        return 2
    print(json.dumps(result))
    return {"completed": 0, "failed": 1, "timeout": 124}[str(result["status"])]


def mcp_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chaperone monitor-mcp",
        description="Serve the stateless job waiter over MCP stdio.",
    )
    parser.add_argument("--server-name", default="chaperone-monitor")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args(argv)
    return serve_mcp(server_name=args.server_name, workspace=args.workspace)
