#!/usr/bin/env -S bash -c 'exec "$(dirname "$0")/.venv/bin/python" "$0" "$@"'
"""Submit one Chaperone evaluation through Harbor, monitor it, and score it."""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
import getpass
import hashlib
from pathlib import Path


from environments.apptainer import (
    ApptainerEnvironment,
    _runtime_bind_pairs,
)
from scripts.snapshot_harness import build as build_harness_bundle
from scripts.snapshot_harness import audit_model_instructions
from scripts.snapshot_harness import validate_snapshot
from launch_manifest import build_manifest, mark_submitted, write_manifest
from scripts.trial_audit import write_trial_audit
from task_paths import (
    PRIVATE_TASKS_ENV,
    PUBLIC_TASKS_ROOT,
    assert_private_tree_unreachable,
    check_inputs_match,
    default_chaperone_repo,
    private_tasks_root,
    task_scorer,
)


# Keep the shared filesystem's logical spelling. Resolving /projects on a CPU
# node rewrites it to /mnt/projects, where the host Chaperone .venv is absent.
from site_config import (
    get_site,
    add_site_arguments,
    configure_site,
    policy_instructions,
    scoring_environment,
)

ROOT = Path(os.path.abspath(__file__)).parent
EVALS_ROOT = ROOT.parent


def _jobs_root(override: Path | None = None) -> Path:
    """Where run output lands.

    A single trial's workspace holds ~30 GB of bootstrap (containers, wheels,
    pydeps, caches) plus its campaign structures, and a trial that dies leaves
    its container .tmp behind — one abandoned run reached 173 GB. That does not
    belong on a home quota, so default to scratch and keep the in-repo path
    only as a fallback for hosts without one.
    """
    configured = override or os.environ.get("HARBOR_JOBS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    scratch = Path(f"/net/scratch/{getpass.getuser()}")
    if get_site()["name"] == "digs" and scratch.is_dir():
        return scratch / "chaperone-evals" / "runs"
    return ROOT / "jobs"


JOBS_ROOT = _jobs_root()
TASKS_ROOT = PUBLIC_TASKS_ROOT
ARM_INSTRUCTIONS = {
    "baseline": ROOT / "instruction_baseline.md",
    "harness": ROOT / "instruction_harness.md",
}
# The controller lives inside the Chaperone checkout it evaluates.
DEFAULT_CHAPERONE_REPO = default_chaperone_repo()
DEFAULT_MODEL = "openai/gpt-5.6-sol"
DEFAULT_REASONING = "medium"
DEFAULT_CONTROLLER_MEMORY_GB = 64
REQUIRED_N_PLACEHOLDER = "{{REQUIRED_N}}"
AGENT_HOURS_PLACEHOLDER = "{{AGENT_HOURS}}"
TASK_NAME_PLACEHOLDER = "{{TASK_NAME}}"
_SCORING_USALIGN_ENV = "USALIGN_BIN"
DEFAULT_AGENT_TIMEOUT_SEC = 12 * 60 * 60
ENVIRONMENT_SETUP = ROOT / "templates" / "apptainer_setup.sh"
TERMINAL_FAILURES = {
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
_SCORING_PACKAGES = ("plip==3.0.1", "openbabel==3.2.1", "lxml==6.1.1")
_SCORING_PACKAGE_MARKER = ".harbor-scoring-packages.json"
_SCORING_RUNDIR_BIND_ENV = "CHAPERONE_CONTAINER_BIND_RUNDIR"
_SCORING_AF3_JAX_CACHE_ENV = "AF3_JAX_CACHE_DIR"
# Only these non-secret, site-local settings may cross the held-out scorer
# boundary. Their values come from the checkout's checked-in site config; this
# tuple is an allowlist, not a second copy of the values.
_SCORING_SITE_ENV_NAMES = (
    "HBPLUS_COPY",
    "CCD_MIRROR_PATH",
    "PYROSETTA_PATH",
    "X3DNA_BIN_DIR",
    "X3DNA_DIR",
)
_CONTROLLER_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
}
_SCORING_UV_ENV = {
    "UV_CONCURRENT_BUILDS": "1",
    "UV_CONCURRENT_DOWNLOADS": "1",
    "UV_CONCURRENT_INSTALLS": "1",
}


def _runtime_capabilities():
    return {
        key: tuple(map(Path, values))
        for key, values in get_site()["capabilities"].items()
    }


_HARNESS_CONTAINER_CAPABILITIES = (
    "AlphaFold3 container",
    "Foundry RF3/RFD3 container",
)
_SHARED_JOB_DIRECTORIES = (
    "host-scorers",
    "results",
    "slurm",
    "task-definitions",
    "workspace-seeds",
)


def _runtime_capability_manifest(
    workspace_seed: Path | None = None,
) -> dict[str, object]:
    """Describe controller-verified capabilities exposed to a harness trial."""
    capabilities = [
        {
            "name": label,
            "available": _capability_present(label, alternatives),
            "version": None,
            "reason": None
            if _capability_present(label, alternatives)
            else "controller preflight could not resolve the required artifact",
        }
        for label, alternatives in sorted(_runtime_capabilities().items())
    ]
    monitor_backend = (
        workspace_seed / "src" / "chaperone_cli" / "job_wait.py"
        if workspace_seed is not None
        else None
    )
    monitor_available = False
    if monitor_backend is not None and monitor_backend.is_file():
        source = monitor_backend.read_text()
        monitor_available = all(
            f"def {name}(" in source
            for name in (
                "wait_for_completion",
                "launch_and_wait",
                "inspect_campaign",
                "serve_mcp",
            )
        )
    capabilities.append(
        {
            "name": "campaign-monitor",
            "available": monitor_available,
            "version": "chaperone-monitor-v2.1",
            "tools": ["wait", "launch_and_wait", "inspect"]
            if monitor_available
            else [],
            "reason": None
            if monitor_available
            else "Chaperone campaign monitor backend is unavailable or incomplete",
        }
    )
    return {
        "schema_version": 1,
        "environment": "harbor-chaperone-harness",
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "capabilities": capabilities,
    }


def _capability_present(label: str, alternatives: tuple[Path, ...]) -> bool:
    candidates = [path for path in alternatives if path.is_file()]
    if label == "HBPLUS":
        return any(os.access(path, os.X_OK) for path in candidates)
    return bool(candidates)


def _required_harness_capabilities(
    task: Path,
    workspace_seed: Path | None,
) -> set[str]:
    required = {"campaign-monitor"}
    contract_path = task / "inputs" / "acceptance_contract.json"
    if contract_path.is_file():
        contract = _load_json(contract_path)
        declared = contract.get("required_capabilities", [])
        gates = contract.get("gates", [])
        if not isinstance(declared, list) or not isinstance(gates, list):
            raise SystemExit(f"Invalid public acceptance contract: {contract_path}")
        required.update(str(item) for item in declared)
        for gate in gates:
            if not isinstance(gate, dict):
                raise SystemExit(
                    f"Invalid public acceptance contract gate: {contract_path}"
                )
            capability = gate.get("required_capability")
            if capability:
                required.add(str(capability))
    available = {
        item["name"]: item["available"]
        for item in _runtime_capability_manifest(workspace_seed)["capabilities"]
    }
    unknown = sorted(required - set(available))
    if unknown:
        raise SystemExit(
            "Acceptance contract names unknown capabilities: " + ", ".join(unknown)
        )
    return required


def _prepare_shared_job_directories(attempts: int = 20) -> None:
    """Create shared parents before concurrent Harbor processes start.

    Each Harbor run owns a distinct leaf below ``results``.  The parent is
    shared, though, and recursive ``Path.mkdir`` calls can still race on NFS:
    one process creates the parent while another briefly sees stale metadata
    and turns the otherwise harmless EEXIST into a fatal error.  Establish
    every shared directory explicitly and retry metadata visibility here so
    Harbor only ever creates independent leaf directories.
    """

    directories = (JOBS_ROOT, *(JOBS_ROOT / name for name in _SHARED_JOB_DIRECTORIES))
    for index, directory in enumerate(directories):
        last_error: OSError | None = None
        for attempt in range(attempts):
            try:
                # Only the root may need recursive creation. Its children are
                # exact mkdir calls after the root has become visible.
                directory.mkdir(parents=index == 0, exist_ok=True)
            except (FileExistsError, FileNotFoundError) as error:
                last_error = error
            try:
                if directory.is_dir():
                    break
            except OSError as error:
                last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.1)
        else:
            raise RuntimeError(
                f"Could not establish shared Harbor directory after {attempts} "
                f"attempts: {directory}"
            ) from last_error


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_name(value: str, label: str, maximum: int = 100) -> str:
    if len(value) > maximum or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise SystemExit(
            f"{label} must start with a letter or number, contain only letters, "
            f"numbers, '.', '_', and '-', and be at most {maximum} characters."
        )
    return value


def resolve_task(task_name: str, tasks_root: Path = TASKS_ROOT) -> Path:
    """Resolve an exact task-folder name beneath the configured tasks root."""
    _validate_name(task_name, "Task name", maximum=64)
    task = (tasks_root / task_name).resolve()
    root = tasks_root.resolve()
    if task.parent != root or not task.is_dir():
        available = ", ".join(
            sorted(path.name for path in root.iterdir() if path.is_dir())
        )
        raise SystemExit(
            f"Task folder not found: {task}. Available tasks: {available or 'none'}"
        )
    required = (task / "prompt.md", task / "inputs")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Task folder is incomplete; missing: " + ", ".join(missing))
    if not (task / "inputs").is_dir():
        raise SystemExit(f"Task inputs path is not a directory: {task / 'inputs'}")
    return task


def render_prompt(prompt_text: str, required_n: int, task_name: str = "task") -> str:
    """Substitute the submission count into a task prompt.

    A prompt that hard-codes the count would tell the agent one number while
    --required-n makes the scorer enforce another, so it is refused outright.
    """
    stale = re.search(r"^REQUIRED_N\s*=\s*\d+\s*$", prompt_text, re.MULTILINE)
    if stale is not None:
        raise SystemExit(
            f"{task_name}/prompt.md hard-codes `{stale.group(0).strip()}`; write "
            f"{REQUIRED_N_PLACEHOLDER} so the count tracks --required-n."
        )
    if REQUIRED_N_PLACEHOLDER not in prompt_text and re.search(
        r"\b\d{2,}\s+(?:unique\s+)?(?:designs|rows|sequences)\b", prompt_text
    ):
        print(
            f"WARNING: {task_name}/prompt.md states a literal submission count; "
            f"it cannot track --required-n {required_n}.",
            flush=True,
        )
    return prompt_text.replace(REQUIRED_N_PLACEHOLDER, str(required_n))


def materialize_harbor_task(
    task: Path,
    run_name: str,
    arm: str,
    image: Path,
    memory_gb: int = DEFAULT_CONTROLLER_MEMORY_GB,
    required_n: int = 96,
    agent_timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC,
    workspace_seed: Path | None = None,
) -> Path:
    """Create the Harbor metadata wrapper kept separate from held-out scoring."""
    try:
        instruction = ARM_INSTRUCTIONS[arm]
    except KeyError:
        raise SystemExit(f"Unsupported evaluation arm: {arm}") from None
    if not instruction.is_file():
        raise SystemExit(f"{arm.title()} instruction not found: {instruction}")
    input_source = task / "inputs"
    for path in input_source.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"Task inputs must not contain symlinks: {path}")
    destination = JOBS_ROOT / "task-definitions" / run_name
    # The wrapper is keyed by run name, not job id, so relaunching a name would
    # otherwise reuse the previous attempt's directory: the copytree below dies
    # on an existing environment/inputs, and anything not explicitly rewritten
    # survives — a stale prompt or input silently served to the next run. Trial
    # results stay keyed by job id and are untouched by this.
    shutil.rmtree(destination, ignore_errors=True)
    environment = destination / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    if arm == "harness":
        environment_files = environment / "files"
        environment_files.mkdir()
        setup_script = environment_files / "setup.sh"
        setup_script.write_text(
            ENVIRONMENT_SETUP.read_text().replace(
                "{{WORKER_PYTHON}}", shlex.quote(get_site()["worker_python"])
            )
        )
        setup_script.chmod(0o755)
        public_inputs = destination / "inputs"
        shutil.copytree(input_source, public_inputs)
        contract_path = public_inputs / "acceptance_contract.json"
        public_scorer = public_inputs / "public_score.py"
        if contract_path.is_file() or public_scorer.is_file():
            if not contract_path.is_file() or not public_scorer.is_file():
                raise SystemExit(
                    "Public acceptance contract and scorer must be supplied together"
                )
            contract = _load_json(contract_path)
            scorer_digest = hashlib.sha256(public_scorer.read_bytes()).hexdigest()
            if contract.get("scorer_sha256") != scorer_digest:
                raise SystemExit(
                    "Public acceptance contract scorer_sha256 does not match public_score.py"
                )
            contract["required_n"] = required_n
            contract_path.write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n"
            )
        (public_inputs / "runtime_capabilities.json").write_text(
            json.dumps(
                _runtime_capability_manifest(workspace_seed),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        # Harbor's stock prebuilt-image flow copies environment/ into the
        # container workdir after startup. Supply only the selected task's
        # public inputs; the scorer and all harness assets stay outside it.
        shutil.copytree(input_source, environment / "inputs")
    tests = destination / "tests"
    tests.mkdir(exist_ok=True)
    test_script = tests / "test.sh"
    test_script.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Held-out scoring runs after Harbor agent completion.' >&2\n"
        "exit 2\n"
    )
    test_script.chmod(0o755)
    # One set of flags drives both what the agent is told and what Harbor
    # enforces. Placeholders prevent the generated prompt and runtime contract
    # from drifting apart.
    instruction_text = instruction.read_text()
    expected_placeholders = [REQUIRED_N_PLACEHOLDER, AGENT_HOURS_PLACEHOLDER]
    if arm == "harness":
        expected_placeholders.append(TASK_NAME_PLACEHOLDER)
    missing_placeholders = [
        placeholder
        for placeholder in expected_placeholders
        if placeholder not in instruction_text
    ]
    if missing_placeholders:
        raise SystemExit(
            f"{instruction} does not contain {', '.join(missing_placeholders)}; "
            "the generated runtime contract could not be injected."
        )
    agent_hours = agent_timeout_sec / 3600
    (destination / "instruction.md").write_text(
        instruction_text.replace(REQUIRED_N_PLACEHOLDER, str(required_n))
        .replace(AGENT_HOURS_PLACEHOLDER, f"{agent_hours:g}")
        .replace(TASK_NAME_PLACEHOLDER, task.name)
        .replace("{{SLURM_POLICY}}", policy_instructions(arm))
        .replace("{{WORKER_PYTHON}}", shlex.quote(get_site()["worker_python"]))
    )
    # The prompt carries the same count placeholder as the arm instruction.
    # Render and serve this copy so --required-n cannot ask for one count in
    # the instruction and another in the task-specific contract.
    (destination / "prompt.md").write_text(
        render_prompt((task / "prompt.md").read_text(), required_n, task.name)
    )
    try:
        audit_model_instructions(
            [destination / "instruction.md", destination / "prompt.md"],
            workspace_seed if arm == "harness" else None,
        )
    except RuntimeError as error:
        raise SystemExit(
            f"Unsafe model-facing bookkeeping contract: {error}"
        ) from error
    if arm == "harness":
        environment_variables = """HOME = "/workspace/.home"
TMPDIR = "/workspace/.tmp"
XDG_CACHE_HOME = "/workspace/.home/.cache"
XDG_CONFIG_HOME = "/workspace/.home/.config"
MPLCONFIGDIR = "/workspace/.home/.config/matplotlib"
PYTHONPATH = "/workspace/chaperone/src"
CHAPERONE_COMPUTE_BACKEND = "auto"
CHAPERONE_CONTAINER_BACKEND = "apptainer"
CHAPERONE_CONTAINER_BIND_RUNDIR = "true"
CHAPERONE_CONTAINER_BIND_WORKSPACE = "true"
"""
        description = f"Chaperone evaluation: {task.name}"
        keywords = '["chaperone", "evaluation", "gpu"]'
        benchmark_role = "chaperone-agent-evaluation"
        mcp_server_config = """
[[environment.mcp_servers]]
name = "campaign-monitor"
transport = "stdio"
command = "{{WORKER_PYTHON}}"
args = ["-m", "chaperone_cli.main", "monitor-mcp", "--server-name", "campaign-monitor"]
""".replace('"{{WORKER_PYTHON}}"', json.dumps(get_site()["worker_python"]))
    else:
        # Harbor and Codex keep their control files outside the workspace. The
        # baseline workspace starts with only the selected task's public inputs.
        # Harbor's codex agent bootstraps node with nvm, then sources
        # "$HOME/.nvm/nvm.sh". nvm's installer only creates NVM_DIR when it
        # equals its own default, and that default is $XDG_CONFIG_HOME/nvm
        # whenever XDG_CONFIG_HOME is set:
        #
        #   nvm_default_install_dir() {
        #     [ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" \
        #                                  || printf %s "${XDG_CONFIG_HOME}/nvm"
        #   }
        #   if [ "${NVM_DIR}" = "$(nvm_default_install_dir)" ]; then mkdir ...
        #   else "You have $NVM_DIR set to ..., but that directory does not
        #   exist."; exit 1
        #
        # So exporting XDG_CONFIG_HOME sends the install to .config/nvm and the
        # source finds nothing, while pinning NVM_DIR against a set
        # XDG_CONFIG_HOME trips the guard instead. Both fail setup before the
        # agent starts. Leave XDG_CONFIG_HOME unset — it would only restate the
        # value XDG already derives from the sandboxed HOME — so the default
        # becomes $HOME/.nvm, NVM_DIR matches it, and nvm creates it itself.
        environment_variables = """HOME = "/tmp/baseline-home"
TMPDIR = "/tmp"
XDG_CACHE_HOME = "/tmp/baseline-home/.cache"
NVM_DIR = "/tmp/baseline-home/.nvm"
"""
        description = f"Baseline evaluation: {task.name}"
        keywords = '["baseline", "evaluation"]'
        benchmark_role = "baseline-agent-evaluation"
        # Baseline measures the raw default agent/runtime. Do not inject any
        # Chaperone harness tools or MCP servers.
        mcp_server_config = ""
    environment_variables += f'HARBOR_AGENT_TIMEOUT_SEC = "{agent_timeout_sec}"\n'
    # build_timeout_sec: Harbor's own default is 600 s. Environment start
    # installs the control-server packages over the network while every
    # sibling trial hammers the same shared filesystem; 300 s lost a
    # baseline attempt to EnvironmentStartTimeoutError mid-run.
    (destination / "task.toml").write_text(
        f"""schema_version = "1.3"
artifacts = ["/workspace/submission"]

[task]
name = "local/{task.name}"
description = "{description}"
keywords = {keywords}

[metadata]
difficulty = "hard"
category = "scientific-computing"
benchmark_role = "{benchmark_role}"

[agent]
timeout_sec = {agent_timeout_sec}

[environment]
docker_image = "{image.expanduser().resolve()}"
network_mode = "public"
cpus = 4
memory_mb = {memory_gb * 1024}
storage_mb = 102400
gpus = 0
build_timeout_sec = 900
workdir = "/workspace"

[environment.env]
{environment_variables}
{mcp_server_config}

[verifier]
timeout_sec = 60
"""
    )
    return destination


def _parse_job_id(output: str) -> str:
    for line in reversed(output.splitlines()):
        match = re.fullmatch(r"\s*(\d+)(?:;[^\s]+)?\s*", line)
        if match:
            return match.group(1)
        match = re.search(r"Submitted batch job (\d+)", line)
        if match:
            return match.group(1)
    raise SystemExit(f"Could not parse a Slurm job ID from:\n{output}")


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    resource_failures = 0
    while True:
        try:
            return subprocess.run(command, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise SystemExit(f"Required command not found: {command[0]}") from exc
        except BlockingIOError as exc:
            if exc.errno != errno.EAGAIN:
                raise
            resource_failures += 1
            if resource_failures == 1 or resource_failures % 60 == 0:
                print(
                    f"Waiting to run {command[0]} after transient process "
                    f"pressure ({resource_failures} failed forks).",
                    file=sys.stderr,
                    flush=True,
                )
            # Detached scorer watchers share a deliberately small process
            # budget on the login host. A short-lived squeue fork can fail
            # while another watcher is querying Slurm; that pressure must not
            # permanently discard this trial's only scorer.
            time.sleep(min(0.1 * 2 ** min(resource_failures - 1, 6), 5.0))


def _slurm_state(job_id: str) -> str | None:
    queued = _run_capture(["squeue", "--noheader", "--jobs", job_id, "--format=%T"])
    if queued.returncode == 0 and queued.stdout.strip():
        return queued.stdout.splitlines()[0].strip().upper()

    accounted = _run_capture(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            job_id,
            "--format=State",
        ]
    )
    if accounted.returncode != 0:
        return None
    states = [line.split("|", 1)[0].strip() for line in accounted.stdout.splitlines()]
    if not states or not states[0]:
        return None
    # sacct may annotate terminal states, e.g. ``CANCELLED by 2597`` or
    # ``OUT_OF_MEMORY+``. Monitoring only needs Slurm's leading state token.
    return states[0].split("+", 1)[0].split()[0].upper()


def _wait_for_job(job_id: str, poll_seconds: float) -> None:
    previous: str | None = None
    while True:
        state = _slurm_state(job_id)
        if state != previous:
            print(f"Slurm job {job_id}: {state or 'waiting for status'}", flush=True)
            previous = state
        if state == "COMPLETED":
            return
        if state in TERMINAL_FAILURES:
            raise SystemExit(f"Slurm job {job_id} ended with state {state}.")
        time.sleep(poll_seconds)


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def _successful_trials(job_dir: Path, settle_seconds: float = 120.0) -> list[Path]:
    """Collect the trials that are ready to score.

    That is every trial Harbor finished cleanly, plus any trial whose agent
    already delivered a ``designs.csv`` even though Harbor's result record is
    failed or missing.

    Scoring starts the moment Slurm reports the outer job COMPLETED, which can
    beat the worker's ``result.json`` onto this shared NFS mount: a finished
    run then looks empty ("failures=none") and its designs are never scored.
    Give the mount time to catch up, but only while no trial result is visible
    at all -- a run whose trials genuinely all failed still answers promptly.
    """
    deadline = time.monotonic() + settle_seconds
    while True:
        successful, failures, saw_result = _scan_trials(job_dir)
        if successful or saw_result or time.monotonic() >= deadline:
            break
        time.sleep(5)
    if not successful:
        raise SystemExit(
            f"No successful Harbor trials in {job_dir}; failures={failures or 'none'}"
        )
    return successful


def _submission_folders(trial: Path) -> tuple[Path, Path]:
    return (
        trial / "artifacts" / "workspace" / "submission",
        trial / "workspace" / "submission",
    )


def _scan_trials(job_dir: Path) -> tuple[list[Path], list[str], bool]:
    """One pass over a run folder: (scoreable, failed names, saw any result).

    A delivered submission remains scoreable when Harbor or the outer Slurm
    allocation fails during teardown. Agent work is immutable at that point;
    requiring a clean ``result.json`` would incorrectly force a new agent run.
    """
    if not job_dir.is_dir():
        raise SystemExit(f"Harbor run folder was not created: {job_dir}")
    successful: list[Path] = []
    failures: list[str] = []
    saw_result = False
    for trial in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        delivered = any(
            (folder / "designs.csv").is_file() for folder in _submission_folders(trial)
        )
        result_path = trial / "result.json"
        if not result_path.is_file():
            if delivered:
                successful.append(trial)
            continue
        saw_result = True
        result = _load_json(result_path)
        if delivered or (
            result.get("exception_info") is None
            and result.get("agent_execution") is not None
        ):
            successful.append(trial)
        else:
            failures.append(trial.name)
    return successful, failures, saw_result


def _submission(trial: Path) -> tuple[Path, Path | None]:
    for folder in _submission_folders(trial):
        designs = folder / "designs.csv"
        if not designs.is_file():
            continue
        try:
            with designs.open(newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
        except (OSError, csv.Error, UnicodeError) as exc:
            raise SystemExit(f"Could not read {designs}: {exc}") from exc
        if not header:
            raise SystemExit(f"Submission CSV has no header: {designs}")
        # Auxiliary files remain accepted for compatibility, but automatic
        # Harbor scoring ignores model-authored cost telemetry.
        return designs, None
    raise SystemExit(f"The successful trial has no submission/designs.csv: {trial}")


def _submission_terminal_error(
    designs: Path,
    required_n: int,
    *,
    require_independent_lineage: bool = False,
    public_contract: Path | None = None,
    sequence_columns: tuple[str, ...] | None = None,
) -> str | None:
    """Validate the agent's structured campaign outcome against its CSV."""
    terminal_path = designs.with_name("terminal_report.json")
    if not terminal_path.is_file():
        return f"required structured terminal report is missing: {terminal_path}"
    try:
        terminal = json.loads(terminal_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return f"invalid structured terminal report: {error}"
    if not isinstance(terminal, dict):
        return "structured terminal report must be a JSON object"
    required_fields = {
        "schema_version",
        "status",
        "required_n",
        "n_submitted",
        "delivery_valid",
        "target_count_met",
        "public_gate_pass_count",
        "errors",
    }
    missing = required_fields - set(terminal)
    if missing:
        return f"structured terminal report is missing fields: {sorted(missing)}"
    try:
        submitted = _submission_row_count(designs, sequence_columns)
    except (OSError, ValueError) as error:
        return f"invalid submission sequence column: {error}"
    if set(terminal) != required_fields:
        return "structured terminal report contains unknown fields"
    if terminal["schema_version"] != 2:
        return "structured terminal report schema_version must be 2"
    if terminal["status"] not in {"complete", "contract_incomplete", "failed"}:
        return f"invalid structured terminal status: {terminal['status']}"
    if terminal["required_n"] != required_n:
        return (
            "structured terminal required_n mismatch: "
            f"{terminal['required_n']} != {required_n}"
        )
    if terminal["n_submitted"] != submitted:
        return (
            "structured terminal n_submitted mismatch: "
            f"{terminal['n_submitted']} != {submitted}"
        )
    if require_independent_lineage:
        lineage_error = _independent_lineage_error(designs)
        if lineage_error is not None:
            return lineage_error
    if public_contract is not None:
        public_score_error = _public_score_error(
            designs,
            submitted,
            terminal["public_gate_pass_count"],
            public_contract,
        )
        if public_score_error is not None:
            return public_score_error
    if not isinstance(terminal["delivery_valid"], bool):
        return "structured terminal delivery_valid must be boolean"
    if not isinstance(terminal["target_count_met"], bool):
        return "structured terminal target_count_met must be boolean"
    public_pass = terminal["public_gate_pass_count"]
    if (
        isinstance(public_pass, bool)
        or not isinstance(public_pass, int)
        or public_pass < 0
    ):
        return "structured terminal public_gate_pass_count must be nonnegative integer"
    if public_pass > submitted:
        return "structured terminal public_gate_pass_count exceeds n_submitted"
    if not isinstance(terminal["errors"], list) or any(
        not isinstance(item, str) or not item.strip() for item in terminal["errors"]
    ):
        return "structured terminal errors must be an array of non-empty strings"
    target_count_met = submitted == required_n
    if target_count_met != terminal["target_count_met"]:
        return "structured terminal target_count_met disagrees with submission count"
    if terminal["delivery_valid"] and target_count_met and terminal["errors"]:
        return "complete structured terminal report must have an empty errors array"
    if (not terminal["delivery_valid"] or not target_count_met) and not terminal[
        "errors"
    ]:
        return "non-complete structured terminal report must explain its errors"
    if not terminal["delivery_valid"]:
        expected_status = "failed"
    elif target_count_met:
        expected_status = "complete"
    else:
        expected_status = "contract_incomplete"
    if terminal["status"] != expected_status:
        return (
            f"structured terminal status must be {expected_status} for "
            f"delivery_valid={terminal['delivery_valid']} and "
            f"{submitted}/{required_n} submitted designs"
        )
    if not terminal["delivery_valid"]:
        return "submission delivery is invalid: " + "; ".join(terminal["errors"])
    if not target_count_met:
        return f"submission contract incomplete: {submitted}/{required_n} designs"
    return None


def _submission_row_count(
    designs: Path, sequence_columns: tuple[str, ...] | None = None
) -> int:
    """Count submitted designs after checking the task's sequence columns.

    Most tasks submit one designed chain using ``sequence`` (or the legacy
    ``sequence_A`` spelling). Multi-chain design tasks instead name every
    designed chain explicitly, such as ``sequence_A`` and ``sequence_B``.
    The held-out task scorer performs full schema and sequence validation,
    while this controller guard only
    verifies that the terminal report counts rows from the correct columns.
    """
    if sequence_columns is None:
        return len(_submission_sequences(designs))
    if not sequence_columns:
        raise ValueError("task sequence column list must not be empty")
    with designs.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(sequence_columns) - fieldnames
        if missing:
            raise ValueError(f"missing task sequence columns: {sorted(missing)}")
        return sum(1 for _ in reader)


def _submission_sequences(designs: Path) -> list[str]:
    """Read the canonical or legacy single-chain sequence column."""
    with designs.open(newline="") as handle:
        reader = csv.DictReader(handle)
        sequence_columns = {"sequence", "sequence_A"} & set(reader.fieldnames or ())
        if not sequence_columns:
            raise ValueError("missing 'sequence' or legacy 'sequence_A'")
        if len(sequence_columns) != 1:
            raise ValueError(
                "supply either 'sequence' or legacy 'sequence_A', not both"
            )
        sequence_column = sequence_columns.pop()
        return [row[sequence_column] for row in reader]


def _independent_lineage_error(designs: Path) -> str | None:
    """Validate a task's one-sequence/one-de-novo-backbone sidecar."""
    lineage_path = designs.with_name("lineage.jsonl")
    if not lineage_path.is_file():
        return f"required lineage sidecar is missing: {lineage_path}"
    try:
        sequences = _submission_sequences(designs)
        records = [
            json.loads(line)
            for line in lineage_path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return f"invalid lineage sidecar: {error}"
    if len(records) != len(sequences):
        return "lineage sidecar row count does not match designs.csv"
    fields = {
        "sequence",
        "backbone_path",
        "backbone_sha256",
        "generator",
        "generator_version",
        "run_id",
        "seed",
        "parent_sequence",
        "parent_backbone_sha256",
    }
    backbone_digests: set[str] = set()
    backbones_root = (lineage_path.parent / "backbones").resolve()
    if not backbones_root.is_dir() or (lineage_path.parent / "backbones").is_symlink():
        return "required backbone artifact directory is missing or unsafe"
    for index, (sequence, record) in enumerate(zip(sequences, records, strict=True)):
        if not isinstance(record, dict) or set(record) != fields:
            return f"lineage row {index + 1} has an invalid schema"
        if record["sequence"] != sequence:
            return f"lineage row {index + 1} sequence does not match designs.csv"
        relative_path = record["backbone_path"]
        if not isinstance(relative_path, str) or not relative_path.strip():
            return f"lineage row {index + 1} has an invalid backbone_path"
        candidate = Path(relative_path)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or candidate.parts[0] != "backbones"
            or ".." in candidate.parts
        ):
            return f"lineage row {index + 1} backbone_path must be under backbones/"
        backbone = (lineage_path.parent / candidate).resolve()
        if (
            not backbone.is_relative_to(backbones_root)
            or not backbone.is_file()
            or backbone.is_symlink()
        ):
            return f"lineage row {index + 1} backbone artifact is missing or unsafe"
        digest = record["backbone_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            return f"lineage row {index + 1} has an invalid backbone_sha256"
        if digest != hashlib.sha256(backbone.read_bytes()).hexdigest():
            return f"lineage row {index + 1} backbone_sha256 does not match artifact"
        try:
            artifact_sequence, canonical_digest = _canonical_backbone(backbone)
        except (OSError, ValueError) as error:
            return f"lineage row {index + 1} has an invalid backbone artifact: {error}"
        if artifact_sequence != sequence:
            return (
                f"lineage row {index + 1} backbone sequence does not match designs.csv"
            )
        if canonical_digest in backbone_digests:
            return "lineage sidecar contains duplicate backbone digests"
        backbone_digests.add(canonical_digest)
        for field in ("generator", "generator_version", "run_id"):
            if not isinstance(record[field], str) or not record[field].strip():
                return f"lineage row {index + 1} has invalid {field}"
        if isinstance(record["seed"], bool) or not isinstance(record["seed"], int):
            return f"lineage row {index + 1} has invalid seed"
        if (
            record["parent_sequence"] is not None
            or record["parent_backbone_sha256"] is not None
        ):
            return f"lineage row {index + 1} is not an independent de novo design"
    return None


_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _canonical_backbone(path: Path) -> tuple[str, str]:
    """Parse a PDB/mmCIF and hash sequence plus rigid-transform-invariant geometry."""
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        residues = _pdb_backbone(path)
    elif suffix in {".cif", ".mmcif"}:
        residues = _cif_backbone(path)
    else:
        raise ValueError("backbone artifact must be PDB or mmCIF")
    if not residues:
        raise ValueError("no protein backbone atoms found")
    sequence = "".join(_THREE_TO_ONE.get(name, "") for name, _ in residues)
    if len(sequence) != len(residues):
        raise ValueError("backbone contains a non-canonical protein residue")
    coordinates = [atoms[name] for _, atoms in residues for name in ("N", "CA", "C")]
    digest = hashlib.sha256()
    for left in range(len(coordinates)):
        x1, y1, z1 = coordinates[left]
        for right in range(left + 1, len(coordinates)):
            x2, y2, z2 = coordinates[right]
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            digest.update(f"{distance:.2f},".encode())
    return sequence, digest.hexdigest()


def _pdb_backbone(
    path: Path,
) -> list[tuple[str, dict[str, tuple[float, float, float]]]]:
    residues: dict[
        tuple[str, str, str], tuple[str, dict[str, tuple[float, float, float]]]
    ] = {}
    for line in path.read_text(errors="strict").splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        atom = line[12:16].strip()
        alt = line[16:17]
        if atom not in {"N", "CA", "C"} or alt not in {" ", "A"}:
            continue
        residue = line[17:20].strip().upper()
        key = (line[21:22], line[22:26].strip(), line[26:27])
        try:
            coordinate = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError as error:
            raise ValueError("invalid PDB coordinate") from error
        stored_name, atoms = residues.setdefault(key, (residue, {}))
        if stored_name != residue or atom in atoms:
            raise ValueError("ambiguous PDB backbone residue")
        atoms[atom] = coordinate
    return _complete_backbone_residues(residues.values())


def _cif_backbone(
    path: Path,
) -> list[tuple[str, dict[str, tuple[float, float, float]]]]:
    lines = path.read_text(errors="strict").splitlines()
    residues: dict[
        tuple[str, str, str], tuple[str, dict[str, tuple[float, float, float]]]
    ] = {}
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("_"):
            headers.append(lines[index].strip())
            index += 1
        if not headers or not all(
            header.startswith("_atom_site.") for header in headers
        ):
            continue
        columns = {
            header.split(".", 1)[1]: position for position, header in enumerate(headers)
        }
        required = {"group_PDB", "Cartn_x", "Cartn_y", "Cartn_z"}
        if not required.issubset(columns):
            raise ValueError("mmCIF atom_site loop lacks coordinates")
        while index < len(lines):
            raw = lines[index].strip()
            if not raw or raw.startswith("#") or raw == "loop_" or raw.startswith("_"):
                break
            values = shlex.split(raw, comments=False, posix=True)
            if len(values) != len(headers):
                raise ValueError("multiline or malformed mmCIF atom_site row")
            if values[columns["group_PDB"]].upper() != "ATOM":
                index += 1
                continue
            atom = _cif_value(values, columns, "auth_atom_id", "label_atom_id")
            alt = _cif_value(values, columns, "label_alt_id", default=".")
            if atom not in {"N", "CA", "C"} or alt not in {".", "?", "A"}:
                index += 1
                continue
            residue = _cif_value(
                values, columns, "auth_comp_id", "label_comp_id"
            ).upper()
            chain = _cif_value(values, columns, "auth_asym_id", "label_asym_id")
            number = _cif_value(values, columns, "auth_seq_id", "label_seq_id")
            insertion = _cif_value(values, columns, "pdbx_PDB_ins_code", default="?")
            key = (chain, number, insertion)
            try:
                coordinate = tuple(
                    float(values[columns[field]])
                    for field in ("Cartn_x", "Cartn_y", "Cartn_z")
                )
            except ValueError as error:
                raise ValueError("invalid mmCIF coordinate") from error
            stored_name, atoms = residues.setdefault(key, (residue, {}))
            if stored_name != residue or atom in atoms:
                raise ValueError("ambiguous mmCIF backbone residue")
            atoms[atom] = coordinate  # type: ignore[assignment]
            index += 1
        break
    return _complete_backbone_residues(residues.values())


def _cif_value(
    values: list[str],
    columns: dict[str, int],
    *names: str,
    default: str | None = None,
) -> str:
    for name in names:
        if name in columns:
            return values[columns[name]]
    if default is not None:
        return default
    raise ValueError("mmCIF atom_site loop lacks " + "/".join(names))


def _complete_backbone_residues(
    values: object,
) -> list[tuple[str, dict[str, tuple[float, float, float]]]]:
    residues = list(values)  # insertion order is the polymer order in PDB/mmCIF
    if any(set(atoms) != {"N", "CA", "C"} for _, atoms in residues):
        raise ValueError("every protein residue must contain N, CA, and C")
    return residues


def _public_score_error(
    designs: Path,
    submitted: int,
    reported_passers: object,
    public_contract: Path,
) -> str | None:
    score_path = designs.with_name("public_score.json")
    if not score_path.is_file():
        return f"required public scorer output is missing: {score_path}"
    try:
        score = json.loads(score_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return f"invalid public scorer output: {error}"
    if not isinstance(score, dict):
        return "public scorer output must be a JSON object"
    contract = _load_json(public_contract)
    scorer_sha256 = contract.get("scorer_sha256")
    scorer_relative = contract.get("scorer_path")
    if not isinstance(scorer_sha256, str) or not isinstance(scorer_relative, str):
        return "public acceptance contract has no scorer path/hash"
    contract_root = public_contract.parent.resolve()
    scorer = (contract_root / scorer_relative).resolve()
    if not scorer.is_relative_to(contract_root) or not scorer.is_file():
        return "public acceptance contract scorer path is invalid"
    if hashlib.sha256(scorer.read_bytes()).hexdigest() != scorer_sha256:
        return "public acceptance contract scorer hash does not match its file"
    metrics = designs.with_name("public_metrics.csv")
    if not metrics.is_file():
        return f"required public metrics artifact is missing: {metrics}"
    try:
        with tempfile.TemporaryDirectory(prefix="public-score-") as temporary:
            recomputed_path = Path(temporary) / "public_score.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(scorer),
                    str(metrics),
                    "--output",
                    str(recomputed_path),
                ],
                text=True,
                capture_output=True,
                timeout=60,
            )
            if completed.returncode:
                detail = (completed.stderr or completed.stdout).strip()
                return f"pinned public scorer failed: {detail}"
            recomputed = json.loads(recomputed_path.read_text())
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        return f"could not rerun pinned public scorer: {error}"
    if score != recomputed:
        return "public scorer output does not match controller recomputation"
    expected_fields = {
        "schema_version",
        "scorer_sha256",
        "n_scored",
        "gate_counts",
        "joint_passers",
        "per_design",
    }
    if set(score) != expected_fields:
        return "public scorer output has an invalid schema"
    if score.get("schema_version") != 1:
        return "public scorer output schema_version must be 1"
    if score.get("scorer_sha256") != scorer_sha256:
        return "public scorer output does not match the pinned scorer hash"
    if score.get("n_scored") != submitted:
        return "public scorer n_scored does not match designs.csv"
    if score.get("joint_passers") != reported_passers:
        return "terminal public_gate_pass_count does not match public scorer output"
    per_design = score.get("per_design")
    if not isinstance(per_design, list) or len(per_design) != submitted:
        return "public scorer per_design rows do not match designs.csv"
    try:
        sequences = _submission_sequences(designs)
    except (OSError, ValueError) as error:
        return f"cannot read designs for public score validation: {error}"
    gate_names: set[str] | None = None
    computed_joint = 0
    computed_gate_passes: dict[str, int] = {}
    for index, (sequence, row) in enumerate(zip(sequences, per_design, strict=True)):
        if not isinstance(row, dict) or set(row) != {
            "sequence",
            "metrics",
            "gates",
            "pass",
        }:
            return f"public scorer per_design row {index + 1} has an invalid schema"
        if row["sequence"] != sequence:
            return f"public scorer per_design row {index + 1} sequence mismatch"
        if not isinstance(row["metrics"], dict) or not isinstance(row["gates"], dict):
            return f"public scorer per_design row {index + 1} metrics/gates are invalid"
        if set(row["metrics"]) != set(row["gates"]):
            return f"public scorer per_design row {index + 1} gate/metric mismatch"
        if any(not isinstance(value, bool) for value in row["gates"].values()):
            return f"public scorer per_design row {index + 1} gates are not boolean"
        if not isinstance(row["pass"], bool) or row["pass"] != all(
            row["gates"].values()
        ):
            return (
                f"public scorer per_design row {index + 1} joint pass is inconsistent"
            )
        names = set(row["gates"])
        if gate_names is None:
            gate_names = names
            computed_gate_passes = {name: 0 for name in names}
        elif names != gate_names:
            return "public scorer per_design gate sets are inconsistent"
        for name, passed in row["gates"].items():
            computed_gate_passes[name] += int(passed)
        computed_joint += int(row["pass"])
    if score["joint_passers"] != computed_joint:
        return "public scorer joint_passers is inconsistent with per_design rows"
    gate_counts = score["gate_counts"]
    if not isinstance(gate_counts, dict) or set(gate_counts) != (gate_names or set()):
        return "public scorer gate_counts does not match per_design gates"
    for name, counts in gate_counts.items():
        if counts != {"passed": computed_gate_passes[name], "total": submitted}:
            return f"public scorer gate_counts is inconsistent for {name}"
    return None


def _scoring_pythonpath(chaperone_repo: Path) -> str:
    """Expose Chaperone plus held-out shared scorer machinery to host Python."""
    entries = [chaperone_repo / "src", EVALS_ROOT]
    library = chaperone_repo / ".venv" / "lib"
    if library.is_dir():
        entries.extend(
            path
            for path in sorted(library.glob("python*/site-packages"))
            if path.is_dir()
        )
    return os.pathsep.join(str(path) for path in entries)


def _host_scoring_python(chaperone_repo: Path) -> Path:
    """Return a host interpreter that can run a held-out Chaperone scorer.

    The scorer is deliberately a host-side controller process. Scientific
    blocks that it launches still use the generated Slurm worker entry point
    and its assigned SIF, but the scorer itself must never fall back into an
    outer container: doing so creates a second mount namespace and can change
    otherwise equivalent ``/projects`` and ``/mnt/projects`` paths between
    compilation and worker execution.
    """
    chaperone_repo = Path(os.path.abspath(chaperone_repo.expanduser()))
    chaperone_python = chaperone_repo / ".venv" / "bin" / "python"
    candidates = [chaperone_python, ROOT / ".venv" / "bin" / "python"]
    current_python = Path(sys.executable)
    if current_python not in candidates:
        candidates.append(current_python)
    environment = os.environ.copy()
    environment.update(_CONTROLLER_THREAD_ENV)
    environment.update(scoring_environment())
    environment["PYTHONPATH"] = _scoring_pythonpath(chaperone_repo)
    probe_code = "; ".join(
        (
            "import numpy",
            "numpy.__version__",
            "import scipy",
            "import polars",
            "import biotite",
            "import atomworks",
            "import submitit",
            "from blocks import Blocks",
            "from chaperone import ExecutionEngine",
            "import scoring",
            "from chaperone_cli.registry import register_flags",
            "register_flags()",
        )
    )
    detail = ""
    for python in candidates:
        if not python.is_file():
            continue
        try:
            probe = subprocess.run(
                [str(python), "-c", probe_code],
                capture_output=True,
                text=True,
                timeout=120,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = f" Probe failed for {python}: {exc}"
            continue
        if probe.returncode == 0:
            return python
        lines = (probe.stderr or probe.stdout).strip().splitlines()
        if lines:
            detail = f" Probe failed for {python}: {lines[-1]}"
    raise SystemExit(
        "Host Chaperone environment cannot run held-out scoring. "
        f"Repair it with: cd {chaperone_repo} && uv sync --frozen.{detail}"
    )


def _path_in_chaperone_namespace(path: Path, chaperone_repo: Path) -> Path:
    """Spell a checkout-owned path under the same alias as ``chaperone_repo``."""
    logical_repo = Path(os.path.abspath(chaperone_repo.expanduser()))
    try:
        relative = path.expanduser().resolve().relative_to(logical_repo.resolve())
    except ValueError:
        return Path(os.path.abspath(path.expanduser()))
    return logical_repo / relative


def _preflight_scoring_site_environment(values: Mapping[str, str]) -> None:
    """Fail before scoring if a required site tool or data path is unavailable."""
    problems: list[str] = []
    paths = {name: Path(value).expanduser() for name, value in values.items()}
    hbplus = paths.get("HBPLUS_COPY")
    if hbplus is not None and (not hbplus.is_file() or not os.access(hbplus, os.X_OK)):
        problems.append("HBPLUS_COPY must be an executable file")
    for name in ("CCD_MIRROR_PATH", "PYROSETTA_PATH", "X3DNA_BIN_DIR", "X3DNA_DIR"):
        if name in paths and (not paths[name].is_dir() or not os.access(paths[name], os.R_OK | os.X_OK)):
            problems.append(f"{name} must be a readable directory")
    for executable in (("x3dna-dssr", "x3dna-snap") if "X3DNA_BIN_DIR" in paths else ()):
        candidate = paths["X3DNA_BIN_DIR"] / executable
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            problems.append(f"X3DNA_BIN_DIR/{executable} must be executable")
    if problems:
        raise SystemExit("Scoring site-config preflight failed: " + "; ".join(problems))


def _scoring_site_environment(
    chaperone_repo: Path, environment: Mapping[str, str] | None = None,
    *, task_name: str | None = None,
) -> dict[str, str]:
    """Resolve the scorer's allowlisted settings from the selected profile.

    Shell values keep the settings loader's normal highest precedence.
    Otherwise only the approved path fields are read from the selected profile; user-local files and unrelated (potentially
    secret) settings are never propagated to the host scorer or its Slurm
    workers. The result is preflighted before it is returned.
    """
    source_environment = os.environ if environment is None else environment
    configured = get_site()["environment"]

    required = _SCORING_SITE_ENV_NAMES
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in _SCORING_SITE_ENV_NAMES:
        value = source_environment.get(name) or configured.get(name)
        if value:
            resolved[name] = value
        elif name in required:
            missing.append(name)
    if missing:
        raise SystemExit(
            "The site config does not define required scoring settings: "
            + ", ".join(missing)
        )
    _preflight_scoring_site_environment(resolved)
    return resolved


def _scoring_worker_environment(
    site_environment: Mapping[str, str], af3_jax_cache: Path
) -> dict[str, str]:
    """Everything the scorer's Slurm worker wrapper must re-export.

    The checkout's AF3 cache default is read-only inside the nested scientific
    container, so the scorer-specific cache replaces it. Both it and the site
    settings are exported after the worker sources its defaults.
    """
    return {
        **site_environment,
        _SCORING_AF3_JAX_CACHE_ENV: str(af3_jax_cache),
    }


def _scoring_packages_complete(target: Path) -> bool:
    marker = target / _SCORING_PACKAGE_MARKER
    try:
        manifest = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return manifest == {"packages": list(_SCORING_PACKAGES)}


def _prepare_scoring_packages(target: Path) -> None:
    """Stage the scorer-only locked PLIP dependencies outside the SIF.

    Many scorer watchers can start at once on one host. A per-user lock keeps
    their uv runs from exhausting the host's process allowance, and the
    packages are installed into a staging directory that replaces ``target``
    only once complete, so a failed or interrupted install never leaves a
    half-populated directory behind.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "uv is required to stage the held-out scorer's locked PLIP dependencies"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep uv's mutable cache private to the trial. Package publication stays
    # atomic and per-trial, while the host lock below serialises the installs.
    uv_cache = target.parent / "uv-cache"
    uv_cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(_CONTROLLER_THREAD_ENV)
    environment.update(scoring_environment())
    environment.update(_SCORING_UV_ENV)
    environment["UV_CACHE_DIR"] = str(uv_cache)
    lock_path = (
        Path(tempfile.gettempdir())
        / f"chaperone-evals-scoring-packages-{os.getuid()}.lock"
    )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, flags, 0o600)
    try:
        if os.fstat(lock_fd).st_uid != os.getuid():
            raise SystemExit(
                f"Scoring package lock is not owned by this user: {lock_path}"
            )
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if _scoring_packages_complete(target):
            return
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        command = [
            uv,
            "pip",
            "install",
            "--target",
            str(staging),
            "--python-version",
            "3.12",
            "--python-platform",
            # PEP 600 spelling of manylinux2014. uv dropped the legacy alias, so
            # "x86_64-manylinux2014" is now rejected outright and scoring dies
            # after the agent has already finished its run.
            "x86_64-manylinux_2_17",
            "--no-deps",
            "--link-mode",
            "copy",
            *_SCORING_PACKAGES,
        ]
        try:
            completed = subprocess.run(command, text=True, env=environment)
            if completed.returncode:
                raise SystemExit(
                    "Failed to stage held-out scoring packages: "
                    + " ".join(_SCORING_PACKAGES)
                )
            (staging / _SCORING_PACKAGE_MARKER).write_text(
                json.dumps({"packages": list(_SCORING_PACKAGES)}, sort_keys=True) + "\n"
            )
            previous = staging.with_name(staging.name + ".previous")
            if target.exists():
                os.replace(target, previous)
            try:
                os.replace(staging, target)
            except BaseException:
                if previous.exists() and not target.exists():
                    os.replace(previous, target)
                raise
            if previous.exists():
                shutil.rmtree(previous)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    finally:
        os.close(lock_fd)


def _run_score_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    trial_name: str,
    max_attempts: int = 3,
    report_path: Path | None = None,
) -> None:
    """Run the host scorer, reusing completed child chunks after a failure."""
    for attempt in range(1, max_attempts + 1):
        completed = subprocess.run(command, cwd=cwd, env=environment)
        if completed.returncode == 0:
            return
        if report_path is not None and report_path.is_file():
            try:
                report = _load_json(report_path)
            except SystemExit:
                report = {}
            if report.get("contract_ok") is False:
                # Some task scorers use a non-zero exit to emphasize a normal
                # contract failure after writing its complete report.
                return
        if attempt < max_attempts:
            print(
                f"Scoring {trial_name} exited {completed.returncode}; "
                f"restarting the host scorer ({attempt + 1}/{max_attempts}) "
                "to reuse completed chunks.",
                flush=True,
            )
            continue
        raise SystemExit(
            f"Scoring {trial_name} failed with exit code "
            f"{completed.returncode} after {max_attempts} attempts."
        )


def _score_trial(
    scorer: Path,
    trial: Path,
    chaperone_repo: Path,
    image: Path,
    required_n: int = 96,
    public_contract: Path | None = None,
) -> Path:
    """Score one Harbor trial and return its score report path.

    Split out of :func:`_score` so a rescue rerun can drive individual trials
    with its own skip/retry policy. This helper assumes ``trial/scoring`` does
    not exist yet; the caller owns that decision, because the right response to
    a pre-existing folder differs — an automatic post-run score must refuse and
    stop, while a rescore wants to skip a finished trial and clear the wreckage
    of a failed one.
    """
    chaperone_repo = Path(os.path.abspath(chaperone_repo.expanduser()))
    scorer = _path_in_chaperone_namespace(scorer, chaperone_repo)
    host_python = _host_scoring_python(chaperone_repo)
    site_environment = _scoring_site_environment(chaperone_repo, task_name=scorer.parent.parent.name)
    image = image.expanduser().resolve()
    if not image.is_file() or image.suffix != ".sif":
        raise SystemExit(f"Chaperone worker image was not found: {image}")
    designs, _legacy_run_meta = _submission(trial)
    task_name = scorer.parent.parent.name
    audit_path, run_meta, audit = write_trial_audit(
        trial, designs, task_name, required_n
    )
    print(f"Controller audit: {audit_path}", flush=True)
    for warning in audit["warnings"]:
        print(f"AUDIT WARNING: {warning}", flush=True)
    results = trial / "scoring"
    results.mkdir(parents=True)
    for directory in (results / ".home", results / ".tmp"):
        directory.mkdir()
    scoring_runtime = results / ".runtime"
    scoring_runtime.mkdir()
    python_packages = scoring_runtime / "python-packages"
    _prepare_scoring_packages(python_packages)
    slurm_python = scoring_runtime / "slurm-python"
    # Compiled blocks resolve their nested-container command while the scorer
    # constructs the block, before ExecutionEngine has a chance to publish its
    # rundir through Chaperone's process-global settings.  Export the exact same
    # absolute path passed via --rundir up front so the opt-in bind is valid at
    # block construction time as well as execution time.
    score_run = (results / "score_run").absolute()
    score_run.mkdir()
    # The checkout-provided AF3 cache lives below /projects, which is read-only
    # in the nested contained tool. Keep held-out compilation artifacts inside
    # the exact writable run bind instead of widening container access. The
    # cache must exist before the worker wrapper is built, because the wrapper
    # re-exports it after the worker defaults are applied.
    af3_jax_cache = score_run / ".cache" / "af3_jax"
    af3_jax_cache.mkdir(parents=True)
    ApptainerEnvironment._prepare_runtime_identity(scoring_runtime, results)
    ApptainerEnvironment._prepare_slurm_python(
        slurm_python,
        results,
        image,
        chaperone_repo=chaperone_repo,
        worker_readonly_paths=(scorer.parent.parent,),
        site_environment=_scoring_worker_environment(site_environment, af3_jax_cache),
    )
    command = [
        str(host_python),
        str(scorer),
        f"--designs={designs}",
        f"--out={results / 'score.json'}",
        f"--per_design_out={results / 'score_per_design.csv'}",
        f"--rundir={score_run}",
        f"--chaperone_repo={chaperone_repo}",
        f"--required_n={required_n}",
    ]
    command.append(f"--run_meta={run_meta}")
    environment = os.environ.copy()
    environment.update(_CONTROLLER_THREAD_ENV)
    environment.update(scoring_environment())
    environment.update(
        {
            "CHAPERONE_REPO": str(chaperone_repo),
            "CHAPERONE_COMPUTE_BACKEND": "slurm",
            "CHAPERONE_CONTAINER_BACKEND": "apptainer",
            # Nested scientific containers use --containall. Bind only this
            # scorer's active RUNDIR so their generated inputs and outputs are
            # visible without exposing the enclosing trial or /net/scratch.
            _SCORING_RUNDIR_BIND_ENV: "true",
            "RUNDIR": str(score_run),
            _SCORING_AF3_JAX_CACHE_ENV: str(af3_jax_cache),
            "CHAPERONE_SLURM_PYTHON": str(slurm_python),
            "CHAPERONE_PLIP_PYTHONPATH": str(python_packages),
            # The worker image's interpreter has its own prefix and
            # cannot discover tools installed into this external checkout by
            # inspecting sys.prefix. The checkout is already mounted read-only
            # into workers; publish its exact US-align executable explicitly.
            _SCORING_USALIGN_ENV: str(chaperone_repo / ".venv" / "bin" / "USalign"),
            "MPLCONFIGDIR": str(results / ".matplotlib"),
        }
    )
    # The frozen source bundle is intentionally sanitized and carries no .env
    # files. Publish only the preflighted settings resolved from the checked-in
    # site config; the Slurm worker wrapper re-exports these exact values.
    environment.update(site_environment)
    scoring_path = _scoring_pythonpath(chaperone_repo)
    environment["PYTHONPATH"] = scoring_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    print(f"Scoring Harbor trial: {trial.name}")
    print(f"Scoring command: {shlex.join(command)}", flush=True)
    _run_score_command(
        command,
        cwd=chaperone_repo,
        environment=environment,
        trial_name=trial.name,
        report_path=results / "score.json",
    )
    report = results / "score.json"
    print(f"Score report: {report}", flush=True)
    return report


def _score(
    task: Path,
    job_dir: Path,
    chaperone_repo: Path,
    image: Path,
    required_n: int = 96,
    public_contract: Path | None = None,
) -> None:
    chaperone_repo = Path(os.path.abspath(chaperone_repo.expanduser()))
    image = image.expanduser().resolve()
    try:
        successful = _successful_trials(job_dir)
    except SystemExit:
        _audit_finished_trials(task, job_dir, required_n)
        raise
    _audit_finished_trials(task, job_dir, required_n)
    scorer = task_scorer(task.name)
    for trial in successful:
        results = trial / "scoring"
        if results.exists():
            raise SystemExit(
                f"Refusing to overwrite existing scoring folder: {results}"
            )
        _score_trial(
            scorer,
            trial,
            chaperone_repo,
            image,
            required_n=required_n,
            public_contract=public_contract,
        )


def _audit_finished_trials(task: Path, job_dir: Path, required_n: int) -> None:
    """Publish controller metadata for every Harbor trial with a result."""

    if not job_dir.is_dir():
        return
    for trial in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        if not (trial / "result.json").is_file():
            continue
        designs = next(
            (
                candidate
                for candidate in (
                    trial / "artifacts" / "workspace" / "submission" / "designs.csv",
                    trial / "workspace" / "submission" / "designs.csv",
                )
                if candidate.is_file()
            ),
            trial / "artifacts" / "workspace" / "submission" / "designs.csv",
        )
        audit_path, _run_meta, audit = write_trial_audit(
            trial, designs, task.name, required_n
        )
        print(f"Controller audit: {audit_path}", flush=True)
        for warning in audit["warnings"]:
            print(f"AUDIT WARNING: {warning}", flush=True)


def _submission_environment(
    args: argparse.Namespace, task: Path, harbor_task: Path, run_name: str
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ARM": args.arm,
            "MODEL": args.model,
            "REASONING_EFFORT": args.reasoning_effort,
            "ATTEMPTS": str(args.attempts),
            "CONCURRENCY": str(args.concurrency),
            "JOB_NAME": run_name,
            # run_eval, the outer worker, Harbor, and the dependent scorer must
            # agree on one shared root. Without exporting this, run_eval puts
            # task definitions/logs in scratch while run.sh silently falls
            # back to the checkout's jobs/ directory for result workspaces.
            "HARBOR_JOBS_DIR": str(JOBS_ROOT),
            "HARBOR_TASK_DIR": str(harbor_task),
            # Serve the rendered wrapper copy, not the source prompt containing
            # {{REQUIRED_N}}.
            "PROMPT_FILE": str((harbor_task / "prompt.md").resolve()),
            "APPTAINER_BUILD_SOURCE": str(args.image.expanduser().resolve()),
            "HARBOR_AGENT_TIMEOUT_SEC": str(
                getattr(args, "agent_timeout_sec", DEFAULT_AGENT_TIMEOUT_SEC)
            ),
        }
    )
    if args.arm == "harness":
        # The skill is always the same validated bundle as WORKSPACE_SEED;
        # never forward an ambient second source path.
        environment.pop("HARNESS_SKILL", None)
        environment.update(
            {
                "TASK_INPUTS": str((harbor_task / "inputs").resolve()),
                "WORKSPACE_SEED": str(args.workspace_seed.expanduser().resolve()),
            }
        )
    else:
        # Do not let ambient launcher variables expose a harness seed or the
        # original task paths. The prompt and copied public inputs are carried
        # by the generated Harbor wrapper instead.
        environment.pop("TASK_INPUTS", None)
        environment.pop("WORKSPACE_SEED", None)
        environment.pop("HARNESS_SKILL", None)
        for name in tuple(environment):
            if name.startswith(
                (
                    "APPTAINERENV_",
                    "SINGULARITYENV_",
                    "APPTAINER_BIND",
                    "SINGULARITY_BIND",
                )
            ):
                environment.pop(name)
    return environment


def _find_codex_vendor() -> Path:
    """Directory bound into the agent image as /opt/harbor-codex."""
    from environments.codex_vendor import find_codex_vendor

    return find_codex_vendor()


def _probe_agent_runtime(image: Path, arm: str) -> None:
    """Verify only the entry points intentionally exposed to the selected arm."""
    site = get_site()
    on_compute = site.get("runtime_probe_on_compute", False)
    vendor = _find_codex_vendor()
    runtime_path = "/opt/harbor-codex/bin:/opt/harbor-codex/codex-path"
    if arm == "harness":
        runtime_path += ":" + str(Path(get_site()["worker_python"]).parent)
    runtime_path += ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    probe = "codex --version"
    if arm == "harness":
        probe += (
            " && test -x /bin/bash && test -x /bin/sh && apptainer --version"
            f" && {shlex.quote(get_site()['worker_python'])} -c "
            + shlex.quote(
                "import biotite, catcif_tools, ml_collections, openbabel, "
                "parasail, plip"
            )
        )
    # The hardened probe intentionally disables the host home and default /tmp
    # mounts. A read-only image then has nowhere for Codex aliases, Python
    # temporary files, or Matplotlib's import-time cache. Give the probe a
    # private, disposable /tmp rather than weakening isolation or depending on
    # an ambient host directory.
    probe_root = None
    if on_compute:
        probe_root = JOBS_ROOT / ".runtime-probes"
        probe_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="harbor-agent-runtime-", dir=probe_root) as temporary:
        command = [
            "apptainer",
            "exec",
            "--no-mount",
            "home,tmp,bind-paths,hostfs,cwd",
            "--cleanenv",
            "--no-eval",
            "--bind",
            f"{vendor}:/opt/harbor-codex:ro",
            "--bind",
            f"{temporary}:/tmp",
            str(image),
            "bash",
            "-lc",
            (
                "mkdir -p /tmp/home/.cache /tmp/home/.config/matplotlib"
                " && export HOME=/tmp/home TMPDIR=/tmp"
                " XDG_CACHE_HOME=/tmp/home/.cache"
                " XDG_CONFIG_HOME=/tmp/home/.config"
                " MPLCONFIGDIR=/tmp/home/.config/matplotlib"
                " OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1"
                " MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
                f" PATH={shlex.quote(runtime_path)}; {probe}"
            ),
        ]
        if on_compute:
            prefix = [
                "srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
                "--mem=4G", "--time=00:03:00", "--immediate=300",
                "--job-name=chaperone-eval-preflight",
            ]
            for option, key in (("partition", "cpu_partition"), ("account", "account"), ("qos", "qos")):
                if site[key]:
                    prefix.append(f"--{option}={site[key]}")
            command = prefix + command
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, timeout=510 if on_compute else 60
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit("Apptainer agent-runtime preflight timed out.") from exc
    if completed.returncode:
        detail = (completed.stdout + completed.stderr).strip()
        raise SystemExit(f"Apptainer agent-runtime preflight failed:\n{detail}")


def job_environment(environment: dict[str, str]) -> dict[str, str]:
    """Return the environment ``--export=ALL`` hands to the outer Slurm job.

    The job hosts the agent runtime but never scores, because scoring is launched
    from the submission host by a detached watcher. It therefore does not get the
    location of the held-out scorers.
    """
    return {
        name: value for name, value in environment.items() if name != PRIVATE_TASKS_ENV
    }


def _sbatch_command(args: argparse.Namespace, run_name: str) -> list[str]:
    logs = JOBS_ROOT / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    worker = ROOT / "scripts" / "run_slurm_node.sh"
    return [
        "sbatch",
        "--parsable",
        # Harbor/Codex state is not checkpointed for a Slurm batch restart.
        "--no-requeue",
        *([f"--partition={args.cpu_partition}"] if args.cpu_partition else []),
        *[
            f"--{key}={get_site()[key]}"
            for key in ("account", "qos")
            if get_site()[key]
        ],
        f"--time={args.time_limit}",
        f"--mem={args.memory_gb}G",
        "--cpus-per-task=4",
        f"--job-name={run_name}",
        f"--output={logs / '%x-%j.out'}",
        f"--error={logs / '%x-%j.err'}",
        "--export=ALL",
        f"--wrap={shlex.quote(str(worker))}",
    ]


def _external_scoring_command(
    args: argparse.Namespace,
    task: Path,
    run_name: str,
    harbor_job_id: str,
    job_dir: Path,
) -> list[str]:
    """Build the host watcher that scores after the Harbor job exits."""
    controller_python = ROOT / ".venv" / "bin" / "python"
    score_driver = ROOT / "scripts" / "score_after_eval.py"
    score_command = [
        str(controller_python),
        str(score_driver),
        "--task",
        task.name,
        "--run-dir",
        str(job_dir),
        "--chaperone-repo",
        str(Path(os.path.abspath(args.chaperone_repo.expanduser()))),
        "--worker-image",
        str(Path(os.path.abspath(args.worker_image.expanduser()))),
        "--required-n",
        str(args.required_n),
        "--harbor-job-id",
        harbor_job_id,
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    public_contract = _pin_public_contract(run_name, harbor_job_id)
    if public_contract is not None:
        score_command.extend(["--public-contract", str(public_contract)])
    return score_command


def _launch_external_scorer(
    command: list[str], run_name: str, harbor_job_id: str
) -> tuple[subprocess.Popen[bytes], Path]:
    """Launch one detached post-run scorer on the submission host.

    Only scientific workers submitted by the scorer enter Slurm.  Keeping this
    watcher on the submission host preserves the logical ``/projects`` path and
    prevents a compute-node mount alias from leaking into serialized blocks.
    """
    logs = JOBS_ROOT / "host-scorers"
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"{run_name}-{harbor_job_id}"
    stdout_path = logs / f"{stem}.out"
    stderr_path = logs / f"{stem}.err"
    receipt_path = logs / f"{stem}.json"
    with (
        stdout_path.open("ab", buffering=0) as stdout,
        stderr_path.open("ab", buffering=0) as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "external-host-watcher",
                "pid": process.pid,
                "harbor_job_id": harbor_job_id,
                "command": command,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
            indent=2,
        )
        + "\n"
    )
    return process, receipt_path


def _pin_public_contract(run_name: str, harbor_job_id: str) -> Path | None:
    """Copy public scoring inputs to a job-ID-keyed immutable controller path."""
    source = JOBS_ROOT / "task-definitions" / run_name / "inputs"
    contract = source / "acceptance_contract.json"
    scorer = source / "public_score.py"
    if not contract.is_file() and not scorer.is_file():
        return None
    if not contract.is_file() or not scorer.is_file():
        raise SystemExit("Pinned public contract and scorer must be supplied together")
    destination = JOBS_ROOT / "scoring-contracts" / harbor_job_id
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in (contract, scorer):
        target = destination / source_file.name
        payload = source_file.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise SystemExit(
                f"Scoring contract for Harbor job {harbor_job_id} is already pinned "
                "with different content"
            )
        if not target.exists():
            target.write_bytes(payload)
    return destination / contract.name


def _preflight(args: argparse.Namespace, task: Path) -> None:
    if not args.no_score and os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Automatic external scoring must be launched from the submission "
            "host, not from inside a Slurm allocation. Run run_eval.py on the "
            "login host, or pass --no-score and score separately."
        )
    required_files = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / "run.sh",
        ROOT / "scripts" / "run_slurm_node.sh",
        ROOT / "scripts" / "harbor_cli.py",
        ROOT / "scripts" / "slurm_gpu_guard.py",
        ROOT / "scripts" / "score_after_eval.py",
        ROOT / "environments" / "slurm_broker.py",
        task / "prompt.md",
    ]
    if not args.no_score:
        required_files.append(task_scorer(task.name))
        check_inputs_match(task.name)
    # Disabling scoring must not grant the agent access to private material.
    # A genuinely public-only checkout can launch without a private tree.
    candidate_private = os.environ.get(PRIVATE_TASKS_ENV)
    if candidate_private or any(PUBLIC_TASKS_ROOT.glob("*/score/score.py")):
        assert_private_tree_unreachable(
            private_tasks_root(), [source for source, _ in _runtime_bind_pairs()]
        )
    if args.arm == "harness":
        required_files.append(ENVIRONMENT_SETUP)
    else:
        required_files.append(ROOT / "environments" / "baseline.py")
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit("Required files are missing: " + ", ".join(missing))
    if args.arm == "harness":
        required_capabilities = _required_harness_capabilities(
            task,
            args.workspace_seed,
        )
        capability_manifest = _runtime_capability_manifest(args.workspace_seed)
        availability = {
            item["name"]: item["available"]
            for item in capability_manifest["capabilities"]
        }
        missing_capabilities = [
            label for label in sorted(required_capabilities) if not availability[label]
        ]
        if missing_capabilities:
            raise SystemExit(
                "Harness runtime capabilities are unavailable: "
                + ", ".join(missing_capabilities)
            )
        try:
            validate_snapshot(args.workspace_seed, args.workspace_seed.name)
        except RuntimeError as error:
            raise SystemExit(f"Unsafe Chaperone harness bundle: {error}") from error
    if not args.image.expanduser().resolve().is_file():
        raise SystemExit(f"Apptainer image not found: {args.image}")
    if not args.no_score and not args.worker_image.expanduser().resolve().is_file():
        raise SystemExit(f"Chaperone worker image not found: {args.worker_image}")
    commands = ["sbatch", "squeue", "sacct", "apptainer"]
    if not args.no_score:
        commands.append("uv")
        _host_scoring_python(args.chaperone_repo)
        _scoring_site_environment(args.chaperone_repo, task_name=task.name)
    for command in commands:
        if shutil.which(command) is None:
            raise SystemExit(f"Required command is unavailable: {command}")
    if args.arm == "harness":
        for label in sorted(
            required_capabilities & set(_HARNESS_CONTAINER_CAPABILITIES)
        ):
            image = next(
                path for path in _runtime_capabilities()[label] if path.is_file()
            )
            inspected = subprocess.run(
                ["apptainer", "inspect", "--json", str(image)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if inspected.returncode:
                detail = (inspected.stderr or inspected.stdout).strip()
                raise SystemExit(f"Invalid {label} {image}: {detail}")
    if (
        not os.environ.get("OPENAI_API_KEY")
        and not os.environ.get("CODEX_AUTH_JSON_PATH")
        and not (Path.home() / ".codex" / "auth.json").is_file()
    ):
        raise SystemExit(
            "Codex authentication is unavailable: set OPENAI_API_KEY or run "
            "'codex login'."
        )
    _probe_agent_runtime(args.image.expanduser().resolve(), args.arm)


def main() -> None:
    global JOBS_ROOT

    parser = argparse.ArgumentParser(
        description=(
            "Launch evals_task/<task-name> through Harbor, monitor completion, "
            "and run its held-out scorer."
        )
    )
    parser.add_argument("task_name", help="Exact folder name beneath evals_task")
    parser.add_argument(
        "--arm",
        "--arms",
        dest="arm",
        choices=("baseline", "harness"),
        default="harness",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING)
    parser.add_argument("--attempts", type=_positive_integer, default=1)
    parser.add_argument("--concurrency", type=_positive_integer, default=1)
    parser.add_argument(
        "--exclude-nodes",
        default="",
        help=(
            "Slurm node list excluded from every GPU job in the agent trial "
            "and its detached held-out scorer."
        ),
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Shared run root for results, logs, task wrappers, and snapshots. "
            "Overrides HARBOR_JOBS_DIR; individual results are written beneath "
            "RUN_DIR/results/<run-name>-<slurm-job-id>."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--cpu-partition", default=None)
    parser.add_argument("--time-limit", default="24:00:00")
    parser.add_argument(
        "--scoring-time-limit",
        default="24:00:00",
        help=(
            "Deprecated compatibility option. External host scoring is no "
            "longer wrapped in a Slurm controller allocation."
        ),
    )
    parser.add_argument(
        "--agent-timeout-sec",
        type=_positive_integer,
        default=DEFAULT_AGENT_TIMEOUT_SEC,
        help=(
            "Harbor/Codex agent timeout in seconds; injected into both the "
            "generated task metadata and the arm instruction."
        ),
    )
    parser.add_argument(
        "--required-n",
        type=_positive_integer,
        default=96,
        help=(
            "Submission size. Injected into the arm instruction in place of "
            "{{REQUIRED_N}} and passed to the held-out scorer, so the prompt "
            "and the contract check always agree. Lower it for smoke tests."
        ),
    )
    parser.add_argument(
        "--memory-gb",
        type=_positive_integer,
        default=DEFAULT_CONTROLLER_MEMORY_GB,
        help=(
            "Memory for the controller allocation, which also hosts the agent "
            "runtime. Sets both the Slurm --mem and the container's memory_mb. "
            f"Defaults to {DEFAULT_CONTROLLER_MEMORY_GB} GB because clean "
            "baseline toolchains have exceeded 32 GB during finalization."
        ),
    )
    parser.add_argument(
        "--chaperone-path",
        type=Path,
        help=(
            "Required for the harness arm: explicit Chaperone working-tree path. "
            "Only allowlisted raw source and agent-skill files are copied; Git "
            "metadata, history, branches, generated docs, examples, tests, "
            "campaigns, and evaluation artifacts are excluded."
        ),
    )
    parser.add_argument(
        "--chaperone-repo",
        type=Path,
        default=DEFAULT_CHAPERONE_REPO,
        help=(
            "Host-only Chaperone checkout used by the held-out scoring controller; "
            "it is never copied or mounted into the agent. This is intentionally "
            "separate from --chaperone-path."
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        help=(
            "agent image override; defaults to the minimal baseline image for "
            "baseline and the Chaperone image for harness"
        ),
    )
    parser.add_argument(
        "--worker-image",
        "--scoring-image",
        dest="worker_image",
        type=Path,
        default=None,
        help=(
            "Chaperone image used by scientific Slurm workers launched by the "
            "host-side scorer; --scoring-image is a compatibility alias"
        ),
    )
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument(
        "--skip-image-digests",
        action="store_true",
        help=(
            "Record image identity (path, size, mtime) in the launch manifest "
            "without hashing the images. Hashing is cached, so this only avoids the "
            "one-time cost of the first launch."
        ),
    )
    parser.add_argument(
        "--private-tasks-path",
        type=Path,
        help=(
            "evals_task/ directory that holds the scorers and audit patterns; "
            f"defaults to ${PRIVATE_TASKS_ENV}, then the one next to this controller"
        ),
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for Harbor and its detached external scorer before returning.",
    )
    parser.add_argument("--dry-run", action="store_true")
    add_site_arguments(parser)
    args = parser.parse_args()
    try:
        configure_site(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.private_tasks_path is not None:
        # Exported so the detached scoring watcher and Slurm jobs inherit it.
        os.environ[PRIVATE_TASKS_ENV] = str(args.private_tasks_path.expanduser())
    if args.exclude_nodes and not re.fullmatch(
        r"[A-Za-z0-9_,.\[\]-]+", args.exclude_nodes
    ):
        parser.error("--exclude-nodes contains unsupported characters")
    if args.exclude_nodes:
        os.environ["HARBOR_EXCLUDED_NODES"] = args.exclude_nodes
    else:
        os.environ.pop("HARBOR_EXCLUDED_NODES", None)
    if args.arm == "harness" and args.chaperone_path is None:
        parser.error("the harness arm requires an explicit --chaperone-path")
    if args.arm == "baseline" and args.chaperone_path is not None:
        parser.error("--chaperone-path is not accepted by the baseline arm")
    JOBS_ROOT = _jobs_root(args.run_dir)
    _prepare_shared_job_directories()
    if args.arm == "harness":
        try:
            args.workspace_seed = build_harness_bundle(
                args.chaperone_path,
                JOBS_ROOT / "workspace-seeds",
            )
        except RuntimeError as error:
            parser.error(str(error))
    else:
        # Keep the internal attribute uniform without accepting or consulting a
        # harness source for the clean-room baseline arm.
        args.workspace_seed = None

    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than zero")
    task = resolve_task(args.task_name)
    default_name = f"{task.name}-{args.arm}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_name = _validate_name(args.run_name or default_name, "Run name")
    _preflight(args, task)
    harbor_task = materialize_harbor_task(
        task,
        run_name,
        args.arm,
        args.image,
        memory_gb=args.memory_gb,
        required_n=args.required_n,
        agent_timeout_sec=args.agent_timeout_sec,
        workspace_seed=args.workspace_seed,
    )
    environment = _submission_environment(args, task, harbor_task, run_name)
    command = _sbatch_command(args, run_name)

    print(f"Task: {task}")
    if args.arm == "harness":
        print(f"Chaperone source: {args.chaperone_path.expanduser().resolve()}")
        print(f"Sanitized source bundle: {args.workspace_seed}")
    print(f"Harbor task wrapper: {harbor_task}")
    print(f"Run name: {run_name}")
    if args.exclude_nodes:
        print(f"Excluded GPU nodes: {args.exclude_nodes}")
    print(f"Submit command: {shlex.join(command)}", flush=True)
    if args.dry_run:
        if not args.no_score:
            print(
                "A detached external host watcher will score the frozen "
                "submission after the Harbor job receives an ID."
            )
        return

    manifest_path = None
    try:
        manifest_path = write_manifest(
            build_manifest(args, task, harbor_task, run_name, JOBS_ROOT), JOBS_ROOT
        )
        print(f"Launch manifest: {manifest_path}", flush=True)
    except SystemExit:
        raise
    except Exception as error:  # a bookkeeping failure must not block a valid launch
        print(f"WARNING: could not write the launch manifest: {error}", flush=True)

    submitted = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=job_environment(environment),
    )
    if submitted.stdout:
        print(submitted.stdout, end="")
    if submitted.stderr:
        print(submitted.stderr, end="", file=sys.stderr)
    if submitted.returncode:
        raise SystemExit(f"Submission failed with exit code {submitted.returncode}.")

    job_id = _parse_job_id(submitted.stdout)
    job_dir = JOBS_ROOT / "results" / f"{run_name}-{job_id}"
    print(f"Submitted Slurm job {job_id}; Harbor results: {job_dir}", flush=True)
    if manifest_path is not None:
        mark_submitted(manifest_path, job_id, job_dir)
    score_process: subprocess.Popen[bytes] | None = None
    score_receipt: Path | None = None
    if not args.no_score:
        score_command = _external_scoring_command(
            args,
            task,
            run_name,
            job_id,
            job_dir,
        )
        try:
            score_process, score_receipt = _launch_external_scorer(
                score_command, run_name, job_id
            )
        except OSError as error:
            raise SystemExit(
                "Harbor is still running, but automatic scoring could not be "
                f"launched on the submission host: {error}. Run directory: {job_dir}"
            ) from error
        print(
            f"Started external scoring watcher PID {score_process.pid}; it will "
            f"score after Harbor job {job_id} exits. Receipt: {score_receipt}",
            flush=True,
        )
    if args.wait:
        try:
            if score_process is None:
                _wait_for_job(job_id, args.poll_seconds)
            else:
                score_status = score_process.wait()
                if score_status:
                    raise SystemExit(
                        f"External scorer exited {score_status}; see {score_receipt}."
                    )
        except KeyboardInterrupt:
            raise SystemExit(
                "Stopped waiting; the Harbor job and external scorer were not "
                "cancelled."
            ) from None


if __name__ == "__main__":
    main()
