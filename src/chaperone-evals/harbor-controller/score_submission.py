#!/usr/bin/env python3
"""Score a cleanly completed Harbor submission for any eval task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from pathlib import Path

from environments.apptainer import ApptainerEnvironment
from run_eval import (
    _CONTROLLER_THREAD_ENV,
    _SCORING_AF3_JAX_CACHE_ENV,
    _SCORING_RUNDIR_BIND_ENV,
    _SCORING_USALIGN_ENV,
    _host_scoring_python,
    _path_in_chaperone_namespace,
    _prepare_scoring_packages,
    _run_score_command,
    _scoring_pythonpath,
    _scoring_site_environment,
    _scoring_worker_environment,
)
from scripts.trial_audit import write_trial_audit
from task_paths import default_chaperone_repo, private_tasks_root, task_scorer


# Resolve host paths before compiling worker jobs so the controller and its
# generated Slurm entry point use the same shared-filesystem spelling.
ROOT = Path(os.path.abspath(__file__)).parent
# The controller lives inside the Chaperone checkout it evaluates; set
# CHAPERONE_REPO to score against a different one.
DEFAULT_CHAPERONE_REPO = default_chaperone_repo()


def _load_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return data


def _has_delivered_submission(trial_dir: Path) -> bool:
    return any(
        (candidate / "designs.csv").is_file()
        for candidate in (
            trial_dir / "artifacts" / "workspace" / "submission",
            trial_dir / "workspace" / "submission",
        )
    )


def _trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in job_dir.iterdir()
        if path.is_dir()
        and (path / "config.json").is_file()
        and ((path / "result.json").is_file() or _has_delivered_submission(path))
    )


def _select_trial(job_dir: Path, trial_name: str | None) -> Path:
    candidates = _trial_dirs(job_dir)
    if trial_name:
        selected = job_dir / trial_name
        if selected not in candidates:
            names = ", ".join(path.name for path in candidates) or "none"
            raise SystemExit(
                f"Trial {trial_name!r} was not found in {job_dir}; available: {names}"
            )
        candidates = [selected]

    successful: list[Path] = []
    delivered: list[Path] = []
    failures: list[str] = []
    for candidate in candidates:
        result_path = candidate / "result.json"
        result = _load_json(result_path) if result_path.is_file() else {}
        exception = result.get("exception_info")
        agent_execution = result.get("agent_execution")
        if exception is None and agent_execution is not None:
            successful.append(candidate)
        elif _has_delivered_submission(candidate):
            # The agent deliverable is immutable once designs.csv is present.
            # A controller interruption may prevent Harbor from finalizing
            # result.json, but it must not force an expensive agent rerun just
            # to retry or repair held-out scoring.
            delivered.append(candidate)
        else:
            exception_type = (
                exception.get("exception_type", "unknown")
                if isinstance(exception, dict)
                else "agent result was not finalized"
            )
            failures.append(f"{candidate.name} ({exception_type})")

    selectable = successful or delivered
    if not selectable:
        detail = ", ".join(failures) or "no trial result directories"
        raise SystemExit(
            "No completed agent trial or delivered submission is available to "
            "score: " + detail
        )
    if len(selectable) > 1 and trial_name is None:
        names = ", ".join(path.name for path in selectable)
        raise SystemExit(
            "Multiple scoreable trials are available; select one with --trial: " + names
        )
    return selectable[0]


def _submission_dir(trial_dir: Path) -> Path:
    candidates = (
        trial_dir / "artifacts" / "workspace" / "submission",
        trial_dir / "workspace" / "submission",
    )
    for candidate in candidates:
        if (candidate / "designs.csv").is_file():
            return candidate
    raise SystemExit(
        "The successful trial did not produce submission/designs.csv beneath "
        f"{trial_dir}"
    )


def _readable_designs(path: Path) -> None:
    """Confirm the CSV is non-empty and leave its task contract to the scorer."""
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            first = next(reader, None)
    except (OSError, csv.Error, UnicodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not header:
        raise SystemExit(f"Submission CSV has no header: {path}")
    if first is None:
        raise SystemExit(f"Submission CSV has no rows: {path}")


def _resolve_task(name: str) -> Path:
    """Return the private task directory that holds ``name``'s held-out scorer."""
    return task_scorer(name).parent.parent


def _task_from_job(job_dir: Path) -> str | None:
    """Infer the task folder from the Harbor task name stored in trial config."""
    for trial in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        config = trial / "config.json"
        if not config.is_file():
            continue
        name = str(_load_json(config).get("task_name") or "")
        candidate = name.rsplit("/", 1)[-1]
        if (
            candidate
            and (private_tasks_root() / candidate / "score" / "score.py").is_file()
        ):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a task's held-out scorer on a completed Harbor trial or a "
            "preserved delivered submission"
        )
    )
    parser.add_argument("--job", type=Path, required=True, help="Harbor job directory")
    parser.add_argument(
        "--task",
        help="Task folder beneath the private evals_task; inferred from trial config when omitted",
    )
    parser.add_argument(
        "--required-n",
        type=int,
        help="Submission size to enforce; match the instruction served to the agent",
    )
    parser.add_argument(
        "--public-contract",
        type=Path,
        help=(
            "Pinned public acceptance contract served to the trial. Defaults to "
            "the task contract when one exists."
        ),
    )
    parser.add_argument("--trial", help="Trial directory name when a job has several")
    parser.add_argument(
        "--chaperone-repo",
        type=Path,
        default=DEFAULT_CHAPERONE_REPO,
        help="Chaperone checkout with its scoring Python environment",
    )
    parser.add_argument(
        "--worker-image",
        "--scoring-image",
        dest="worker_image",
        type=Path,
        default=None,
        help=(
            "Chaperone image used by scientific Slurm workers; the scorer "
            "itself always runs in the host .venv"
        ),
    )
    parser.add_argument("--out", type=Path, help="Output score JSON path")
    parser.add_argument("--rundir", type=Path, help="Scoring pipeline run directory")
    parser.add_argument(
        "--worker-readonly-path",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional existing path to mount read-only in scientific Slurm "
            "workers. Repeat for retained oracle caches whose absolute structure "
            "paths live outside --rundir."
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Run the held-out fold on this host instead of submitting to Slurm. "
            "Only valid where a GPU is present."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print without scoring"
    )
    from site_config import add_site_arguments, configure_site

    add_site_arguments(parser)
    args = parser.parse_args()
    try:
        configure_site(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    job_dir = args.job.expanduser().resolve()
    if not job_dir.is_dir():
        parser.error(f"job directory does not exist: {job_dir}")

    task_name = args.task or _task_from_job(job_dir)
    if not task_name:
        parser.error("could not infer the task from the Harbor job; pass --task")
    task = _resolve_task(task_name)
    task_scorer = task / "score" / "score.py"
    if args.required_n is not None and args.required_n < 1:
        parser.error("--required-n must be a positive integer")

    trial_dir = _select_trial(job_dir, args.trial)
    submission = _submission_dir(trial_dir)
    designs = submission / "designs.csv"
    _readable_designs(designs)

    public_contract = args.public_contract
    if public_contract is None:
        candidate = task / "inputs" / "acceptance_contract.json"
        public_contract = candidate if candidate.is_file() else None
    contract: dict[str, object] = {}
    if public_contract is not None:
        public_contract = public_contract.expanduser().resolve()
        contract = _load_json(public_contract)
    required_n = args.required_n
    if required_n is None:
        pinned_required_n = contract.get("required_n")
        if isinstance(pinned_required_n, int) and not isinstance(
            pinned_required_n, bool
        ):
            required_n = pinned_required_n
    if required_n is None:
        required_n = 96
    _audit_path, run_meta, _audit = write_trial_audit(
        trial_dir, designs, task.name, required_n
    )

    chaperone_repo = Path(os.path.abspath(args.chaperone_repo.expanduser()))

    default_scoring_dir = trial_dir / "scoring"
    out_path = (args.out or default_scoring_dir / "score.json").expanduser().resolve()
    scoring_dir = out_path.parent
    scoring_dir.mkdir(parents=True, exist_ok=True)
    scoring_runtime = scoring_dir / ".runtime"
    scoring_runtime.mkdir(exist_ok=True)
    python_packages = scoring_runtime / "python-packages"
    slurm_python = scoring_runtime / "slurm-python"
    worker_image = Path(os.path.abspath(args.worker_image.expanduser()))
    if not worker_image.is_file() or worker_image.suffix != ".sif":
        raise SystemExit(f"Chaperone worker image was not found: {worker_image}")
    worker_readonly_paths = tuple(
        Path(os.path.abspath(path.expanduser())) for path in args.worker_readonly_path
    )
    missing_worker_paths = [path for path in worker_readonly_paths if not path.exists()]
    if missing_worker_paths:
        missing = ", ".join(str(path) for path in missing_worker_paths)
        raise SystemExit(f"Worker read-only path does not exist: {missing}")
    host_python = _host_scoring_python(chaperone_repo)
    site_environment = _scoring_site_environment(chaperone_repo, task_name=task.name)
    _prepare_scoring_packages(python_packages)
    # The run directory and AF3 cache must exist before the worker wrapper is
    # built, because the wrapper re-exports the cache after worker defaults are applied.
    rundir = (args.rundir or scoring_dir / "score_run").expanduser().resolve()
    rundir.mkdir(parents=True, exist_ok=True)
    af3_jax_cache = rundir / ".cache" / "af3_jax"
    af3_jax_cache.mkdir(parents=True, exist_ok=True)
    ApptainerEnvironment._prepare_runtime_identity(scoring_runtime, scoring_dir)
    ApptainerEnvironment._prepare_slurm_python(
        slurm_python,
        scoring_dir,
        worker_image,
        chaperone_repo=chaperone_repo,
        worker_readonly_paths=(task, *worker_readonly_paths),
        site_environment=_scoring_worker_environment(site_environment, af3_jax_cache),
    )

    per_design_out = out_path.parent / "score_per_design.csv"
    scorer = _path_in_chaperone_namespace(task_scorer, chaperone_repo)
    command = [
        str(host_python),
        str(scorer),
        f"--designs={designs}",
        f"--out={out_path}",
        f"--per_design_out={per_design_out}",
        f"--rundir={rundir}",
        f"--chaperone_repo={chaperone_repo}",
    ]
    command.append(f"--run_meta={run_meta}")
    command.append(f"--required_n={required_n}")
    # Local execution was hardcoded here, so the holo fold asked JAX for a GPU
    # backend on whatever host ran the rescue — on a login node that is
    # "Unknown backend: 'gpu' requested ... Platforms present are: cpu", and
    # scoring dies having produced no outputs. Submit to Slurm by default, the
    # same way run_eval scores, and keep --local as an opt-in for GPU hosts.
    if args.local:
        command += ["--local", "--use_local_gpus"]

    print(f"Scoring trial: {trial_dir.name} (task: {task.name})")
    print(f"Submission: {submission}")
    print(f"Command: {shlex.join(command)}")
    if args.dry_run:
        return

    from site_config import scoring_environment

    environment = {**os.environ, **scoring_environment()}
    environment.update(
        {
            "CHAPERONE_REPO": str(chaperone_repo),
            "CHAPERONE_COMPUTE_BACKEND": "local" if args.local else "slurm",
            "CHAPERONE_CONTAINER_BACKEND": "apptainer",
            _SCORING_RUNDIR_BIND_ENV: "true",
            # Nested scientific containers use --containall. Export the exact
            # active scoring run before the task module constructs any blocks,
            # matching run_eval._score_trial and exposing only this run rather
            # than the enclosing trial or all of /net/scratch.
            "RUNDIR": str(rundir),
            _SCORING_AF3_JAX_CACHE_ENV: str(af3_jax_cache),
            "CHAPERONE_SLURM_PYTHON": str(slurm_python),
            "CHAPERONE_PLIP_PYTHONPATH": str(python_packages),
            _SCORING_USALIGN_ENV: str(chaperone_repo / ".venv" / "bin" / "USalign"),
        }
    )
    environment.update(_CONTROLLER_THREAD_ENV)
    environment.update(site_environment)
    source_path = _scoring_pythonpath(chaperone_repo)
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    _run_score_command(
        command,
        cwd=chaperone_repo,
        environment=environment,
        trial_name=trial_dir.name,
        report_path=out_path,
    )


if __name__ == "__main__":
    main()
