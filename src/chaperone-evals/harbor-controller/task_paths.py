"""Resolve public prompts/inputs and explicitly configured private evaluation data.

Scorers, reference answers and audit fingerprints live outside the release
checkout. Only the trusted controller receives CHAPERONE_EVALS_PRIVATE.
Removing files from a branch does not remove earlier copies from Git history.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

CONTROLLER_ROOT = Path(os.path.abspath(__file__)).parent
EVALS_ROOT = CONTROLLER_ROOT.parent  # src/chaperone-evals
PUBLIC_TASKS_ROOT = EVALS_ROOT / "evals_task"
PRIVATE_TASKS_ENV = "CHAPERONE_EVALS_PRIVATE"


def default_chaperone_repo() -> Path:
    """Return the Chaperone checkout used for held-out scoring.

    The controller is part of that checkout (``<repo>/src/chaperone-evals``), so
    the default is the repository containing it. ``CHAPERONE_REPO`` overrides it.
    """
    configured = os.environ.get("CHAPERONE_REPO")
    return Path(configured) if configured else EVALS_ROOT.parent.parent


def private_tasks_root(override: str | os.PathLike | None = None) -> Path:
    """Return the ``evals_task/`` directory that holds the scorers.

    An explicit path or ``CHAPERONE_EVALS_PRIVATE`` is required.
    """
    configured = override or os.environ.get(PRIVATE_TASKS_ENV)
    if not configured:
        raise SystemExit(
            f"Held-out evaluation data is stored separately. Set {PRIVATE_TASKS_ENV} "
            "to a private evals_task/ directory (or pass --private-tasks-path)."
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Private tasks directory not found: {root}")
    return root


def task_scorer(task_name: str, private_root: str | os.PathLike | None = None) -> Path:
    """Return the held-out ``score.py`` for ``task_name``."""
    root = private_tasks_root(private_root)
    scorer = root / task_name / "score" / "score.py"
    # A task name is one folder name; never let it walk out of the tasks root.
    plain_name = Path(task_name).name == task_name and task_name not in {"", ".", ".."}
    if not plain_name or not scorer.is_file():
        available = ", ".join(
            sorted(
                path.name
                for path in root.iterdir()
                if (path / "score" / "score.py").is_file()
            )
        )
        raise SystemExit(
            f"No scorer for task {task_name!r} under {root}. "
            f"Available: {available or 'none'}"
        )
    return scorer


def _digests(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def check_inputs_match(
    task_name: str, private_root: str | os.PathLike | None = None
) -> None:
    """Fail if the private and public ``inputs/`` differ for ``task_name``.

    Agents are given the public copy while the scorer reads the private one, so
    any drift would score designs against a target the agent never saw.
    """
    public = PUBLIC_TASKS_ROOT / task_name / "inputs"
    private = private_tasks_root(private_root) / task_name / "inputs"
    if not private.is_dir():
        raise SystemExit(f"Private task has no inputs/ directory: {private}")
    public_files, private_files = _digests(public), _digests(private)
    if public_files != private_files:
        changed = sorted(
            name
            for name in public_files.keys() | private_files.keys()
            if public_files.get(name) != private_files.get(name)
        )
        raise SystemExit(
            f"Public and private inputs/ differ for {task_name!r}: "
            + ", ".join(changed)
        )


def assert_private_tree_unreachable(
    private_root: str | os.PathLike, bound_sources: list[str | os.PathLike]
) -> None:
    """Fail if the agent container can see the private tree through a bind mount.

    The agent may read anything under a bound host path. Symlinks are resolved on
    both sides, because a cluster path such as ``/net`` can alias ``/mnt/net``.
    """
    private = Path(os.path.realpath(private_root))
    for source in bound_sources:
        for candidate in {Path(source), Path(os.path.realpath(source))}:
            if private == candidate or private.is_relative_to(candidate) or candidate.is_relative_to(private):
                raise SystemExit(
                    f"The private tasks directory {private} is inside {candidate}, "
                    "which is mounted into the agent container. Move the private "
                    "tree somewhere the container cannot see."
                )
