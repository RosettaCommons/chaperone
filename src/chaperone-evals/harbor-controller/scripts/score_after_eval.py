#!/usr/bin/env -S bash -c 'exec "$(dirname "$0")/../.venv/bin/python" "$0" "$@"'
"""Wait outside Harbor, then score its frozen submission on the host."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(os.path.abspath(__file__)).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_eval import (  # noqa: E402
    DEFAULT_CHAPERONE_REPO,
    _score,
    _wait_for_job,
    resolve_task,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a Harbor run after its outer Slurm job exits."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--chaperone-repo", type=Path, default=DEFAULT_CHAPERONE_REPO)
    parser.add_argument("--worker-image", type=Path, default=None)
    parser.add_argument("--required-n", type=_positive_integer, default=96)
    parser.add_argument("--public-contract", type=Path)
    parser.add_argument(
        "--harbor-job-id",
        help="Outer Harbor Slurm job to await before inspecting its result.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    from site_config import add_site_arguments, configure_site

    add_site_arguments(parser)
    args = parser.parse_args()
    try:
        configure_site(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than zero")

    task = resolve_task(args.task)
    run_dir = Path(os.path.abspath(args.run_dir.expanduser()))
    chaperone_repo = Path(os.path.abspath(args.chaperone_repo.expanduser()))
    worker_image = Path(os.path.abspath(args.worker_image.expanduser()))
    if args.harbor_job_id:
        print(
            f"External scorer watcher awaiting Harbor job {args.harbor_job_id}",
            flush=True,
        )
        try:
            _wait_for_job(args.harbor_job_id, args.poll_seconds)
        except SystemExit as exc:
            # The outer allocation can fail during teardown after the agent has
            # already delivered. _score audits every trial either way and still
            # refuses a run with no delivered submission.
            print(
                f"{exc} Checking for an immutable delivered submission before "
                "stopping.",
                flush=True,
            )
    print(f"External scoring started for: {run_dir}", flush=True)
    _score(
        task,
        run_dir,
        chaperone_repo,
        worker_image,
        required_n=args.required_n,
        public_contract=args.public_contract,
    )


if __name__ == "__main__":
    main()
