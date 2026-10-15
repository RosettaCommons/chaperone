"""Export private evaluation assets as Harbor packages; restore controller data.

Only run this on the trusted controller. Dataset-level files are never copied
into an agent's task environment. The external Chaperone scorer remains the
scientific oracle; Harbor's standalone verifier fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_dataset(private: Path, destination: Path, name: str) -> Path:
    from harbor.models.dataset.manifest import DatasetManifest
    from harbor.models.task.config import TaskConfig
    from harbor.publisher.packager import Packager
    import toml

    DatasetManifest.model_validate({"dataset": {"name": name}})
    private = private.resolve()
    if destination.exists() or destination.resolve().is_relative_to(private):
        raise ValueError("Export destination must be new and outside the private tree")
    files = sorted(private.rglob("*"))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in files):
        raise ValueError("Private datasets may contain only regular files/directories")
    if not (private / "audit_hints.json").is_file():
        raise ValueError("Private dataset requires audit_hints.json")
    tasks = sorted(p for p in private.iterdir() if p.is_dir())
    if not tasks:
        raise ValueError("Private dataset contains no tasks")
    for task in tasks:
        for required in ("prompt.md", "inputs", "score/score.py", "example"):
            if not (task / required).exists():
                raise ValueError(f"Incomplete private task: {task.name}/{required}")
    destination.mkdir(parents=True, mode=0o700)
    hashes = {
        p.relative_to(private).as_posix(): digest(p) for p in files if p.is_file()
    }
    (destination / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with tarfile.open(destination / "private-evals.tar.gz", "w:gz") as archive:
        for p in files:
            if p.is_file():
                archive.add(
                    p, arcname=p.relative_to(private).as_posix(), recursive=False
                )
    refs = []
    org = name.split("/")[0]
    for task in tasks:
        output = destination / task.name
        (output / "environment").mkdir(parents=True)
        (output / "tests").mkdir()
        shutil.copytree(task / "inputs", output / "environment/inputs")
        (output / "instruction.md").write_text(task.joinpath("prompt.md").read_text())
        # Keep the same external-scoring contract as run_eval.render_task().
        (output / "tests/test.sh").write_text(
            "#!/bin/sh\n"
            'echo "Use the Chaperone Harbor controller for scientific scoring; see dataset README." >&2\n'
            "exit 2\n"
        )
        (output / "environment/Dockerfile").write_text(
            "FROM python:3.12-slim\nWORKDIR /workspace\nCOPY inputs/ /workspace/inputs/\n"
        )
        config = {
            "schema_version": "1.3",
            "task": {
                "name": f"{org}/chaperone-{task.name}",
                "description": "Chaperone controller task; external scientific scoring required",
            },
            "metadata": {"scoring_mode": "chaperone-controller-external"},
            "environment": {"cpus": 2, "memory_mb": 4096},
        }
        TaskConfig.model_validate(config)
        (output / "task.toml").write_text(toml.dumps(config))
        value, _ = Packager.compute_content_hash(output)
        refs.append({"name": config["task"]["name"], "digest": "sha256:" + value})
    manifest = DatasetManifest.model_validate(
        {
            "dataset": {
                "name": name,
                "description": "Private Chaperone evaluation tasks and held-out controller assets",
            },
            "tasks": refs,
            "files": [
                {"path": p, "digest": "sha256:" + digest(destination / p)}
                for p in ("private-evals.tar.gz", "SHA256SUMS.json")
            ],
        }
    )
    (destination / "dataset.toml").write_text(manifest.to_toml())
    (destination / "README.md").write_text(
        "# Private Chaperone evaluation dataset\n\n"
        "This package preserves prompts, inputs, scientific scorers, reference "
        "submissions and audit fingerprints. Keep the dataset and every task private.\n\n"
        "Scoring requires the Chaperone controller, configured scientific containers, "
        "weights and Slurm. These are not standalone hosted-Harbor benchmarks. "
        "The task verifier exits with an error to prevent reporting an unmeasured reward.\n\n"
        "Download the dataset on the trusted controller, then run:\n\n"
        "```sh\npython scripts/private_dataset.py restore --dataset /path/to/download "
        "--output /private/evals_task\nexport CHAPERONE_EVALS_PRIVATE=/private/evals_task\n"
        "```\n\n"
        "Use run_eval.py to render the selected task, enforce the sample count and "
        "compute budget, launch the contained agent, and score its sequences afterward. "
        "Do not mount the dataset download, archive or Hub credentials into agent/worker containers. "
        "Restoration verifies every original file against SHA-256 before writing.\n"
    )
    return destination


def restore_dataset(dataset: Path, output: Path) -> Path:
    """Restore only verified regular files to a new private directory."""
    if output.exists():
        raise ValueError("Restore destination must not already exist")
    expected = json.loads((dataset / "SHA256SUMS.json").read_text())
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Empty/invalid dataset checksum inventory")
    with tarfile.open(dataset / "private-evals.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in seen
                or member.name not in expected
            ):
                raise ValueError(f"Unsafe or unexpected archive member: {member.name}")
            seen.add(member.name)
            with archive.extractfile(member) as handle:
                if (
                    hashlib.file_digest(handle, "sha256").hexdigest()
                    != expected[member.name]
                ):
                    raise ValueError(f"Checksum mismatch: {member.name}")
        if seen != set(expected):
            raise ValueError("Archive does not match checksum inventory")
        output.mkdir(parents=True, mode=0o700)
        for member in members:
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--private-tasks", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--name", required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--dataset", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(export_dataset(args.private_tasks, args.output, args.name))
    else:
        print(restore_dataset(args.dataset, args.output))


if __name__ == "__main__":
    main()
