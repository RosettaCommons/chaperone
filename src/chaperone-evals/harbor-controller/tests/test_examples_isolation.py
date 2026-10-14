"""Agent snapshots exclude worked examples, recipe docs and held-out material."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts.snapshot_harness import build, validate_snapshot, iter_bundle_files
from task_paths import EVALS_ROOT, PRIVATE_TASKS_ENV, private_tasks_root

REPO = EVALS_ROOT.parent.parent
STRUCTURE_SUFFIXES = {
    ".pdb",
    ".cif",
    ".sdf",
    ".mol2",
}  # coordinates match digit patterns


def _tracked_examples() -> list[Path]:
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--", "examples"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0 or not listing.stdout:
        pytest.skip("examples/ is not tracked in this checkout")
    return [REPO / name for name in listing.stdout.split("\0") if name]


@pytest.fixture
def private_root():
    if not os.environ.get(PRIVATE_TASKS_ENV):
        pytest.skip("Private corpus audit requires CHAPERONE_EVALS_PRIVATE")
    return private_tasks_root()


def _held_out_digests(private_root) -> dict[str, str]:
    digests = {}
    for task in private_root.iterdir():
        for part in ("score", "example"):
            for path in (task / part).rglob("*") if (task / part).is_dir() else []:
                if path.is_file():
                    digests[hashlib.sha256(path.read_bytes()).hexdigest()] = (
                        f"{task.name}/{part}/{path.relative_to(task / part)}"
                    )
    return digests


def test_no_agent_bundled_file_is_a_copy_of_a_held_out_file(private_root):
    held_out = _held_out_digests(private_root)
    assert held_out
    # Public source fixtures can also be evaluator references. They must never
    # enter the agent workspace; do not obscure their provenance by rewriting
    # coordinates merely to change a digest.
    copies = [
        f"{path.relative_to(REPO)} == {held_out[digest]}"
        for _, path in iter_bundle_files(REPO)
        if (digest := hashlib.sha256(path.read_bytes()).hexdigest()) in held_out
    ]

    assert not copies, copies


def test_no_agent_bundled_text_matches_an_audit_pattern(private_root):
    hints = [
        (
            entry["label"],
            re.compile(entry["pattern"], re.I if entry.get("ignore_case", True) else 0),
        )
        for entry in json.loads((private_root / "audit_hints.json").read_text())
    ]
    hits = []
    # Public examples may describe the same scientific systems as benchmark
    # tasks. Only agent-visible content must be free of task-specific hints.
    for _, path in iter_bundle_files(REPO):
        if path.suffix in STRUCTURE_SUFFIXES:
            continue
        text = path.read_text(errors="ignore")
        hits += [
            f"{path.relative_to(REPO)}: {label}"
            for label, pattern in hints
            if pattern.search(text)
        ]

    assert not hits, hits


def test_real_bundle_contains_no_example_files(tmp_path, monkeypatch):
    # Structural CI check uses synthetic fingerprints; real corpus checks above
    # are separately required on the private controller before launching trials.
    if not os.environ.get(PRIVATE_TASKS_ENV):
        private = tmp_path / "private"
        private.mkdir()
        (private / "audit_hints.json").write_text(json.dumps([
            {"label": "synthetic", "pattern": "HELD_OUT_TEST_SENTINEL_48723"}
        ]))
        monkeypatch.setenv(PRIVATE_TASKS_ENV, str(private))
    examples = {path.relative_to(REPO).as_posix() for path in _tracked_examples()}

    seed = build(REPO, tmp_path / "bundles")

    bundled = {
        path.relative_to(seed).as_posix() for path in seed.rglob("*") if path.is_file()
    }
    assert bundled
    assert not bundled & examples
    assert not [name for name in bundled if "examples" in Path(name).parts]
    assert {name for name in bundled if name.startswith("docs/")} <= {
        "docs/site/blocks-reference.md"
    }
    assert "docs/site/core-examples.md" not in bundled
    assert not any(name.startswith("docs/site/workflows/") for name in bundled)
    assert validate_snapshot(seed, seed.name) == seed.name
