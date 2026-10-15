#!/usr/bin/env python3
"""Build a minimal, history-free Chaperone bundle from an explicit source tree.

The harness needs executable Chaperone source and its agent-skill instructions;
it does not need a repository clone.  This module copies an allowlisted set of
regular files directly from the caller-supplied working tree.  It never invokes
Git, follows no symlinks, and never copies repository metadata, documentation
sites beyond the generated block reference, examples, tests, campaigns, prior
outputs, or evaluation artifacts.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import unicodedata
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator


ROOT_FILES = frozenset(
    {
        "AGENTS.md",
        "SKILL.md",
        "pyproject.toml",
    }
)
SOURCE_SUFFIXES = frozenset({".py", ".md", ".xml", ".params"})
SOURCE_BASENAMES = frozenset({"flags_rescore_genpot"})
SKILL_SUFFIXES = frozenset({".md", ".py", ".json", ".toml", ".yaml", ".yml"})
FORBIDDEN_GIT_NAMES = frozenset({".git", ".gitignore", ".gitmodules", ".gitattributes"})
FORBIDDEN_PARTS = frozenset(
    {
        ".claude",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "campaigns",
        "evals",
        "examples",
        "outputs",
        "runs",
        "tests",
    }
)
# Only these packages under src/ are bundled. Everything else there, notably
# src/chaperone-evals (this controller, the held-out scoring toolkit and the
# public task material), is never copied and is not even walked.
BUNDLED_SRC_PACKAGES = frozenset({"blocks", "chaperone", "chaperone_cli"})
FORBIDDEN_ARTIFACT_NAMES = frozenset(
    {
        "score.json",
        "score_per_design.csv",
        "submission.csv",
        "top_design.cif",
    }
)
REQUIRED_FILES = frozenset(
    {
        "AGENTS.md",
        "SKILL.md",
        "pyproject.toml",
        "src/blocks/__init__.py",
        "src/chaperone/__init__.py",
        "src/chaperone_cli/job_wait.py",
        "src/chaperone_cli/main.py",
    }
)
_PACKAGING_SKILLS = frozenset({"order-collection", "order-designs", "order-ssm"})
_BOOKKEEPING_TERMS = (
    "terminal_report.json",
    "terminal report",
    "lineage.jsonl",
    "lineage sidecar",
    "provenance.json",
    "provenance file",
    "submission/backbones",
    "submission/run_meta.json",
    "run_meta.json",
    "public_score.json",
    "campaign ledger",
    "campaign.json",
    "decision journal",
    "acceptance-contract copy",
    "acceptance_contract.json copy",
    "cost summary",
    "cost report",
    "trial record",
    "final validation report",
)
_MANDATORY_WORDS = (
    "must",
    "required",
    "require ",
    "write ",
    "create ",
    "provide ",
    "submit ",
    "include ",
    "retain ",
    "preserve ",
    "record ",
    "initialize ",
    "maintain ",
)
_NEGATIONS = ("do not", "don't", "never", "not required", "optional", "ignored")

# The harness receives reusable scientific APIs, not knowledge of any held-out
# eval target. The fingerprints that detect an accidental leak (target names and
# ids, calibrated cutoffs, panel sizes) are themselves held-out facts, so they
# are not stored here: they are read from audit_hints.json in the tasks tree, and
# a bundle cannot be built without them.
HINTS_FILENAME = "audit_hints.json"


def load_evaluation_hints(
    private_root: str | os.PathLike | None = None,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Load the (label, pattern) fingerprints from the tasks tree.

    The file is a JSON list of ``{"label": ..., "pattern": ..., "ignore_case":
    true}`` objects. Keep the list narrow and target-specific so ordinary
    scientific vocabulary stays available while accidental fingerprints fail
    closed before a workspace is launched.
    """
    from task_paths import private_tasks_root

    path = private_tasks_root(private_root) / HINTS_FILENAME
    if not path.is_file():
        raise RuntimeError(
            f"Held-out audit patterns not found: {path}. A harness bundle cannot "
            "be built without them; add audit_hints.json."
        )
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{path} must be a non-empty JSON list of patterns")
    return tuple(
        (
            str(entry["label"]),
            re.compile(
                entry["pattern"],
                re.IGNORECASE if entry.get("ignore_case", True) else 0,
            ),
        )
        for entry in entries
    )


def audit_model_instructions(paths: list[Path], snapshot: Path | None = None) -> None:
    """Reject mandatory model-authored bookkeeping outside explicit skills."""

    candidates = list(paths)
    if snapshot is not None:
        candidates.extend(
            path
            for relative, path in iter_bundle_files(snapshot)
            if path.suffix == ".md"
            if not any(part in _PACKAGING_SKILLS for part in relative.parts)
        )
    offenders: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        # Paragraph scope recognizes explanatory negatives ("do not create")
        # without letting a later mandatory paragraph hide behind them.
        for paragraph_number, paragraph in enumerate(
            re.split(r"\n\s*\n", text), start=1
        ):
            normalized = " ".join(paragraph.lower().split())
            if not any(term in normalized for term in _BOOKKEEPING_TERMS):
                continue
            if any(negation in normalized for negation in _NEGATIONS):
                continue
            if any(word in normalized for word in _MANDATORY_WORDS):
                offenders.append(f"{path}:paragraph {paragraph_number}")
    if offenders:
        raise RuntimeError(
            "Model-facing instructions require controller-owned bookkeeping: "
            + ", ".join(offenders)
        )


def _is_allowed(relative: PurePosixPath) -> bool:
    """Return whether a regular source file belongs in the agent bundle."""
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES
    if relative.parts[:1] == ("src",):
        if len(relative.parts) < 3 or relative.parts[1] not in BUNDLED_SRC_PACKAGES:
            return False
        # Generic evaluation/reporting helpers are not required to compose or
        # execute scientific Chaperone blocks and do not belong in an agent
        # evaluation workspace.
        if relative.as_posix() == "src/chaperone/evals.py":
            return False
        return relative.suffix in SOURCE_SUFFIXES or relative.name in SOURCE_BASENAMES
    if relative.parts[:2] in {
        (".agents", "skills"),
        (".agents", "templates"),
    }:
        return relative.suffix in SKILL_SUFFIXES
    return relative.as_posix() == "docs/site/blocks-reference.md"


def _forbidden_reason(relative: PurePosixPath) -> str | None:
    # Compared case-insensitively so `Examples/` or `Campaigns/` cannot slip past
    # the lower-case lists on a case-sensitive filesystem.
    parts = [part.lower() for part in relative.parts]
    if any(part in FORBIDDEN_GIT_NAMES for part in parts):
        return "Git metadata"
    if any(part in FORBIDDEN_PARTS for part in parts):
        return "forbidden generated or evaluation directory"
    if relative.name.lower() in FORBIDDEN_ARTIFACT_NAMES:
        return "evaluation artifact"
    return None


def validate_advertised_skills(root: Path) -> None:
    """Reject model-facing skill routes whose SKILL.md is not bundled.

    AGENTS.md advertises slash commands, while INDEX.md uses the first column
    of its Markdown table. Keeping the parser intentionally narrow avoids
    treating ordinary backticked package or file names as skill references.
    """
    skills_root = root / ".agents" / "skills"
    available = {
        path.parent.name
        for path in skills_root.glob("*/SKILL.md")
        if path.is_file() and not path.is_symlink()
    }
    advertised: dict[str, str] = {}
    agents = root / "AGENTS.md"
    if agents.is_file():
        for name in re.findall(r"`/([a-z][a-z0-9-]*)`", agents.read_text()):
            advertised[name] = "AGENTS.md"
    index = skills_root / "INDEX.md"
    if index.is_file():
        for line in index.read_text().splitlines():
            if not line.lstrip().startswith("|"):
                continue
            first_column = line.split("|", 2)[1]
            for name in re.findall(r"`([a-z][a-z0-9-]*)`", first_column):
                advertised[name] = ".agents/skills/INDEX.md"
    missing = sorted(set(advertised) - available)
    if missing:
        details = ", ".join(f"{name} ({advertised[name]})" for name in missing)
        raise RuntimeError(
            "Chaperone bundle advertises skills without a SKILL.md: " + details
        )


def iter_bundle_files(source: Path) -> Iterator[tuple[PurePosixPath, Path]]:
    """Yield the allowlisted regular files in a source tree.

    Symlinks beneath an allowlisted tree are rejected instead of followed. This
    prevents an apparently safe path from importing files outside the supplied
    checkout.
    """
    source = source.resolve()
    candidates: list[tuple[PurePosixPath, Path]] = []
    roots = [
        source / "src",
        source / ".agents" / "skills",
        source / ".agents" / "templates",
    ]
    for name in sorted(ROOT_FILES):
        path = source / name
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlink in Chaperone source: {name}")
        if path.is_file():
            candidates.append((PurePosixPath(name), path))
    block_reference = source / "docs" / "site" / "blocks-reference.md"
    if block_reference.is_symlink():
        raise RuntimeError(
            "Refusing symlink in Chaperone source: docs/site/blocks-reference.md"
        )
    if block_reference.is_file():
        candidates.append(
            (PurePosixPath("docs/site/blocks-reference.md"), block_reference)
        )

    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink():
            raise RuntimeError(
                f"Refusing symlinked Chaperone source directory: {root.relative_to(source)}"
            )
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            if current_path == source / "src":
                # Prune before the symlink check below: the evals tree carries
                # its own virtualenv, and none of it belongs in a bundle.
                directory_names[:] = [
                    name for name in directory_names if name in BUNDLED_SRC_PACKAGES
                ]
            for directory_name in tuple(directory_names):
                directory = current_path / directory_name
                if directory.is_symlink():
                    raise RuntimeError(
                        "Refusing symlink in Chaperone source: "
                        f"{directory.relative_to(source)}"
                    )
            for file_name in file_names:
                path = current_path / file_name
                relative = PurePosixPath(path.relative_to(source).as_posix())
                if path.is_symlink():
                    raise RuntimeError(
                        f"Refusing symlink in Chaperone source: {relative}"
                    )
                if _is_allowed(relative):
                    candidates.append((relative, path))

    for relative, path in sorted(candidates, key=lambda item: item[0].as_posix()):
        reason = _forbidden_reason(relative)
        if reason is not None:
            raise RuntimeError(f"Refusing {reason} in Chaperone bundle: {relative}")
        if not path.is_file():
            raise RuntimeError(
                f"Chaperone bundle entry is not a regular file: {relative}"
            )
        yield relative, path


def _inventory_hash(files: list[tuple[PurePosixPath, Path]]) -> str:
    digest = hashlib.sha256()
    for relative, path in files:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def snapshot_hash(root: Path) -> str:
    return _inventory_hash(list(iter_bundle_files(root)))


def assert_no_git_metadata(root: Path) -> None:
    """Reject every Git control/configuration file at any depth."""
    offenders = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.name in FORBIDDEN_GIT_NAMES
    )
    if offenders:
        preview = ", ".join(str(path) for path in offenders[:5])
        if len(offenders) > 5:
            preview += f", ... ({len(offenders)} total)"
        raise RuntimeError(f"Bundle contains forbidden Git metadata: {preview}")


_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def _scannable_text(data: bytes) -> str:
    """Decode a file for fingerprint scanning, defeating trivial obfuscation.

    UTF-16 files (recognised by their byte-order mark) are decoded as such, and
    compatibility forms plus zero-width characters are folded away so a pattern
    cannot be hidden inside a token.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    else:
        text = data.decode("utf-8", errors="replace")
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE)


def assert_no_evaluation_hints(root: Path, hints=None) -> None:
    """Reject target-specific scientific hints from a model-facing bundle.

    File names are scanned along with contents.
    """
    hints = load_evaluation_hints() if hints is None else hints
    offenders: list[str] = []
    for relative, path in iter_bundle_files(root):
        text = _scannable_text(path.read_bytes())
        name = unicodedata.normalize("NFKC", relative.as_posix()).translate(_INVISIBLE)
        for label, pattern in hints:
            if pattern.search(name) is not None:
                offenders.append(f"{relative} ({label}, in the file name)")
                continue
            match = pattern.search(text)
            if match is None:
                continue
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line} ({label})")
    if offenders:
        preview = ", ".join(offenders[:8])
        if len(offenders) > 8:
            preview += f", ... ({len(offenders)} total)"
        raise RuntimeError(
            "Chaperone bundle contains evaluation-specific hints: " + preview
        )


def validate_snapshot(
    root: Path, expected_hash: str | None = None, *, check_private_hints: bool = True
) -> str:
    """Validate structure, contents and (on the trusted submission host) private hints.

    Compute-side revalidation may omit the private audit only when checking the
    exact SHA-256 of a bundle already audited by the submission host. The outer
    job intentionally receives no private task data or hint patterns.
    """
    if not check_private_hints and (
        expected_hash is None or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
    ):
        raise ValueError("Private-audit revalidation requires an audited SHA-256")
    root = root.expanduser()
    if root.is_symlink():
        raise RuntimeError(f"Chaperone bundle root may not be a symlink: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Chaperone bundle is not a directory: {root}")
    assert_no_git_metadata(root)
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise RuntimeError(f"Bundle contains forbidden symlink: {relative}")
        if path.is_file():
            reason = _forbidden_reason(relative)
            if reason is not None:
                raise RuntimeError(f"Bundle contains {reason}: {relative}")
            if not _is_allowed(relative):
                raise RuntimeError(f"Bundle contains non-allowlisted file: {relative}")
            actual_files.add(relative.as_posix())
    missing = sorted(REQUIRED_FILES - actual_files)
    if missing:
        raise RuntimeError(
            "Chaperone bundle is missing required files: " + ", ".join(missing)
        )
    validate_advertised_skills(root)
    if check_private_hints:
        assert_no_evaluation_hints(root)
    digest = snapshot_hash(root)
    if expected_hash is not None and digest != expected_hash:
        raise RuntimeError(
            f"Chaperone bundle hash mismatch: expected {expected_hash}, found {digest}"
        )
    return digest


def build(source: Path, output_root: Path) -> Path:
    """Copy a minimal raw-file bundle from an explicit source directory."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"Chaperone source directory not found: {source}")
    validate_advertised_skills(source)
    files = list(iter_bundle_files(source))
    relative_files = {relative.as_posix() for relative, _ in files}
    missing = sorted(REQUIRED_FILES - relative_files)
    if missing:
        raise RuntimeError(
            "Chaperone source is missing required harness files: " + ", ".join(missing)
        )
    digest = _inventory_hash(files)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / digest
    manifest_path = output_root / f"{digest}.json"
    if destination.is_symlink():
        raise RuntimeError(
            f"Refusing symlinked Chaperone bundle destination: {destination}"
        )
    if destination.exists():
        validate_snapshot(destination, digest)
        return destination

    staging = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=output_root))
    try:
        for relative, source_file in files:
            destination_file = staging.joinpath(*relative.parts)
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file, follow_symlinks=False)
        validate_snapshot(staging, digest)
        try:
            staging.rename(destination)
        except OSError as error:
            # Another launcher may publish the same content-addressed bundle
            # after our initial existence check.  A non-empty destination is
            # success only when that winner's bundle validates to the exact
            # digest we independently computed.
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            if destination.is_symlink():
                raise RuntimeError(
                    f"Refusing symlinked Chaperone bundle destination: {destination}"
                ) from error
            validate_snapshot(destination, digest)
            shutil.rmtree(staging)
        manifest = {
            "schema_version": 2,
            "source_basename": source.name,
            "content_sha256": digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(files),
            "policy": "raw-allowlisted-files-no-git",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a minimal history-free Chaperone harness bundle."
    )
    parser.add_argument(
        "--chaperone-path",
        type=Path,
        required=True,
        help="Explicit Chaperone working-tree path whose raw allowlisted files are copied.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.chaperone_path, args.output_root))


if __name__ == "__main__":
    main()
