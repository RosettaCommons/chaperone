"""Guard: the project is called Chaperone; the old name ``pipelines`` must not return.

The project was renamed from "Pipelines" to "Chaperone" with no deprecated
aliases. This test scans every tracked text file and fails if the old name shows
up as a project identity (imports, ``PIPELINES_*`` env vars, the ``pipelines_cli``
package, ``--pipelines-*`` flags, the display name) outside the allowlists below.

Every allowlist entry carries a reason. Add to a list only for something that
really is not our name: a path that exists on the cluster, a file already on disk,
or ``pipelines`` used as an ordinary English noun for a design workflow.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = "tests/test_naming.py"

# ---------------------------------------------------------------------------
# Strings that contain "pipelines" but are not our name. They are masked out of
# the text before scanning.
# ---------------------------------------------------------------------------
SITE_AND_IMAGE_PATHS = [
    # Shared IPD tree: weights, software, containers. Exists on disk; renaming it
    # is a filesystem operation, not a code change.
    r"/projects/ml/pipelines[\w./{},:-]*",
    # The same tree as seen through the /mnt alias on compute nodes (the test
    # fixtures also use a checkout path below it).
    r"/mnt/projects/ml/pipelines[\w./-]*",
    # Shared apptainer image directory.
    r"/net/software/containers/pipelines[\w./{},:-]*",
    # Interpreter root baked into the existing worker SIFs. The directory is part
    # of the image; it follows the image rebuild, not this repo.
    r"/opt/pipelines(?![\w-])",
    # Existing agent image files on disk (base layer, hardened worker image) and
    # the symlink the unvendored agent/ tooling refers to.
    r"pipelines_agent(?:_base|_hardened_\d+)?\.sif",
    r"agent/pipelines\.sif",
    # A previous developer's home checkout in a pixi task; not ours, not editable.
    r"/home/anmittal/Code/protein-design-pipelines[\w./-]*",
]

# (path regex, text regex, reason). The text is matched case-insensitively.
NOUN_USES = [
    (r"^docs/README\.md$", r"original Pipelines site's", "attribution to the upstream documentation theme"),
    (r"^README\.md$", r"reproducible pipelines", "generic workflows"),
    (r"^README\.md$", r"debugging pipelines", "generic workflows"),
    (r"^README\.md$", r"worked pipelines", "generic workflows"),
    (r"^\.agents/skills/design/SKILL\.md$", r"worked pipelines", "generic workflows"),
    (r"^\.agents/skills/slurm-background/SKILL\.md$", r"design pipelines", "generic"),
    (r"^docs/site/syntax\.md$", r"running pipelines", "generic workflows"),
    (r"^examples/benchmarks/ppi/README\.md$", r"Both pipelines", "two workflows"),
    (r"^src/blocks/.*\.py$", r"blocks for design pipelines", "generic workflows"),
    (r"^src/chaperone/data/handler\.py$", r'"evaluation sidecar" pipelines', "generic"),
    # campaign_inspect summarises the .pipeline run manifests under this key; the
    # key names the runs, not the project.
    (r"^src/chaperone_cli/campaign_inspect\.py$", r'"pipelines": \{', "output key"),
    (r"^tests/test_monitoring\.py$", r'\["pipelines"\]', "output key"),
    # Baseline isolation must stay hostile to BOTH names: real site paths and
    # images that predate the rename still say "pipelines".
    (
        r"^src/chaperone-evals/harbor-controller/environments/baseline\.py$",
        r"pipelines",
        "baseline isolation checks cover the pre-rename names too",
    ),
    # Inside the existing hardened worker image the venv root is /opt/pipelines.
    # (The mask above covers the bare path; these lines spell it in other forms.)
    (r"^src/chaperone-evals/harbor-controller/.*", r"/opt/pipelines", "image path"),
    (r"^src/chaperone_cli/job_wait\.py$", r"/opt/pipelines", "image path"),
]

# Held-out scorers, example solutions and their notes are not edited in place
# (see the evals README). Only the *mechanical* rewrites needed for them to run
# were applied; stale prose/comments/docstrings in them may still say pipelines.
HELD_OUT = re.compile(
    r"^src/chaperone-evals/evals_task/[^/]+/(score|example)/"
    r"|^src/chaperone-evals/evals_task/audit_hints\.json$"
)

# ---------------------------------------------------------------------------
# What counts as the old project identity.
# ---------------------------------------------------------------------------
# Never allowed, held-out files included: anything that would stop code running.
RUNTIME_IDENTITY = re.compile(
    r"(?<![A-Za-z])PIPELINES_[A-Z0-9_]+"  # env vars
    r"|\bpipelines_cli\b"  # CLI package
    r"|\b(?:from|import)\s+pipelines\b"  # imports
    r"|\bPipelinesSettings\b"
)
# Also not allowed outside held-out files: the old name in any other spelling.
ANY_OLD_NAME = re.compile(r"pipelines", re.IGNORECASE)

SKIP_SUFFIXES = {".png", ".jpg", ".pdb", ".cif", ".ckpt", ".pt", ".sif", ".gz"}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.split("\0") if f]


def _mask(text: str) -> str:
    for pattern in SITE_AND_IMAGE_PATHS:
        text = re.sub(pattern, "<site-path>", text)
    return text


def _scan(relative: str, text: str) -> list[str]:
    """Return offending ``path:line: text`` strings for one file."""
    problems = []
    held_out = bool(HELD_OUT.match(relative))
    noun_rules = [
        (re.compile(path_re), re.compile(text_re, re.IGNORECASE))
        for path_re, text_re, _ in NOUN_USES
        if re.match(path_re, relative)
    ]
    for number, line in enumerate(_mask(text).splitlines(), start=1):
        if "pipelines" not in line.lower():
            continue
        if RUNTIME_IDENTITY.search(line):
            problems.append(f"{relative}:{number}: {line.strip()[:140]}")
            continue
        if held_out:
            continue
        remaining = line
        for _, text_rule in noun_rules:
            remaining = text_rule.sub("<noun>", remaining)
        if ANY_OLD_NAME.search(remaining):
            problems.append(f"{relative}:{number}: {line.strip()[:140]}")
    return problems


def test_old_project_name_does_not_return():
    problems: list[str] = []
    for relative in _tracked_files():
        if relative == THIS_FILE or Path(relative).suffix.lower() in SKIP_SUFFIXES:
            continue
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        problems.extend(_scan(relative, text))
    assert not problems, (
        "The project is named Chaperone. Old-name occurrences found (rename them, or "
        "add a justified entry to the allowlists in tests/test_naming.py):\n"
        + "\n".join(problems[:50])
    )


@pytest.mark.parametrize(
    "line",
    [
        "from pipelines import ExecutionEngine",
        "import pipelines_cli.main",
        "export PIPELINES_REPO=/x",
        "pipelines install foundry",
        "Pipelines is a framework",
        "--pipelines-path /x",
        "src/pipelines/io.py",
    ],
)
def test_scanner_flags_old_identity(line):
    assert _scan("docs/example.md", line)


@pytest.mark.parametrize(
    "line",
    [
        "weights live in /projects/ml/pipelines/weights/boltz2",
        "image /net/software/containers/pipelines/pipelines_agent_base.sif",
        "reproducible pipelines for design",
        "from chaperone import init_cli",
    ],
)
def test_scanner_allows_listed_uses(line):
    assert not _scan("README.md", line)
