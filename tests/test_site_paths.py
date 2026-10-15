"""Guard: tracked files must not carry digs (IPD) site paths outside an allowlist.

Site paths belong in ``src/chaperone/site_profiles/digs.env`` and the evals
harness. Elsewhere they leak one site's layout into code and docs that should
work anywhere. This test scans every tracked text file for ``/home/<user>``,
``/net/``, ``/projects``, ``/mnt/home``, ``/software/`` and ``/databases/`` and
fails on any line not covered by ``ALLOWED``.

Each ALLOWED entry is ``(path regex, line regex, reason)``: a hit is excused only
when its file matches the path *and* the line matches the pattern. Keep both
narrow. To add a path, prefer moving it to the site profile; if it really is a
deliberate on-prem fallback, add an entry with a reason.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = "tests/test_site_paths.py"

# A site path starts a path component (so ``eddylab.org/software/`` is not one).
SITE_PATH = re.compile(
    r"(?<![\w.~-])/(?:home/[\w.$-]+|net/|projects\b|mnt/home|software/|databases/)"
)

SKIP_SUFFIXES = {".png", ".jpg", ".pdb", ".cif", ".ckpt", ".pt", ".sif", ".gz", ".zip"}


def _files(*names: str) -> str:
    """Path regex matching exactly these repo-relative files."""
    return "^(?:" + "|".join(re.escape(name) for name in names) + ")$"


ALLOWED: list[tuple[str, str, str]] = [
    # -- the digs site itself ------------------------------------------------
    (r"^src/chaperone/site_profiles/", r"", "the digs profile is where site paths live"),
    (r"^src/chaperone-evals/", r"", "the evals harness intentionally targets digs"),
    # -- container search roots / default roots, gated on the site -----------
    (
        _files("src/chaperone/containers.py"),
        r'^\s*"(?:/mnt)?/(?:net(?:/scratch)?|projects|mnt/net/scratch|mnt/projects)",$',
        "roots a run directory must never be bound at (safety list)",
    ),
    (
        _files("src/chaperone/containers.py", "src/chaperone_cli/job_wait.py"),
        r"/projects|/mnt/projects",
        "default /projects bind (and its /mnt/projects alias) when the host has it",
    ),
    (
        _files("src/chaperone/settings.py"),
        r"/net/scratch/\{user\}/chaperone",
        "default CHAPERONE_ROOT, used only when CHAPERONE_SITE=digs",
    ),
    (
        _files("src/chaperone/engine/chunking.py"),
        r"/net/scratch.*(?:acregmax|mounts)",
        "comments describing the digs NFS attribute-cache behaviour",
    ),
    # -- on-prem fallbacks in blocks ------------------------------------------
    (
        _files("src/blocks/scripts/ddg_protocol.xml"),
        r"/software/psipred4/",
        "commented-out psipred filter in the default ddG protocol",
    ),
    (
        _files("src/blocks/networks/prediction/alphafold3.py"),
        r'"--model_dir=/databases/alphafold",',
        "container-internal AF3 fallback, overridden by configured model directory",
    ),
    # -- docs and skills that say they are IPD / digs specific ---------------
    (
        _files("docs/site/evaluations.md"),
        r"`/projects`|/net/scratch/\$USER/chaperone-evals",
        "evals harness isolation and its digs run root",
    ),
    (
        _files("docs/site/syntax.md"),
        r"/net/scratch/\{user\}/chaperone",
        "documents the digs default CHAPERONE_ROOT",
    ),
    (
        _files("docs/extras/containers.md"),
        r"/projects/ml",
        "IPD example paths and the .env.ipd trigger",
    ),
    (
        _files(
            ".agents/skills/order-designs/SKILL.md",
            ".agents/skills/rosetta-params/SKILL.md",
        ),
        r"/(?:net|software)/",
        "skill states at the top that it is IPD-specific",
    ),
    (
        _files(".agents/skills/slurm-background/SKILL.md"),
        r"/net/scratch",
        "skill states at the top that its values are IPD-specific",
    ),
    (
        _files(".agents/skills/design/SKILL.md", ".agents/skills/scripting/SKILL.md"),
        r"/net/scratch",
        "IPD scratch used as the example run directory",
    ),
    (
        _files(".agents/skills/scripting/SKILL.md"),
        r"/mnt/home|/home/user/chaperone",
        "bind-root advice and a placeholder checkout path",
    ),
    (
        _files(".agents/skills/build-block/SKILL.md"),
        r'\("/mnt/home", "/mnt/home"\)',
        "example of the standard bind roots",
    ),
    (
        _files(".agents/skills/setup/SKILL.md"),
        r"`/projects` or `/net`",
        "explains how IPD is detected",
    ),
    (
        _files(".agents/skills/visualize/SKILL.md"),
        r"/net/software/lab/catcif_tools/",
        "IPD-only PyMOL helper",
    ),
    # -- tests: fixtures that mirror the constants above ----------------------
    (
        _files("tests/test_bindcraft2.py", "tests/test_proteina_complexa.py", "tests/test_rfoptimization.py"),
        r'for site_path in |assert "/net/" not in recipe',
        "negative assertions that new recipes contain no institutional paths",
    ),
    (
        _files(
            "tests/blocks/test_container_definitions.py",
            "tests/blocks/test_container_selection.py",
            "tests/test_monitoring.py",
            "tests/test_naming.py",
            "tests/test_network_blocks.py",
            "tests/test_runtime.py",
        ),
        r"",
        "fixtures and scanners that reference the on-prem constants",
    ),
]


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.split("\0") if f]


def _scan(relative: str, text: str) -> list[str]:
    """Return offending ``path:line: text`` strings for one file."""
    rules = [re.compile(line) for path, line, _ in ALLOWED if re.match(path, relative)]
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        if SITE_PATH.search(line) and not any(rule.search(line) for rule in rules):
            problems.append(f"{relative}:{number}: {line.strip()[:140]}")
    return problems


def test_no_unlisted_site_paths():
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
        "Site paths found outside the allowlist. Move them to the site profile, "
        "or add a justified entry to ALLOWED in tests/test_site_paths.py:\n"
        + "\n".join(problems[:50])
    )


def test_allowlist_entries_are_used():
    """A stale entry hides future leaks, so every one must still match something."""
    unused = set(range(len(ALLOWED)))
    for relative in _tracked_files():
        if relative == THIS_FILE or Path(relative).suffix.lower() in SKIP_SUFFIXES:
            continue
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for index, (path_re, line_re, _) in enumerate(ALLOWED):
            if index in unused and re.match(path_re, relative):
                if any(SITE_PATH.search(x) and re.search(line_re, x) for x in lines):
                    unused.discard(index)
    assert not unused, "Unused ALLOWED entries: " + ", ".join(
        ALLOWED[i][2] for i in sorted(unused)
    )


@pytest.mark.parametrize(
    "line",
    [
        'binds=[("/net/databases", "/db")]',
        "cache at /home/alice/cache",
        "rundir=/projects/ml/run",
        "ckpt=/databases/weights/x.pt",
        "bin=/software/rosetta/main",
    ],
)
def test_scanner_flags_site_paths(line):
    assert _scan("src/blocks/metrics/example.py", line)


@pytest.mark.parametrize(
    "line",
    [
        "wget http://eddylab.org/software/hmmer/hmmer.tar.gz",
        "see https://github.com/org/projects",
        "/home alone",
        "from chaperone import init_cli",
    ],
)
def test_scanner_ignores_other_text(line):
    assert not _scan("src/blocks/metrics/example.py", line)


def test_scanner_allows_listed_use_only_in_listed_file():
    line = 'binds=[("/mnt/home", "/mnt/home"), ("/projects", "/projects")]'
    assert _scan("src/blocks/networks/design/bindcraft.py", line)
    assert _scan("src/blocks/metrics/chirality.py", line)
    # In a listed file, an unrelated site path is still caught.
    assert _scan("src/blocks/networks/design/bindcraft.py", "x = '/home/alice/w'")
