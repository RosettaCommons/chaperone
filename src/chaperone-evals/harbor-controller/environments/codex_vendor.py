"""Locate the Codex vendor directory bound into the agent image.

This lived in two places — run_eval and the Apptainer environment — and they
drifted: run_eval learned about standalone releases, apptainer.py did not, so
the day Codex was installed standalone every trial died seconds in with
"Could not locate the native Codex binary". One copy now, imported by both.
"""

import shutil
from pathlib import Path


def find_codex_vendor() -> Path:
    """Return the directory holding ``bin/`` and ``codex-path/``.

    Both are required: the probe and the trial put
    ``/opt/harbor-codex/bin:/opt/harbor-codex/codex-path`` on PATH, so a vendor
    directory missing either produces a container that cannot run the agent.

    Two layouts exist. A standalone release keeps the pair at the release root,
    which is the launcher's own grandparent. An npm install nests it under
    ``node_modules/@openai/codex-linux-*/vendor/<triple>/``.
    """
    launcher = shutil.which("codex")
    if launcher is None:
        raise FileNotFoundError(
            "Codex CLI is not installed on the host; it is needed as the "
            "agent runtime binary"
        )
    package_root = Path(launcher).resolve().parent.parent

    if (package_root / "bin" / "codex").is_file() and (
        package_root / "codex-path"
    ).is_dir():
        return package_root

    candidates = sorted(
        package_root.glob(
            "node_modules/@openai/codex-linux-*/vendor/*-unknown-linux-*/bin/codex"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate the native Codex binary beneath {package_root}. "
            "Expected either a standalone release (bin/ and codex-path/ at the "
            "release root) or an npm install under node_modules/@openai/."
        )
    return candidates[0].parent.parent
