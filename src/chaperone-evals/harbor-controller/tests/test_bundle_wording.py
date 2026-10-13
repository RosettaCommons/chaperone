"""Files shipped to the agent must not announce that it is being evaluated."""

from __future__ import annotations

import re

from scripts.snapshot_harness import iter_bundle_files
from task_paths import default_chaperone_repo

# Prose that gives the game away. Code identifiers such as HARBOR_HOST_WORKSPACE are
# not matched: the first pattern needs the word "Harbor" with no underscore after it.
FORBIDDEN = (
    (r"\bHarbor\b", "names the runner"),
    (r"held[- ]out", "refers to hidden scoring"),
    (
        r"\beval(?:uation)? (?:transcript|snapshot|worker|controller|task|run|harness)s?\b",
        "refers to the evaluation setup",
    ),
    (r"\bchaperone\.evals\b", "points at a module the bundle does not contain"),
)


def test_bundled_prose_does_not_announce_an_evaluation():
    offenders = []
    for relative, path in iter_bundle_files(default_chaperone_repo()):
        text = path.read_text(errors="replace")
        for pattern, reason in FORBIDDEN:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{relative}:{line} ({reason})")

    assert not offenders, "\n".join(offenders)
