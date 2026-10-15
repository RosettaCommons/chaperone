"""Guarded repair for RFO's inference transform, applied when building its image.

RFO sequence candidates get random placeholder coordinates in [0, 1]. RF3's
training cyclic-bond detector interprets these as real terminal geometry and
adds spurious closure bonds to linear proteins. Do not infer topology from
those placeholders. Explicit RF3 cyclic design is not implemented by RFO.
Also disable the checkpoint's lab-specific residue cache so cluster-local
filesystem contents cannot silently change inference conditioning.
"""

import hashlib
from pathlib import Path
import sys

EXPECTED_SHA256 = "4bb3a8041938a50bc3ffb1628859cdd6789b08a980a271f58e66114dd7568682"
_BEFORE = '            "run_confidence_head": True,\n'
_AFTER = (
    _BEFORE
    + '            "add_cyclic_bonds": False,\n'
    + '            "residue_cache_dir": None,\n'
)


def repair(path: Path) -> None:
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError(
            f"Unexpected upstream RFO source: {path}; review cyclic-bond repair"
        )
    if source.count(_BEFORE) != 1:
        raise RuntimeError(
            "Expected exactly one RFO inference-transform overrides mapping"
        )
    path.write_text(source.replace(_BEFORE, _AFTER))


if __name__ == "__main__":
    repair(Path(sys.argv[1]))
