"""Repair formal-charge dtype in the pinned Foundry revision before image validation."""

from hashlib import sha256
from pathlib import Path
import sys


_SOURCE_HASHES = {
    "297402d648f19a760c8ae6d00f790abc7c1916018da5ae2231b9ad37553bf5d2",  # RFD3
    "42f8e5b5b60178d72a52a700a30e4e5d6054e79e6fa7a766119ff78b35f35a88",  # RFD3NA
}


def patch(source: Path) -> None:
    original = source.read_text()
    old = 'array.set_annotation("charge", np.zeros(array.shape[0], dtype=float))'
    new = 'array.set_annotation("charge", np.zeros(array.shape[0], dtype=np.int8))'
    if (
        sha256(original.encode()).hexdigest() not in _SOURCE_HASHES
        or original.count(old) != 1
    ):
        raise RuntimeError("Foundry formal-charge source changed; review the repair")
    source.write_text(original.replace(old, new))


def check(model: str = "rfd3") -> None:
    from importlib import import_module
    import tempfile

    import biotite.structure as struc
    import numpy as np
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure.io import pdbx

    set_common_annotations = import_module(
        f"{model}.utils.inference"
    ).set_common_annotations

    # Generated atoms must not promote integer motif charges to floating point.
    generated = struc.AtomArray(1)
    generated.bonds = struc.BondList(1)
    generated.chain_id[:] = "A"
    generated.res_id[:] = 1
    generated.res_name[:] = "GLY"
    generated.atom_name[:] = "CA"
    generated.element[:] = "C"
    generated = set_common_annotations(generated)
    motif = generated.copy()
    motif.res_id[:] = 2
    motif.charge[:] = 1
    atoms = struc.concatenate([generated, motif])
    assert np.issubdtype(atoms.charge.dtype, np.integer)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "charged.cif"
        to_cif_file(atoms, path, include_entity_poly=False)
        result = pdbx.get_structure(
            pdbx.CIFFile.read(path), model=1, extra_fields=["charge"]
        )
        np.testing.assert_array_equal(result.charge, [0, 1])


if __name__ == "__main__":
    for name in sys.argv[1:]:
        if name != "--check":
            source = Path(name)
            patch(source)
            if "--check" in sys.argv:
                check(source.parent.parent.name)
