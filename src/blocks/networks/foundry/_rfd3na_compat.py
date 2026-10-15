"""Guarded input-parser repair for the pinned Foundry image.

The upstream builder already understands R/D contig suffixes, but its validator
rejects them without a motif input. Only wholly generated contigs may omit that
input. No model, sampler, or input-backed selection behavior is changed.
This file also runs standalone inside the image during its build.
"""

from hashlib import sha256
from pathlib import Path
import sys


_INPUT_OLD = """                if selections:
                    raise ValueError(
"""
_INPUT_NEW = """                # Chaperone: generated-only contigs have no input motif atoms.
                if selections == ["contig"] and isinstance(data["contig"], str):
                    import re

                    parts = [part.strip() for part in data["contig"].split(",")]
                    if (
                        any(part != "/0" for part in parts)
                        and all(re.fullmatch(r"(?:[0-9]+(?:-[0-9]+)?[RD]?|/0)", part)
                                for part in parts)
                    ):
                        data["contig"] = InputSelection(
                            raw=",".join(parts), data={},
                            mask=np.zeros(0, dtype=bool), tokens={},
                        )
                        return data
                if selections:
                    raise ValueError(
"""
_TOKENS_OLD = """    def get_tokens(self, aa):
        _, _, tokens = from_any_(v=self.raw, atom_array=aa)
"""
_TOKENS_NEW = """    def get_tokens(self, aa):
        if aa is None:
            if self.data:
                raise ValueError("Input atoms are required for selected motif tokens")
            return {}
        _, _, tokens = from_any_(v=self.raw, atom_array=aa)
"""
_PATCHES = (
    (
        "input_parsing.py",
        "97a52db6722dbb76e3ac20d2efaf7f2082c5b4ecfd536bce81f79f60c305ba3b",
        _INPUT_OLD,
        _INPUT_NEW,
    ),
    (
        "parsing.py",
        "4e54938157cfb2db31bcc3b30c6ae51ecf471bfaed30793e6fccebcc422a4678",
        _TOKENS_OLD,
        _TOKENS_NEW,
    ),
)


def patch(source: Path) -> None:
    """Validate both source files before writing; accept our exact repair again."""
    pending = []
    for filename, expected, old, new in _PATCHES:
        path = source / filename
        content = path.read_text()
        original = content.replace(new, old) if content.count(new) == 1 else content
        if (
            sha256(original.encode()).hexdigest() != expected
            or original.count(old) != 1
        ):
            raise RuntimeError(f"Unexpected pinned Foundry source: {path}")
        pending.append((path, original.replace(old, new)))
    for path, content in pending:
        path.write_text(content)


def check_runtime() -> None:
    """Exercise the actual pinned parser, including its downstream atom builder."""
    import numpy as np
    from biotite.structure import filter_amino_acids, get_residue_starts
    from rfd3na._core.utils.components import ComponentValidationError
    from pydantic import ValidationError
    from rfd3na.inference.input_parsing import DesignInputSpecification
    from rfd3na.inference.parsing import InputSelection

    for contig, length, counts in (
        ("4-4R", "4", (0, 4, 0)),
        ("3D", "3", (0, 0, 3)),
        ("2,/0,3R,/0,4D", "9", (2, 3, 4)),
    ):
        specification = DesignInputSpecification(contig=contig, length=length)
        assert specification.atom_array_input is None
        atoms = specification.build()
        residues = atoms[get_residue_starts(atoms)]
        actual = (
            int(filter_amino_acids(residues).sum()),
            int(np.isin(residues.res_name, ["A", "C", "G", "U"]).sum()),
            int(np.isin(residues.res_name, ["DA", "DC", "DG", "DT"]).sum()),
        )
        assert actual == counts, (contig, actual, counts)
        assert not atoms.is_motif_atom_with_fixed_coord.any()
        assert np.isfinite(atoms.coord).all()

    # A generated fixture tests selection retention, not physical RNA geometry.
    source = DesignInputSpecification(contig="4R", length="4").build()
    source.coord = np.arange(len(source) * 3, dtype=np.float32).reshape(-1, 3) * 0.01
    result = DesignInputSpecification(
        input=None,
        atom_array_input=source,
        ori_token=[0, 0, 0],
        contig="A1-2,2R",
        length="4",
    ).build()
    assert len(get_residue_starts(result)) == 4
    src = source[np.isin(source.res_id, [1, 2])]
    dst = result[np.isin(result.res_id, [1, 2])]
    assert list(zip(src.res_name, src.atom_name)) == list(
        zip(dst.res_name, dst.atom_name)
    )
    np.testing.assert_array_equal(src.coord, dst.coord)
    assert dst.is_motif_atom_with_fixed_coord.all()

    for arguments in (
        {"contig": "A1-3,4R"},
        {"contig": "4R", "select_hotspots": "A1"},
        {"contig": "4R", "unindex": "A1"},
        {"contig": "4X"},
        {"contig": "4R", "partial_t": 0.5},
    ):
        try:
            DesignInputSpecification(**arguments)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Accepted selections without input: {arguments}")
    try:
        DesignInputSpecification(contig="4R", length="5").build()
    except ComponentValidationError:
        pass
    else:
        raise AssertionError("Ignored the requested total length")
    selected = InputSelection(
        raw="A1", data={"A1": ["CA"]}, mask=np.ones(1, dtype=bool), tokens=None
    )
    try:
        selected.get_tokens(None)
    except ValueError:
        pass
    else:
        raise AssertionError("Silently dropped a selected motif")
    print("RFD3NA generated RNA/DNA/mixed contig and selection guards passed")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
    if "--check" in sys.argv[2:]:
        check_runtime()
