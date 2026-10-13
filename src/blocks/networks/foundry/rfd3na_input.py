"""Build RFD3NA input JSON for ligand-conditioned aptamer design.

Given a structure file containing a small molecule, emit the RFD3NA spec that
asks for a nucleic-acid chain designed around it. Mirrors the released
``AMP_aptamer`` example in ``models/rfd3na/docs/examples/atom23_design.json``.

This is a helper module rather than a block: RFD3NA consumes ``inputs=<path to
json>``, so the spec is built once in the pipeline script before ``engine.run``,
the same way ``ori_sweep.py`` builds sweep JSONs.

    from blocks.networks.foundry.rfd3na_input import build_ligand_spec, write_rfd3na_json

    spec = build_ligand_spec("AMP.pdb", "AMP", nucleic="RNA", length="40-60")
    path = write_rfd3na_json({"amp_aptamer": spec}, "rfd3na_input.json")
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# N/O bearing at least one H donate, any N/O accepts. The notebook's
# rfd_input_json.py wrote "[#7H,#8H]", which in SMARTS means *exactly* one H and
# so silently skips NH2 groups (e.g. adenine N6) despite its comment saying "at
# least one H"; !H0 is what was intended.
HBOND_DONOR_SMARTS = "[#7!H0,#8!H0]"
HBOND_ACCEPTOR_SMARTS = "[#7,#8]"

_NUCLEIC_SUFFIX = {"RNA": "R", "DNA": "D", "protein": ""}


def detect_hbond_atoms(structure, ligand: str, ref_smiles: str):
    """Detect H-bond donor/acceptor atom names on a ligand.

    Bond orders cannot be read from a structure file, so a reference SMILES is
    required to perceive them; without correct bond orders the implicit-hydrogen
    counts that distinguish donors from acceptors are meaningless.

    Returns:
        (donors, acceptors) as lists of atom names, or (None, None) if
        perception failed - callers should then omit the hbond keys entirely,
        which is what the released ``AMP_aptamer_noh`` spec does.
    """
    from rdkit import Chem

    from pipelines.io import parse_to_atom_array
    from blocks.utils.ligand_mol import (
        ligand_heavy_atoms,
        map_ref_to_names,
        mol_from_ligand,
    )

    names, elements, coords = ligand_heavy_atoms(
        parse_to_atom_array(str(structure)), ligand
    )
    ref = Chem.MolFromSmiles(ref_smiles)
    if ref is None:
        logger.warning("Could not parse ref_smiles %r", ref_smiles)
        return None, None

    mol = mol_from_ligand(elements, coords, ref)
    if mol is None:
        return None, None

    # Perceive on the reference, which has correct hydrogen counts, then map
    # indices back to structure atom names.
    ref_to_name = map_ref_to_names(mol, ref, names)
    if ref_to_name is None:
        return None, None

    def _match(smarts):
        patt = Chem.MolFromSmarts(smarts)
        hits = []
        for match in ref.GetSubstructMatches(patt):
            name = ref_to_name.get(match[0])
            if name is not None and name not in hits:
                hits.append(name)
        return hits

    return _match(HBOND_DONOR_SMARTS), _match(HBOND_ACCEPTOR_SMARTS)


def build_ligand_spec(
    structure,
    ligand: str,
    *,
    nucleic: str = "RNA",
    length: str = "40-60",
    ori_jitter: float = 1,
    ref_smiles: Optional[str] = None,
    hbond_donors: Optional[list[str]] = None,
    hbond_acceptors: Optional[list[str]] = None,
) -> dict:
    """Build a single RFD3NA spec designing a nucleic-acid chain around a ligand.

    Args:
        structure: Path to a PDB/CIF containing the ligand.
        ligand: Ligand residue name. Non-CCD names using the semicolon
            convention (e.g. "A;P") work unchanged.
        nucleic: "RNA", "DNA" or "protein" - sets the contig suffix.
        length: Design length range, e.g. "40-60".
        ori_jitter: Jitter applied to the origin token.
        ref_smiles: Reference SMILES, enabling H-bond auto-detection.
        hbond_donors: Explicit donor atom names, overriding auto-detection.
        hbond_acceptors: Explicit acceptor atom names.

    Returns:
        RFD3NA spec dict. The ligand is left out of ``select_fixed_atoms``
        because RFD3 fixes non-protein components by default.
    """
    if nucleic not in _NUCLEIC_SUFFIX:
        raise ValueError(
            f"nucleic must be one of {sorted(_NUCLEIC_SUFFIX)}, got {nucleic!r}"
        )

    spec = {
        "input": os.path.abspath(str(structure)),
        "ligand": ligand,
        "contig": f"{length}{_NUCLEIC_SUFFIX[nucleic]}",
        "length": length,
        "ori_jitter": ori_jitter,
        "select_buried": {ligand: "ALL"},
    }

    if hbond_donors is None and hbond_acceptors is None and ref_smiles:
        hbond_donors, hbond_acceptors = detect_hbond_atoms(
            structure, ligand, ref_smiles
        )

    if hbond_donors:
        spec["select_hbond_donor"] = {ligand: ",".join(hbond_donors)}
    if hbond_acceptors:
        spec["select_hbond_acceptor"] = {ligand: ",".join(hbond_acceptors)}

    return spec


def build_motif_spec(
    structure,
    unindex: list[str],
    *,
    nucleic: str = "RNA",
    length: str = "60-60",
    ligand: Optional[str] = None,
    fixed_atoms: str | dict = "ALL",
) -> dict:
    """Build a spec that rescaffolds an unindexed motif into one designed chain.

    Mirrors the released ``unindexed_rnasep`` example: motif residues are handed
    to the model without their sequence positions, so RFD3NA is free to place
    them anywhere in the new chain. The design length is the contig length -
    unindexed residues occupy positions within it rather than adding to it.

    Args:
        structure: Path to the structure holding the motif.
        unindex: Motif components as residue-id groups, e.g.
            ``["A10-12", "A24", "B4-5"]``. Residues inside a group are tied in
            sequence; separate groups are independent. Groups may come from
            different chains of the input even though the design is one chain.
        nucleic: "RNA", "DNA" or "protein" - sets the contig suffix.
        length: Design length, e.g. "60-60" for a fixed length.
        ligand: Ligand/ion residue names to carry through, e.g. "MN".
        fixed_atoms: Atoms held fixed. A string applies to every group; a dict
            keyed by residue id (or by group) overrides per residue. "ALL" fixes
            whole residues; the ``BKBN``/``TIP`` shorthands do NOT apply to
            nucleic acids, so pass explicit atom names instead when subsetting.
            Atoms left unfixed are not copied from the input at all - they are
            diffused, which is what lets the model weld a group into the chain.

    Returns:
        RFD3NA spec dict.
    """
    if nucleic not in _NUCLEIC_SUFFIX:
        raise ValueError(
            f"nucleic must be one of {sorted(_NUCLEIC_SUFFIX)}, got {nucleic!r}"
        )

    # Groups are joined with a plain comma, never "/0". In `unindex` a "/0" is
    # not merely a separator: foundry's accumulate_components advances the
    # guidepost chain id AND increments molecule_id on it, so the motif
    # fragments are handed to the model as separate molecules. Independence is
    # already carried by the breakpoint flags that
    # get_motif_components_and_breaks derives from the commas, so for a
    # single-chain design "/0" only tells the model not to bond the fragments in.
    if any("/0" in group for group in unindex):
        raise ValueError(
            "unindex groups must not contain '/0'; pass them as separate list "
            "entries. See the note in build_motif_spec."
        )

    if isinstance(fixed_atoms, str):
        fixed = {group: fixed_atoms for group in unindex}
    else:
        fixed = dict(fixed_atoms)

    spec = {
        "input": os.path.abspath(str(structure)),
        "contig": f"{length}{_NUCLEIC_SUFFIX[nucleic]}",
        "length": length,
        "unindex": ",".join(unindex),
        "select_fixed_atoms": fixed,
    }
    if ligand:
        spec["ligand"] = ligand
    return spec


def write_rfd3na_json(specs: dict, out_path) -> str:
    """Write specs to JSON and return the absolute path.

    Spec keys become RFD3NA task names, so they must match ``[A-Za-z0-9_-]+``.
    """
    out_path = Path(out_path).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(specs, indent=2))
    logger.info("Wrote %d RFD3NA spec(s) to %s", len(specs), out_path)
    return str(out_path)
