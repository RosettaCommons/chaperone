"""Consolidated functional tests for structure functions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from blocks.metrics.hbond import HBondBlock
import numpy as np
from biotite.structure import AtomArray
from blocks.utils.ligand_bias import ligand_proximity_bias
import pytest
from blocks.util_blocks.mutate_residues import mutate_residues
from pathlib import Path
from atomworks.enums import ChainType
from blocks.util_blocks.remap_ligand import (
    _rewrite_cif_copy,
    _select_ligand_atoms,
)
from pydantic import ValidationError
from blocks.util_blocks.repack_sidechains import (
    SelectiveSidechainRepackBlock,
    protein_positions_in_clashes,
)
import biotite.structure as struc
from blocks.util_blocks.restore_sidechains import restore_selected_sidechains
from blocks.util_blocks.rotamer_sweep import (
    SidechainRotamerSweepBlock,
    enumerate_rotamers,
    rotamer_design_id,
    uniform_chi_grid,
)
from blocks.metrics.secondary_structure import _compute_ss_metrics


# test_hbond_path.py


def test_hbond_block_propagates_setup_executable_to_worker(tmp_path):
    hbplus = tmp_path / "hbplus"
    hbplus.write_text("#!/bin/sh\nexit 0\n")
    hbplus.chmod(0o755)
    block = HBondBlock(hbplus_path=str(hbplus))
    context = block.setup(None, tmp_path, 0)
    loaded = SimpleNamespace(atom_array=object(), path="design.cif")

    with patch(
        "blocks.utils.hbonds.calculate_hbonds",
        return_value=(object(), [], None),
    ) as calculate:
        metrics = block.compute_metrics(loaded, context=context)

    assert metrics == {"num_hbonds": 0}
    assert calculate.call_args.kwargs["hbplus_path"] == str(hbplus)


# test_ligand_bias.py


def _complex() -> AtomArray:
    atoms = AtomArray(7)
    atoms.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.5, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    atoms.chain_id = np.asarray(["A", "A", "A", "A", "A", "L", "L"])
    atoms.res_id = np.asarray([1, 1, 2, 2, 3, 1, 1])
    atoms.res_name = np.asarray(["ALA", "ALA", "GLY", "GLY", "SER", "ADN", "ADN"])
    atoms.atom_name = np.asarray(["CA", "CB", "CA", "N", "CA", "O2*", "N1"])
    atoms.element = np.asarray(["C", "C", "C", "N", "C", "O", "N"])
    atoms.hetero = np.asarray([False, False, False, False, False, True, True])
    return atoms


def test_ligand_proximity_bias_distinguishes_sugar_polar_and_pocket_positions():
    bias = ligand_proximity_bias(
        _complex(),
        ligand_resnames=["ADN"],
        pocket_radius=4.0,
        polar_radius=2.0,
    )

    assert set(bias) == {"A1", "A2"}
    assert bias["A1"]["N"] > 0
    assert bias["A1"]["Y"] > bias["A2"]["Y"]
    assert bias["A2"]["F"] > 0


def test_position_bias_adds_soft_motif_preferences_without_replacing_proximity():
    baseline = ligand_proximity_bias(
        _complex(), ligand_resnames=["ADN"], pocket_radius=4.0, polar_radius=2.0
    )
    biased = ligand_proximity_bias(
        _complex(),
        ligand_resnames=["ADN"],
        pocket_radius=4.0,
        polar_radius=2.0,
        position_bias={"A1": {"F": 2.5}, "A3": {"N": 1.25}},
    )

    assert biased["A1"]["F"] == baseline["A1"]["F"] + 2.5
    assert biased["A1"]["N"] == baseline["A1"]["N"]
    assert biased["A3"] == {"N": 1.25}


def test_position_bias_rejects_missing_residues():
    import pytest

    with pytest.raises(ValueError, match="missing residue"):
        ligand_proximity_bias(
            _complex(),
            ligand_resnames=["ADN"],
            position_bias={"A99": {"F": 2.0}},
        )


# test_mutate_residues.py


def _mutation_array():
    array = AtomArray(8)
    array.chain_id = np.array(["A"] * 7 + ["L"])
    array.res_id = np.array([73] * 5 + [74] * 2 + [1])
    array.res_name = np.array(["VAL"] * 5 + ["ALA"] * 2 + ["ADN"])
    array.atom_name = np.array(["N", "CA", "C", "CB", "CG1", "N", "CA", "C1"])
    array.element = np.array(["N", "C", "C", "C", "C", "N", "C", "C"])
    array.hetero = np.array([False] * 7 + [True])
    array.coord = np.arange(24, dtype=float).reshape(8, 3)
    return array


def test_mutate_residues_preserves_backbone_and_strips_old_sidechain():
    source = _mutation_array()
    mutated, changes = mutate_residues(source, {"A73": "PHE"})

    target = (mutated.chain_id == "A") & (mutated.res_id == 73)
    assert set(mutated.atom_name[target]) == {"N", "CA", "C", "CB"}
    assert set(mutated.res_name[target]) == {"PHE"}
    assert mutated[(mutated.chain_id == "L")].res_name.tolist() == ["ADN"]
    assert changes == [{"position": "A73", "from": "VAL", "to": "PHE"}]
    assert set(source.res_name[source.res_id == 73]) == {"VAL"}


def test_mutate_residues_rejects_missing_or_nonstandard_targets():
    with pytest.raises(KeyError, match="A99"):
        mutate_residues(_mutation_array(), {"A99": "PHE"})
    with pytest.raises(ValueError, match="standard three-letter"):
        mutate_residues(_mutation_array(), {"A73": "MSE"})


# test_remap_ligand.py


def test_rewrite_cif_copy_applies_overlapping_atom_names_simultaneously(tmp_path: Path):
    source = tmp_path / "source.cif"
    output = tmp_path / "output.cif"
    source.write_text("ATOM C1 LIG\nATOM C2 LIG\nATOM C3 LIG\n")

    _rewrite_cif_copy(
        str(source),
        str(output),
        {"C1": "C2", "C2": "C3", "C3": "C1", "LIG": "A;N"},
    )

    assert output.read_text() == "ATOM C2 A;N\nATOM C3 A;N\nATOM C1 A;N\n"


def test_select_ligand_prefers_hetero_for_pdb_without_entity_metadata():
    array = AtomArray(3)
    array.element = np.array(["C", "N", "O"])
    array.hetero = np.array([False, False, True])
    array.set_annotation(
        "chain_type",
        np.array([ChainType.NON_POLYMER, ChainType.NON_POLYMER, ChainType.NON_POLYMER]),
    )

    ligand = _select_ligand_atoms(array)

    assert len(ligand) == 1
    assert ligand.element.tolist() == ["O"]


# test_repack_sidechains.py


def test_selective_repack_validates_positions_and_rejects_duplicates():
    block = SelectiveSidechainRepackBlock(positions=["A42", "B:7"])

    assert block.positions == ["A42", "B:7"]

    with pytest.raises(ValidationError, match="expected A42"):
        SelectiveSidechainRepackBlock(positions=["not-a-position"])
    with pytest.raises(ValidationError, match="must be unique"):
        SelectiveSidechainRepackBlock(positions=["A42", "A42"])

    with pytest.raises(ValidationError, match="provide positions or clash_distance"):
        SelectiveSidechainRepackBlock()

    assert SelectiveSidechainRepackBlock(clash_distance=2.0).positions == []


def test_protein_positions_in_clashes_excludes_adjacent_residues():
    atoms = AtomArray(4)
    atoms.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    atoms.chain_id = np.asarray(["A", "A", "A", "A"])
    atoms.res_id = np.asarray([1, 2, 8, 20])
    atoms.res_name = np.asarray(["ALA", "ALA", "LEU", "SER"])
    atoms.atom_name = np.asarray(["CB", "CB", "CG", "OG"])
    atoms.element = np.asarray(["C", "C", "C", "O"])
    atoms.hetero = np.asarray([False, False, False, False])

    assert protein_positions_in_clashes(atoms, distance=1.0) == ["A1", "A8"]


# test_restore_sidechains.py


def _sidechain_array(sidechain_x: float, residue_name: str = "PHE"):
    atoms = struc.AtomArray(6)
    atoms.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [sidechain_x, 1.0, 0.0],
            [sidechain_x, 2.0, 0.0],
        ]
    )
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 42
    atoms.res_name[:] = residue_name
    atoms.atom_name[:] = ["N", "CA", "C", "O", "CB", "CG"]
    atoms.element[:] = ["N", "C", "C", "O", "C", "C"]
    atoms.hetero[:] = False
    return atoms


def test_restore_selected_sidechains_preserves_designed_backbone():
    reference = _sidechain_array(5.0)
    designed = _sidechain_array(9.0)
    designed.coord[:4, 1] = 3.0

    restored, n_residues, n_atoms, skipped = restore_selected_sidechains(
        reference, designed, ["A42"]
    )

    assert np.array_equal(restored.coord[:4], designed.coord[:4])
    assert np.array_equal(restored.coord[4:], reference.coord[4:])
    assert (n_residues, n_atoms, skipped) == (1, 2, [])


def test_restore_selected_sidechains_skips_sequence_mismatch():
    restored, n_residues, n_atoms, skipped = restore_selected_sidechains(
        _sidechain_array(5.0, "PHE"), _sidechain_array(9.0, "TYR"), ["A:42"]
    )

    assert np.array_equal(restored.coord, _sidechain_array(9.0, "TYR").coord)
    assert (n_residues, n_atoms, skipped) == (0, 0, ["A:42"])


# test_rotamer_sweep.py


def test_enumerate_rotamers_is_deterministic_cartesian_product():
    combinations = enumerate_rotamers(
        {
            "A10": [[-60.0, 90.0], [60.0, -90.0]],
            "A20": [[180.0, 90.0], [180.0, -90.0]],
        }
    )

    assert combinations == [
        {"A10": [-60.0, 90.0], "A20": [180.0, 90.0]},
        {"A10": [-60.0, 90.0], "A20": [180.0, -90.0]},
        {"A10": [60.0, -90.0], "A20": [180.0, 90.0]},
        {"A10": [60.0, -90.0], "A20": [180.0, -90.0]},
    ]


def test_rotamer_sweep_requires_matching_positions_and_honors_cap():
    with pytest.raises(ValueError, match="identical positions"):
        SidechainRotamerSweepBlock(
            mutations={"A10": "PHE"},
            rotamers={"A11": [[60.0, 90.0]]},
        )
    with pytest.raises(ValueError, match="exceeding max_combinations"):
        SidechainRotamerSweepBlock(
            mutations={"A10": "PHE", "A20": "PHE"},
            rotamers={
                "A10": [[60.0, 90.0], [-60.0, 90.0]],
                "A20": [[60.0, 90.0], [-60.0, 90.0]],
            },
            max_combinations=3,
        )


def test_uniform_chi_grid_supports_functional_sidechain_sweeps():
    assert uniform_chi_grid([120.0]) == [[-180.0], [-60.0], [60.0]]
    assert uniform_chi_grid([180.0, 180.0]) == [
        [-180.0, -180.0],
        [-180.0, 0.0],
        [0.0, -180.0],
        [0.0, 0.0],
    ]
    with pytest.raises(ValueError, match="between 0 and 360"):
        uniform_chi_grid([0.0])


def test_rotamer_design_id_is_stable_and_arm_specific():
    assert rotamer_design_id("parent", "/tmp/input.cif", "a32_asn", 7) == (
        "parent__a32_asn_0007"
    )
    assert rotamer_design_id(None, "/tmp/input.cif", "a33_asn", 7) == (
        "input__a33_asn_0007"
    )


# test_secondary_structure.py


def test_secondary_structure_accepts_missing_b_factor_annotation():
    atom_names = ["N", "CA", "C", "O"] * 5
    array = AtomArray(len(atom_names))
    array.chain_id = np.array(["A"] * len(array))
    array.res_id = np.repeat(np.arange(1, 6), 4)
    array.res_name = np.array(["GLY"] * len(array))
    array.atom_name = np.array(atom_names)
    array.element = np.array(["N", "C", "C", "O"] * 5)
    array.hetero = np.array([False] * len(array))
    array.coord = np.array(
        [
            [residue * 3.8 + offset, 0.1 * offset, 0.0]
            for residue in range(5)
            for offset in (0.0, 1.3, 2.5, 3.2)
        ]
    )

    assert "b_factor" not in array.get_annotation_categories()
    metrics = _compute_ss_metrics(
        array,
        pocket_cutoff=8.0,
        plddt_threshold=70.0,
        plddt_min_signal=50.0,
        verbose=False,
    )

    assert metrics["n_residues"] == 5
    assert metrics["mean_plddt"] is None
    assert metrics["fraction_low_plddt"] is None


# test_dssr_motif_base_pairing.py


def _dssr_pair(index, nt1, nt2, *, name, saenger, lw, dssr):
    return {
        "index": index,
        "nt1": nt1,
        "nt2": nt2,
        "bp": "A-C",
        "name": name,
        "Saenger": saenger,
        "LW": lw,
        "DSSR": dssr,
    }


def test_dssr_pair_parser_preserves_multiple_partners_and_classifications():
    from blocks.metrics.dssr_snap_metrics import parse_dssr_pairs_data

    pairs = parse_dssr_pairs_data(
        {
            "pairs": [
                _dssr_pair(
                    23,
                    "1..A.A.17.",
                    "1..C.C.8.",
                    name="--",
                    saenger="n/a",
                    lw="tHW",
                    dssr="tM-W",
                ),
                _dssr_pair(
                    24,
                    "1..A.A.18.",
                    "1..C.C.8.",
                    name="--",
                    saenger="n/a",
                    lw="cWW",
                    dssr="cW-W",
                ),
            ]
        }
    )

    assert len(pairs) == 2
    assert pairs[0]["nt2"]["residue_id"] == "C8"
    assert pairs[0]["nt1"]["base"] == "A"
    classifications = (
        pairs[0]["name"],
        pairs[0]["Saenger"],
        pairs[0]["LW"],
        pairs[0]["DSSR"],
    )
    assert classifications == ("--", "n/a", "tHW", "tM-W")


def test_dssr_motif_metric_keeps_all_native_partners():
    import json

    from blocks.metrics.dssr_snap_metrics import (
        motif_base_pairing_metrics,
        parse_dssr_pairs_data,
    )

    native = parse_dssr_pairs_data(
        {
            "pairs": [
                _dssr_pair(
                    23,
                    "1..A.A.17.",
                    "1..C.C.8.",
                    name="--",
                    saenger="n/a",
                    lw="tHW",
                    dssr="tM-W",
                ),
                _dssr_pair(
                    24,
                    "1..A.A.18.",
                    "1..C.C.8.",
                    name="--",
                    saenger="n/a",
                    lw="cWW",
                    dssr="cW-W",
                ),
            ]
        }
    )
    predicted = parse_dssr_pairs_data(
        {
            "pairs": [
                _dssr_pair(
                    4,
                    "1..D.A.12.",
                    "1..D.C.5.",
                    name="--",
                    saenger="n/a",
                    lw="cWW",
                    dssr="cW-W",
                ),
                _dssr_pair(
                    5,
                    "1..D.G.20.",
                    "1..D.C.5.",
                    name="--",
                    saenger="n/a",
                    lw="tHW",
                    dssr="tM-W",
                ),
            ]
        }
    )

    metrics = motif_base_pairing_metrics(predicted, native, {"C8": "D5"}, ["C8"])

    assert metrics["paired_C8"] == 1
    assert metrics["n_partners_C8"] == 2
    assert metrics["partner_native_type_C8"] == 1
    assert metrics["native_lw_match_C8"] == 1
    assert metrics["native_classification_match_C8"] == 1
    assert len(metrics["pair_details"]["C8"]) == 2
    assert len(metrics["native_pair_details"]["C8"]) == 2
    classifications = json.loads(metrics["pair_classifications_C8"])
    assert len(classifications) == 2
    assert set(classifications[0]) == {
        "partner_id",
        "partner_base",
        "name",
        "Saenger",
        "LW",
        "DSSR",
    }


def test_dssr_snap_parser_retains_exact_11_column_contract(tmp_path):
    import json

    from blocks.metrics.dssr_snap_metrics import (
        METRIC_LABELS,
        parse_dssr_snap_outputs,
    )

    dssr_json = tmp_path / "dssr.json"
    dssr_json.write_text(json.dumps({"hbonds": [], "pairs": []}))
    snap_txt = tmp_path / "snap.txt"
    snap_txt.write_text(
        "  3 nucleotide/amino-acid interactions\n"
        "  2 base-pair/amino-acid interactions\n"
        "  1 base/amino-acid pseudo pairs\n"
        "  4 base/amino-acid pseudo stacks\n"
    )

    metrics = parse_dssr_snap_outputs(dssr_json, snap_txt)

    assert tuple(metrics) == METRIC_LABELS
    assert metrics == {
        "Total_hbond": 0,
        "Major_hbond": 0,
        "Minor_hbond": 0,
        "Backbone_hbond": 0,
        "hbond_support": 0,
        "hbond_support_no_alpha": 0,
        "hbond_support_no_backbone": 0,
        "nt_aa": 3,
        "bp_aa": 2,
        "pseudo_pairs": 1,
        "pseudo_stacks": 4,
    }


def test_dssr_motif_block_is_separate_and_has_no_snap_configuration():
    from blocks.metrics.dssr_snap_metrics import (
        DSSRMotifBasePairingBlock,
        DSSRSnapMetricsBlock,
    )

    assert DSSRMotifBasePairingBlock._name == "dssr_motif_base_pairing"
    assert "x3dna_snap_bin" not in DSSRMotifBasePairingBlock.model_fields
    assert DSSRSnapMetricsBlock._name == "dssr_snap_metrics"
    assert DSSRSnapMetricsBlock.model_fields["skip_snap"].default is False
    assert "x3dna_snap_bin" in DSSRSnapMetricsBlock.model_fields
