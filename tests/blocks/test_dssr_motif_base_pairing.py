"""Unit tests for the dssr_motif_base_pairing block.

Like the dssr_snap_metrics tests beside them, these exercise the parsers on
hand-crafted JSON rather than invoking x3dna-dssr, and they lock the boundary
between the two blocks: the motif-pairing block must never acquire SNAP
configuration, and the older block's 11-column output contract must not move.

Run:
    pytest tests/blocks/test_dssr_motif_base_pairing.py -v
"""

from __future__ import annotations


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
                    "1..B.C.3.",
                    name="--",
                    saenger="n/a",
                    lw="tHW",
                    dssr="tM-W",
                ),
                _dssr_pair(
                    24,
                    "1..A.A.18.",
                    "1..B.C.3.",
                    name="--",
                    saenger="n/a",
                    lw="cWW",
                    dssr="cW-W",
                ),
            ]
        }
    )

    assert len(pairs) == 2
    assert pairs[0]["nt2"]["residue_id"] == "B3"
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
                    "1..B.C.3.",
                    name="--",
                    saenger="n/a",
                    lw="tHW",
                    dssr="tM-W",
                ),
                _dssr_pair(
                    24,
                    "1..A.A.18.",
                    "1..B.C.3.",
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
                    "1..A.A.12.",
                    "1..A.C.5.",
                    name="--",
                    saenger="n/a",
                    lw="cWW",
                    dssr="cW-W",
                ),
                _dssr_pair(
                    5,
                    "1..A.G.20.",
                    "1..A.C.5.",
                    name="--",
                    saenger="n/a",
                    lw="tHW",
                    dssr="tM-W",
                ),
            ]
        }
    )

    metrics = motif_base_pairing_metrics(predicted, native, {"B3": "A5"}, ["B3"])

    assert metrics["paired_B3"] == 1
    assert metrics["n_partners_B3"] == 2
    assert metrics["partner_native_type_B3"] == 1
    assert metrics["native_lw_match_B3"] == 1
    assert metrics["native_classification_match_B3"] == 1
    assert len(metrics["pair_details"]["B3"]) == 2
    assert len(metrics["native_pair_details"]["B3"]) == 2
    classifications = json.loads(metrics["pair_classifications_B3"])
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
