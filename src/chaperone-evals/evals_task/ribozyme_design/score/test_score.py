import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _load_scorer():
    path = Path(__file__).with_name("score.py")
    spec = importlib.util.spec_from_file_location("ribozyme_t6_scorer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Scorers initialize the shared absl CLI at import time. Avoid parsing
    # pytest's own flags; the tests only inspect the scorer configuration.
    with patch("chaperone.init_cli"):
        spec.loader.exec_module(module)
    return module


def test_t6_is_reported_per_residue_but_never_gated():
    scorer = _load_scorer()

    assert set(scorer.GATES) == {
        "motif_rmsd",
        "inline_attack_dist",
        "inline_attack_angle",
        "plddt",
        "novelty",
    }
    for reference_id in scorer.MOTIF_REFERENCE_IDS:
        assert f"t6.paired_{reference_id}" not in scorer.METRIC_COLUMNS
        assert f"t6.paired_{reference_id}" in scorer.T6_REPORT_COLUMNS
        assert (
            f"t6.native_classification_match_{reference_id}"
            in scorer.T6_REPORT_COLUMNS
        )
        assert (
            f"t6.pair_classifications_{reference_id}"
            in scorer.T6_DIAGNOSTIC_COLUMNS
        )


def test_post_fold_metrics_do_not_reserve_a_gpu():
    scorer = _load_scorer()

    assert scorer.GPU_ARGS["partition"] == "gpu"
    assert scorer.GPU_ARGS["gres"] == "gpu:small:1"
    assert scorer.GPU_ARGS["constraint"] == "A4000"
    assert scorer.METRIC_ARGS["partition"] == "cpu"
    assert "gres" not in scorer.METRIC_ARGS
    assert "constraint" not in scorer.METRIC_ARGS


def test_native_control_fixture_satisfies_the_public_contract():
    scorer = _load_scorer()
    from scoring import SequenceSubmissionSpec, load_sequence_submission

    fixture = scorer.SCRIPT_DIR / "fixtures" / "native_control.csv"
    loaded = load_sequence_submission(
        fixture,
        SequenceSubmissionSpec(
            required_n=1,
            chains=(scorer.DESIGN_CHAIN,),
            molecule_type="rna",
            sequence_length_range=scorer.LENGTH_RANGE,
            allow_motif_indices=False,
            motif_reference_ids=scorer.MOTIF_REFERENCE_IDS,
            motif_residue_types=scorer.MOTIF_RESIDUE_TYPES,
            motif_contiguity_groups=scorer.MOTIF_CONTIGUITY_GROUPS,
            motif_map_format="zero_based_indices",
            max_pairwise_identity=scorer.MAX_PAIRWISE_IDENTITY,
        ),
    )

    assert loaded["design_id"].to_list() == ["native_2oeu_merged"]


def test_heldout_assets_are_not_public_task_inputs():
    scorer = _load_scorer()

    public_files = {
        path.name for path in scorer.INPUTS_DIR.iterdir() if path.is_file()
    }
    assert public_files == {"2oeu_motif.pdb"}
    assert scorer.MOTIF.parent == scorer.INPUTS_DIR
    assert scorer.NATIVE.parent == scorer.ASSETS_DIR
    assert scorer.HOMOLOGS.parent == scorer.ASSETS_DIR
    assert scorer.NATIVE.is_file()
    assert scorer.HOMOLOGS.is_file()


def test_t6_classifications_are_in_final_report_metadata():
    scorer = _load_scorer()
    row = {
        "design_id": "rz0",
        "score_motif_rmsd.motif_rmsd": 0.8,
        "t6.native_pair_details": {
            "B6": np.array([{"LW": "tWH"}, {"LW": "cWW"}], dtype=object)
        },
    }
    for reference_id in scorer.MOTIF_REFERENCE_IDS:
        row[f"t6.paired_{reference_id}"] = 1
        row[f"t6.native_lw_match_{reference_id}"] = 1
        row[f"t6.pair_classifications_{reference_id}"] = json.dumps(
            [
                {
                    "partner_id": "A12",
                    "partner_base": "A",
                    "name": "--",
                    "Saenger": "n/a",
                    "LW": "cWW",
                    "DSSR": "cW-W",
                }
            ]
        )

    metadata = scorer._report_metadata(pd.DataFrame([row]))

    details = metadata["t6"]["pair_classifications_by_design"]["rz0"]
    assert metadata["motif_rmsd_only_pass"] == {
        "threshold": "score_motif_rmsd.motif_rmsd < 2.0",
        "n_pass": 1,
        "pass_rate": 1.0,
        "design_ids": ["rz0"],
    }
    assert details["B6"][0]["partner_id"] == "A12"
    assert details["B6"][0]["LW"] == "cWW"
    assert len(metadata["t6"]["native_pair_details"]["B6"]) == 2
    assert metadata["t6"]["status"] == "ok"
    assert metadata["t6"]["n_complete"] == 1
    assert metadata["t6"]["n_failed"] == 0
    json.dumps(metadata)


def test_t6_row_failure_is_reported_without_suppressing_primary_metadata():
    scorer = _load_scorer()
    successful = {
        "design_id": "rz-ok",
        "score_motif_rmsd.motif_rmsd": 0.9,
        "t6.status": "ok",
        "t6.error_type": "",
        "t6.error": "",
        "t6.native_pair_details": {"B6": np.array([{"LW": "cWW"}], dtype=object)},
    }
    failed = {
        "design_id": "rz-dssr-error",
        "score_motif_rmsd.motif_rmsd": 1.1,
        "t6.status": "error",
        "t6.error_type": "RuntimeError",
        "t6.error": "x3dna-dssr failed: rc=-11",
    }
    for reference_id in scorer.MOTIF_REFERENCE_IDS:
        successful[f"t6.paired_{reference_id}"] = 1
        successful[f"t6.native_lw_match_{reference_id}"] = 1
        successful[f"t6.pair_classifications_{reference_id}"] = "[]"

    metadata = scorer._report_metadata(pd.DataFrame([failed, successful]))

    assert metadata["motif_rmsd_only_pass"]["n_pass"] == 2
    assert metadata["t6"]["status"] == "partial"
    assert metadata["t6"]["n_complete"] == 1
    assert metadata["t6"]["n_failed"] == 1
    assert metadata["t6"]["failed_designs"] == [
        {
            "design_id": "rz-dssr-error",
            "error_type": "RuntimeError",
            "error": "x3dna-dssr failed: rc=-11",
        }
    ]
    assert set(metadata["t6"]["pair_classifications_by_design"]) == {"rz-ok"}
    json.dumps(metadata)
