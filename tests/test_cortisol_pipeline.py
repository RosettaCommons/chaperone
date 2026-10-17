"""Cortisol chemistry, state-specific gates, and bounded refinement control flow."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blocks import Blocks
from chaperone import PipelineDataHandler
from chaperone.data.schema import make_row

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cortisol-binder"


@pytest.fixture
def recipe(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    score_spec = importlib.util.spec_from_file_location("score", EXAMPLE / "score.py")
    scorer = importlib.util.module_from_spec(score_spec)
    monkeypatch.setitem(sys.modules, "score", scorer)
    score_spec.loader.exec_module(scorer)
    spec = importlib.util.spec_from_file_location(
        "cortisol_recipe", EXAMPLE / "pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handler(metrics, *, step="input", design_id="candidate"):
    return PipelineDataHandler(
        [
            make_row(
                name=step,
                path="/tmp/cortisol-test.cif",
                metrics=metrics,
                design_id=design_id,
            )
        ]
    )


@pytest.mark.parametrize(
    "final,pae,hbonds,plddt,expected",
    [
        (False, 2, 2, 20, True),
        (False, 2.001, 2, 99, False),
        (False, 1, 1, 99, False),
        (True, 1.25, 4, 90, True),
        (True, 1.251, 4, 99, False),
        (True, 1, 3, 99, False),
        (True, 1, 4, 89.999, False),
        (True, None, 4, 99, False),
        (True, 1, None, 99, False),
        (True, 1, 4, None, False),
        (True, np.nan, 4, 99, False),
        (True, 1, np.inf, 99, False),
    ],
)
def test_scorer_matches_engine_filter(recipe, final, pae, hbonds, plddt, expected):
    from score import score_frame

    af3 = "refine/cycle_1/af3"
    hb = af3 + "_hbond"
    h = handler(
        {af3: {"min_pae_interaction": pae, "plddt": plddt}, hb: {"num_hbonds": hbonds}}
    )
    frame, _ = score_frame(h.unpack(), af3, hb, final=final)
    assert bool(frame["pass"].iloc[0]) is expected
    result = Blocks["filter"](filters=recipe.gate_filters(af3, hb, final=final))(h)
    assert len(result) == int(expected)


def test_earlier_prediction_cannot_rescue_missing_current_metrics(recipe):
    from score import score_frame

    h = handler(
        {
            "refine/cycle_0/af3": {"min_pae_interaction": 1, "plddt": 99},
            "refine/cycle_0/af3_hbond": {"num_hbonds": 5},
        }
    )
    current = "refine/cycle_1/af3"
    frame, _ = score_frame(h.unpack(), current, current + "_hbond")
    assert not frame["pass"].any()
    assert (
        len(
            Blocks["filter"](filters=recipe.gate_filters(current, current + "_hbond"))(
                h
            )
        )
        == 0
    )


def test_cortisol_chemistry_and_conditioning(recipe):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    from blocks.networks.prediction.helpers import build_remap_reference
    from chaperone.io import parse_to_atom_array

    mol = Chem.MolFromSmiles(recipe.LIGAND_SMILES)
    assert rdMolDescriptors.CalcMolFormula(mol) == "C21H30O5"
    assert Chem.MolToInchiKey(mol) == "JYGXADMDTFJGBT-VWUMJDOOSA-N"
    atoms = parse_to_atom_array(recipe.FILES_DIR / "cortisol.cif")
    heavy = atoms[atoms.element != "H"]
    assert len(heavy) == 26 and np.isfinite(heavy.coord).all()
    assert set(heavy.res_name) == {"HCY"}
    config = json.loads((recipe.FILES_DIR / "cortisol.json").read_text())["cortisol"]
    assert config["ligand"] == "HCY"
    assert set(config["select_fixed_atoms"]["HCY"].split(",")) == set(heavy.atom_name)
    assert config["select_hbond_donor"]["HCY"] == "O2,O3,O5"
    reference = build_remap_reference(
        str(recipe.FILES_DIR / "cortisol.cif"), target_name="HCY"
    )
    assert reference["n_ref_atoms"] == 26
    assert Chem.MolToInchiKey(
        Chem.Mol(reference["ref_mol_bytes"])
    ) == Chem.MolToInchiKey(mol)
    block = recipe.sequence_block()
    assert block.aromatic_fix and block.hbond_fix
    assert (
        block.output_mode == "structures" and block.args.number_of_packs_per_design == 1
    )


class FakeEngine:
    """Replace expensive model execution, retaining the real filter and scorer."""

    def __init__(self, rundir, *, reject_cycle=None):
        self.rundir = rundir
        self.reject_cycle = reject_cycle
        self.cycle_index = None
        self.predictions = []
        self.inputs = []

    def cycle(self, name, n):
        for i in range(n):
            self.cycle_index = i
            yield self
        self.cycle_index = None

    def run(self, blocks, data=None):
        if isinstance(blocks, list):
            final = blocks[0].name == "af3_final"
            if not final:
                self.inputs.append(data)
            step = "af3_final" if final else f"refine/cycle_{self.cycle_index}/af3"
            self.predictions.append(step)
            pae = 3 if not final and self.cycle_index == self.reject_cycle else 1.25
            return handler(
                {
                    step: {"min_pae_interaction": pae, "plddt": 90},
                    step + "_hbond": {"num_hbonds": 4},
                },
                step=step,
            )
        if blocks._name == "filter":
            return blocks(data)
        assert blocks._name == "rfd3"
        return handler({})


def test_cycles_feed_surviving_prediction_then_final_oracle(recipe, tmp_path):
    engine = FakeEngine(tmp_path)
    result = recipe.run_pipeline(engine, cycles=2)
    assert len(result) == 1
    assert engine.predictions == [
        "refine/cycle_0/af3",
        "refine/cycle_1/af3",
        "af3_final",
    ]
    assert engine.inputs[1].names[0] == "refine/cycle_0/af3"
    assert (
        json.loads((tmp_path / "scores/final/report.json").read_text())["n_pass"] == 1
    )


@pytest.mark.parametrize("reject_cycle", [0, 1])
def test_failed_cycle_stops_without_final_prediction(recipe, tmp_path, reject_cycle):
    engine = FakeEngine(tmp_path, reject_cycle=reject_cycle)
    assert len(recipe.run_pipeline(engine, cycles=2)) == 0
    assert len(engine.predictions) == reject_cycle + 1
    assert "af3_final" not in engine.predictions
    results = pd.read_csv(tmp_path / "results.csv")
    assert results.empty
    assert {"design_id", "af3_final.min_pae_interaction", "af3_final.plddt"} <= set(
        results
    )
