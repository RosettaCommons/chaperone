"""Opt-in paid GPU smoke for the pinned AtomWorks 2 Foundry image.

Requires a Modal volume containing the public RFD3 and RF3 checkpoints.
Run with CHAPERONE_RUN_MODAL_FOUNDRY=1; optionally set
CHAPERONE_MODAL_FOUNDRY_VOLUME (default: chaperone-foundry-checkpoints).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from chaperone import ExecutionEngine, PipelineDataHandler, get_settings
from chaperone.execution_tracking import load_execution_records
from chaperone.io import write_pdb


@pytest.mark.cloud
@pytest.mark.skipif(
    os.environ.get("CHAPERONE_RUN_MODAL_FOUNDRY") != "1",
    reason="Opt-in paid Foundry image build and GPU inference",
)
@pytest.mark.parametrize("model", ["rfd3", "rf3"])
def test_live_modal_foundry(tmp_path, monkeypatch, model):
    from blocks.networks.foundry.container import _FOUNDRY_REVISION
    from blocks.networks.foundry.rfdiffusion3 import RFD3Block
    from blocks.networks.prediction.rosettafold3 import RosettaFold3Block

    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "CHAPERONE_MODAL_VOLUMES",
        {
            "/weights/foundry": os.environ.get(
                "CHAPERONE_MODAL_FOUNDRY_VOLUME", "chaperone-foundry-checkpoints"
            )
        },
    )
    monkeypatch.setattr(settings, "FOUNDRY_CHECKPOINT_DIRS", "/weights/foundry")

    class RuntimeProbe:
        def create_command(self, handler, sandbox_dir, chunk_id):
            command = super().create_command(handler, sandbox_dir, chunk_id)
            probe = (
                "import torch, json, subprocess; from pathlib import Path; "
                "from importlib.metadata import version; "
                "assert torch.cuda.is_available(); "
                "assert torch.ones(1, device='cuda').sum().item() == 1; "
                "data=dict(gpu=torch.cuda.get_device_name(), "
                "torch=version('torch'), atomworks=version('atomworks'), "
                "biotite=version('biotite'), "
                "foundry_revision=subprocess.check_output(['git','-C','/opt/foundry','rev-parse','HEAD'],text=True).strip()); "
                f"Path({str(Path(sandbox_dir) / 'runtime.json')!r}).write_text(json.dumps(data))"
            )
            return ["/opt/foundry-env/bin/python", "-c", probe, "&&", *command]

    class CheckedRFD3(RuntimeProbe, RFD3Block):
        pass

    class CheckedRF3(RuntimeProbe, RosettaFold3Block):
        pass

    data = None
    if model == "rfd3":
        block = CheckedRFD3(
            inputs={"smoke": {"length": 32}},
            batch_size=1,
            n_batches=1,
            ckpt_path="/weights/foundry/rfd3_latest.ckpt",
        )
    else:
        source = (
            Path(__file__).resolve().parents[1]
            / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
        )
        reference = next(
            PipelineDataHandler.from_paths([source], name="reference").get_data(
                load=True
            )
        )
        atoms = reference.atom_array
        atoms = atoms[atoms.chain_id == atoms.chain_id[0]]
        residues = list(dict.fromkeys(atoms.res_id))[:32]
        assert len(residues) == 32
        path = tmp_path / "input32.pdb"
        write_pdb(atoms[np.isin(atoms.res_id, residues)], path)
        data = PipelineDataHandler.from_paths([path], name="reference")
        block = CheckedRF3(
            input_mode="sequence",
            diffusion_batch_size=1,
            num_diffusion_steps=50,
            n_recycles=10,
            early_stopping_plddt_threshold=0,
            ckpt_path="/weights/foundry/rf3_foundry_01_24_latest_remapped.ckpt",
        )
    engine = ExecutionEngine(
        tmp_path / "run",
        run_args_default={"mode": "modal", "max_retries": 0, "max_workers": 1},
    )
    result = engine.run(block, data=data)
    assert len(result) == 1
    output = next(result.get_data(which="last", load=True))
    assert len(output.atom_array) > 0
    assert np.isfinite(output.atom_array.coord).all()
    assert np.ptp(output.atom_array.coord, axis=0).max() < 1000
    assert sum(map(len, output.sequence.values())) == 32
    runtime = json.loads(next((tmp_path / "run").rglob("runtime.json")).read_text())
    assert runtime["atomworks"] == "2.2.1"
    assert runtime["biotite"] == "1.4.0"
    assert runtime["foundry_revision"] == _FOUNDRY_REVISION
    assert len(engine.run(block, data=data)) == 1
    (record,) = load_execution_records(tmp_path / "run" / model)
    assert record.status == "succeeded"
    assert record.backend == "modal" and record.allocated_gpus == 1
    print(
        json.dumps(
            {"model": model, "runtime": runtime, "gpu_seconds": record.gpu_seconds}
        )
    )
    print(result.unpack())
