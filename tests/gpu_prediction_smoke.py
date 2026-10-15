"""Bounded runtime check using 48 residues from the bundled insulin-receptor input.

Inside the allocated GPU shell:
  SMOKE_MODEL=simplefold SMOKE_RUNDIR=/.../simplefold-smoke \
    CHAPERONE_REPO=/.../repo /.../repo/.venv/bin/python this.py --test --local
Repeat with SMOKE_MODEL=esmfold2 or af3. Image search must select the fresh build.
This tests execution/output integration, not structure prediction accuracy.
"""
from pathlib import Path
import os
import numpy as np
from biotite.structure.io.pdb import PDBFile
from blocks import Blocks
from chaperone import ExecutionEngine, PipelineDataHandler, init_cli, get_settings

init_cli()
repo = Path(os.environ["CHAPERONE_REPO"])
model = os.environ["SMOKE_MODEL"]
if model not in {"simplefold", "esmfold2", "af3"}:
    raise ValueError(model)
source = repo / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"
reference = next(PipelineDataHandler.from_paths([str(source)], name="reference").get_data(load=True))
sequence = next(iter(reference.sequence.values()))[:48]
assert len(sequence) == 48, sequence
# Keep the original coordinates to exercise blocks requiring structural input.
residues = list(dict.fromkeys(reference.atom_array.res_id))[:48]
atoms = reference.atom_array[np.isin(reference.atom_array.res_id, residues)]
input_path = Path(os.environ["SMOKE_RUNDIR"]) / "input_4zxb_first48.pdb"
input_path.parent.mkdir(parents=True, exist_ok=True)
pdb = PDBFile()
pdb.set_structure(atoms)
pdb.write(input_path)
handler = PipelineDataHandler.from_paths([str(input_path)], name="reference")
get_settings().CHAPERONE_COMPUTE_BACKEND = "local"
kwargs = {"diffusion_batch_size": 1}
if model == "simplefold":
    kwargs.update(model="simplefold_100M", predict_plddt=False)
elif model == "esmfold2":
    kwargs.update(use_local_msas=False)
    if snapshot := os.environ.get("SMOKE_ESMFOLD2_SNAPSHOT"):
        kwargs["model"] = snapshot
block = Blocks[model](**kwargs)
print("SMOKE INPUT", source, sequence, flush=True)
print("SMOKE BLOCK", block, flush=True)
engine = ExecutionEngine(rundir=os.environ["SMOKE_RUNDIR"], reinit=True)
result = engine.run(block, data=handler, run_args={"mode": "local", "max_retries": 0})
assert len(result) == 1, f"Expected one prediction, got {len(result)}"
output = next(result.get_data(which="last", load=True))
assert output.atom_array is not None and len(output.atom_array) > 0
assert np.isfinite(output.atom_array.coord).all(), "Nonfinite prediction coordinates"
assert float(np.ptp(output.atom_array.coord, axis=0).max()) < 1000, "Coordinate extent exceeds 1000 Angstrom"
print("SMOKE PASS", model, output.path, len(output.atom_array), flush=True)
print(result.unpack(), flush=True)
