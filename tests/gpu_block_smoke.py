#!/usr/bin/env python
"""Run one bounded, real GPU block check through the normal execution engine.

Run from an allocated GPU shell after installing the selected image and weights::

    python tests/gpu_block_smoke.py --test --local --smoke_block=boltz2 \
        --rundir=/absolute/path/to/boltz2-smoke

Repeat with chai1, protenix, or e1. The input is 48 residues from the bundled
PPI reference, renamed to chain A for the embedding block. Predictors use one
sample with the normal diffusion/recycle schedule; checkpoint/kernel/MSA
settings are unchanged. This verifies execution, artifact loading and finite values, not
scientific accuracy. Use a fresh run directory when validating a new image.
"""
from pathlib import Path

from absl import flags
import numpy as np

BLOCK_NAMES = ("boltz2", "chai1", "protenix", "e1")
REPO_ROOT = Path(__file__).resolve().parents[1]


def stage_reference(source: Path, destination: Path, residues: int = 48) -> None:
    """Crop one real protein chain and write it with the E1-required chain ID."""
    from biotite.structure import filter_amino_acids, get_residue_starts
    from biotite.structure.io.pdb import PDBFile
    from chaperone.io import parse_to_atom_array

    if not 1 <= residues <= 48:
        raise ValueError("GPU smoke inputs must contain between 1 and 48 residues")
    atoms = parse_to_atom_array(str(source))
    protein = atoms[filter_amino_acids(atoms)]
    if len(protein) == 0:
        raise ValueError(f"No protein atoms in {source}")
    chain = protein[protein.chain_id == protein.chain_id[0]]
    starts = get_residue_starts(chain, add_exclusive_stop=True)
    if len(starts) - 1 < residues:
        raise ValueError(f"First protein chain has fewer than {residues} residues")
    atoms = chain[:starts[residues]].copy()
    atoms.chain_id[:] = "A"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(str(destination))


def validate_artifacts(handler, block_name: str) -> list[str]:
    """Read the block's recorded artifacts and reject empty/nonfinite outputs."""
    if len(handler) != 1:
        raise AssertionError(f"Expected one output row from {block_name}, got {len(handler)}")
    outputs = list(handler.get_data(which="last", load=False))
    if len(outputs) != 1:
        raise AssertionError(f"Expected one {block_name} artifact, got {len(outputs)}")
    paths = []
    for output in outputs:
        if output.name != block_name:
            raise AssertionError(f"Last artifact is {output.name}, expected {block_name}")
        path = Path(output.path)
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")
        if block_name == "e1":
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as tensors:
                keys = set(tensors.keys())
                required = {"logits", "token_embeddings", "mean_token_embeddings"}
                if not required <= keys:
                    raise AssertionError(f"Missing E1 tensors: {required - keys}")
                for name in required:
                    values = tensors.get_tensor(name).float().numpy()
                    if values.size == 0 or not np.isfinite(values).all():
                        raise AssertionError(f"Empty/nonfinite E1 {name}: {path}")
        else:
            from chaperone.io import parse_to_atom_array
            atoms = parse_to_atom_array(str(path))
            if len(atoms) == 0 or not np.isfinite(atoms.coord).all():
                raise AssertionError(f"Empty/nonfinite coordinates: {path}")
            if float(np.ptp(atoms.coord, axis=0).max()) >= 1000:
                raise AssertionError(f"Coordinate extent exceeds 1000 Angstrom: {path}")
        paths.append(str(path))
    return paths


def main() -> None:
    from chaperone import ExecutionEngine, PipelineDataHandler, init_cli
    from blocks import Blocks

    flags.DEFINE_enum("smoke_block", "boltz2", BLOCK_NAMES, "Block to test on the allocated GPU")
    flags.DEFINE_string(
        "smoke_input", str(REPO_ROOT / "examples/benchmarks/ppi/files/4zxb_cropped.pdb"),
        "Reference protein structure to crop",
    )
    flags.DEFINE_integer("smoke_residues", 48, "Residues in the smoke input", lower_bound=1, upper_bound=48)
    init_cli(blocks=[Blocks[name] for name in BLOCK_NAMES])
    options = flags.FLAGS
    name = options.smoke_block
    engine = ExecutionEngine(rundir=REPO_ROOT / "smoke-runs" / name)
    input_path = engine.rundir / "input" / "reference48.pdb"
    stage_reference(Path(options.smoke_input).expanduser().resolve(), input_path, options.smoke_residues)
    handler = PipelineDataHandler.from_paths([str(input_path)], name="reference")
    kwargs = {} if name == "e1" else {
        "diffusion_batch_size": 1,
    }
    block = Blocks[name](**kwargs)
    print(f"GPU SMOKE {name}: input={input_path}, image={block._get_image().get_image_path()}", flush=True)
    result = engine.run(block, data=handler, run_args={"max_retries": 0})
    paths = validate_artifacts(result, name)
    print(f"GPU SMOKE PASS {name}: {paths}", flush=True)
    print(result.unpack())
    print(engine)


if __name__ == "__main__":
    main()
