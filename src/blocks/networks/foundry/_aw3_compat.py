"""Backport AtomWorks 3 inference fixes without changing Foundry 1.0 ownership.

The companion patch adapts fixes from upstream 1178aba6 to PR #404. Both the
pinned source hashes and patch applicability are checked before any file changes.
"""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

_SOURCE_HASHES = {
    "models/rf3/src/rf3/inference_engines/rf3.py": "e1877ad92b3834c3a81a7b52945b42f55f8baf6603551994ce8ba69d39bc86b8",
    "models/rf3/src/rf3/utils/io.py": "9900700ac8316e86d2a5293a45564064cb413682fbdb3d7458cb293338a99f62",
    "models/rfo/src/rfo/backprop/inference_pipeline/af3_inference_pipeline.py": "50fb50ec3b3b2c854b28940fec09871f9b76bd764851e720949932be84af328f",
    "models/rfo/src/rfo/backprop/tools/af3.py": "eccbe83b94df2151f262b3614a7c87e8a60cdfb46fcd2d12ca99eafc2a9f4797",
    "models/rfo/src/rfo/backprop/tools/af3_auto_detect_breaking.py": "554489ae07daef3af5bc4d8fc5619e881462d367474e93bbc5732f560d30d421",
    "models/rfo/src/rfo/backprop/tools/af3_no_breaking_point.py": "ddf35fb361ae13a03d4334ed1e438ef3b1f34ba81936a86c30d50cbf4cecb513",
}


def patch(root: Path, patch_file: Path) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unexpected Foundry 1.0 source: {name}")
    command = ["git", "-C", str(root), "apply"]
    subprocess.run([*command, "--check", str(patch_file)], check=True)
    subprocess.run([*command, str(patch_file)], check=True)


def check() -> None:
    import importlib
    import tempfile

    import numpy as np
    import torch
    from biotite.structure import AtomArray, BondList
    from biotite.structure.io import pdbx
    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.io import dump_trajectories

    assert (
        RF3InferenceEngine(ckpt_path="unused.ckpt").transform_overrides[
            "residue_cache_dir"
        ]
        is None
    )
    atoms = AtomArray(3)
    atoms.chain_id[:] = "A"
    atoms.res_name[:] = "GLY"
    atoms.res_id[:] = [1, 2, 3]
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    atoms.bonds = BondList(3)
    frames = [
        torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) + i for i in (0, 1)
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dump_trajectories(
            frames,
            atoms,
            root / "trajectory",
            align_structures=False,
            file_type="cif",
            final_frame_first=False,
        )
        for sample in range(2):
            result = pdbx.get_structure(
                pdbx.CIFFile.read(root / f"trajectory_model_{sample}.cif")
            )
            np.testing.assert_allclose(result.coord[-1], frames[-1][sample])
        for name in ("af3", "af3_auto_detect_breaking", "af3_no_breaking_point"):
            adapter = importlib.import_module(f"rfo.backprop.tools.{name}")
            path, components = adapter.seq2cif(
                "AG",
                add_info={
                    "chain_id": "A",
                    "msa_paths": None,
                    "smiles": None,
                    "chain_iid_token_level": None,
                },
                out_file=root / f"{name}.cif",
            )
            assert components[0]["chain_type"] == "polypeptide(L)"

            class Parsed(Exception):
                pass

            def capture(data):
                parsed = data["atom_array"]
                assert list(dict.fromkeys(parsed.res_name)) == ["ALA", "GLY"]
                raise Parsed

            for function in (adapter.cif2input, adapter.cif2input_with_template):
                try:
                    function(path, capture)
                except Parsed:
                    pass
                else:
                    raise AssertionError(
                        "RFO parser did not reach the inference pipeline"
                    )


if __name__ == "__main__":
    patch(Path(sys.argv[1]), Path(sys.argv[2]))
    if "--check" in sys.argv[3:]:
        check()
