"""Run with an installed wheel/sdist, outside the checkout and without PYTHONPATH."""
from importlib import metadata, resources
from pathlib import Path
import tempfile
import subprocess
import sys

import chaperone
from blocks import Blocks
from blocks.utils.params import generate_params, list_available_ligands, resolve_param_file
from chaperone import ExecutionEngine, PipelineDataHandler
from chaperone.io import parse_to_atom_array
import numpy as np

distribution = metadata.distribution("chaperone-pipelines")
assert distribution.metadata["Name"] == "chaperone-pipelines"
assert distribution.metadata["License-Expression"] == "MIT AND BSD-3-Clause AND CC-BY-NC-SA-4.0"
license_files = [
    path for path in distribution.files
    if ".dist-info/licenses/" in str(path)
]
assert license_files
for path in license_files:
    assert distribution.locate_file(path).is_file(), path
notices = next(path for path in license_files if path.name == "THIRD_PARTY_NOTICES.md")
assert "Copyright (c) 2026 Brian Coventry" in distribution.locate_file(notices).read_text()
assert metadata.version("atomworks").split(".")[0] == "3"
assert Path(chaperone.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not resources.files("chaperone").joinpath("site_profiles/digs.env").is_file()
assert not (Path(chaperone.__file__).resolve().parent.parent / "chaperone-evals").exists()
assert Blocks["ligand_ddg_soft"]().replicates == 5
assert callable(generate_params)
assert Blocks["ppi_ddg"]().replicates == 1
assert {"FAD", "IAI", "OQO", "SAM"} <= set(list_available_ligands())
for ligand in ("FAD", "IAI", "OQO", "SAM"):
    assert Path(resolve_param_file(ligand)).is_file()
for command in ("--help", "list-blocks", "list-available"):
    subprocess.run([sys.executable, "-I", "-m", "chaperone_cli.main", command], check=True)
for package, filename in [("blocks", "scripts/ddg_protocol.xml"), ("blocks", "networks/foundry/.assets/LICENSE.md")]:
    assert resources.files(package).joinpath(filename).is_file(), filename

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    pdb = root / "input.pdb"
    pdb.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  GLY A   1       1.450   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      3  C   GLY A   1       2.000   1.400   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   GLY A   1       1.500   2.450   0.000  1.00  0.00           O  \nEND\n"
    )
    data = PipelineDataHandler.from_paths([str(pdb)], name="input")
    options = dict(rundir=root / "run", run_args_default=dict(mode="local", use_local_gpus=False, max_workers=1))
    result = ExecutionEngine(**options).run(Blocks["copy"](n_multiply=2), data=data)
    assert len(result) == 2
    output = root / "run/copy/output.parquet"
    mtime = output.stat().st_mtime_ns
    cached = ExecutionEngine(**options).run(Blocks["copy"](n_multiply=2), data=data)
    assert len(cached) == 2
    assert output.stat().st_mtime_ns == mtime
    # Exercise the installed AtomWorks/CatCIF integration, including cleanup.
    original = parse_to_atom_array(str(pdb))
    data.store("input", root / "packed.catcif", cleanup=True)
    restored = next(data.get_data(load=True, which="last"))
    assert ".catcif:" in restored.path
    assert not pdb.exists()
    np.testing.assert_allclose(original.coord, restored.atom_array.coord, atol=1e-3)
    for annotation in ("chain_id", "res_id", "res_name", "atom_name"):
        np.testing.assert_array_equal(
            getattr(original, annotation), getattr(restored.atom_array, annotation)
        )
print("PASS installed assets, CPU execution, cache replay, and CatCIF roundtrip")
