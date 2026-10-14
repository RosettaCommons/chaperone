"""Run with an installed wheel/sdist, outside the checkout and without PYTHONPATH."""
from importlib import metadata, resources
from pathlib import Path
import tempfile
import subprocess
import sys

import chaperone
from blocks import Blocks
from chaperone import ExecutionEngine, PipelineDataHandler

assert metadata.metadata("chaperone-pipelines")["Name"] == "chaperone-pipelines"
assert Path(chaperone.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
for command in ("--help", "list-blocks", "list-available"):
    subprocess.run([sys.executable, "-I", "-m", "chaperone_cli.main", command], check=True)
for package, filename in [("chaperone", "site_profiles/digs.env"), ("blocks", "scripts/ddg_protocol.xml")]:
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
print("PASS installed distribution assets, CPU execution, and cache replay")
