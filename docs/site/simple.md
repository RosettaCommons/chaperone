# Simple Example

A pipeline is a Python script: configure a few blocks, pass their outputs to the
next block, then inspect or save the resulting table. This walkthrough generates
one backbone with RFdiffusion3, designs one sequence with FoundryMPNN, and folds
that sequence with RF3.

## Before running

Install Chaperone using the [installation guide](https://github.com/RosettaCommons/chaperone#installation-manual).
Build the `foundry` container with `chaperone install foundry`, obtain the upstream
model checkpoints, and configure `FOUNDRY_CHECKPOINT_DIRS` for your installation.
GPU inference requires a supported NVIDIA driver and container backend. Configure
your cluster's [Slurm settings](syntax.md#slurm-site-configuration), or run on a
GPU workstation with `--partition=local`. IPD users can supply an untracked
`.env.ipd`; other installations do not need it.

This is an integration example, not a validated protein-design protocol.
FoundryMPNN has a public upstream implementation; equivalence to the legacy
`mpnn` backend has not been established by a published comparison in this repository.
The legacy backend requires access to a private upstream repository.

## Files and script

Create a small working directory with an input configuration and a script:

```text
my-design/
    files/config.json
    pipeline.py
    output/              # created when the pipeline runs
```

Use this unconditional 100-residue design specification in `files/config.json`:

```json
{"example": {"length": "100"}}
```

Save the following as `pipeline.py`:

```python
from pathlib import Path

from chaperone import ExecutionEngine, init_cli

init_cli()  # enables --test, --partition, and per-block CLI overrides
root = Path(__file__).resolve().parent
engine = ExecutionEngine(rundir=root / "output")

backbones = engine.run(
    "rfd3",
    inputs=str(root / "files" / "config.json"),
    batch_size=1,
    n_batches=1,
)
sequences = engine.run(
    "foundry_mpnn",
    data=backbones,
    model_type="protein_mpnn",
    num_seq_per_pdb=1,
)
predictions = engine.run("rf3", data=sequences, diffusion_batch_size=1)

# A handler preserves the backbone → sequence → prediction lineage.
# Unpack only when you want a flat table for inspection or export.
df = predictions.unpack()
print(df.head())
df.write_parquet(root / "output" / "results.parquet")
```

Run with the activated Chaperone environment:

```bash
python my-design/pipeline.py --test
# On a GPU workstation without Slurm:
python my-design/pipeline.py --test --partition=local
```

The test mode reduces work to check inputs, containers, and output flow. It does
not establish design quality. Check the run's outputs before increasing sample
counts; an absent model weight or incompatible GPU driver still needs to be fixed.

## How a larger pipeline fits together

Keep paths and scientific parameters near the top of the script, then configure
blocks, execute them, and export results at the end. `engine.run()` returns a
`PipelineDataHandler` carrying structures, sequences, metrics, and parentage.
Sequence design may produce multiple rows per backbone; prediction adds another
step without losing that relationship. Run directories retain ordinary manifests,
logs, and result tables for inspection and restart.

You can add metric and filter blocks, use a different predictor, branch from the
same input handler, or cycle sequence design and prediction. Use a measured
acceptance criterion before scaling: a completed prediction alone does not show
that a design will work experimentally.

## Where to go next

- [Core Examples](core-examples.md): inspect complete binder, ligand, enzyme, and storage workflows.
- [Block Reference](blocks-reference.md): choose tools and inspect their parameters, source, and metric definitions.
- [Syntax Reference](syntax.md): configure caching, chunking, branching, cycling, and Slurm resources.
- [Agent Skills](skills.md): have a coding agent help plan and test a workflow using the repository guides.
