# Examples

Reference pipeline scripts that show how blocks fit together. Most examples provide a
`pipeline.py` (the PPI benchmark has two named variants) that configures blocks and runs them with `ExecutionEngine`.
They are worked examples, not tests: they need a GPU cluster and are not run in CI.

| Example | Shows |
|---|---|
| [fgfr2-binder/](fgfr2-binder/) | Competitive mini-protein binders for FGFR2 (PDB 1DJS), ranked by ipSAE |
| [benchmarks/ppi/](benchmarks/ppi/) | Binder design for the insulin receptor (4ZXB) with two inverse-folding backends |
| [benchmarks/disco/](benchmarks/disco/) | DISCO co-design of a heme B binder, checked by AF3 refolding |
| [benchmarks/ame/](benchmarks/ame/) | Active-site motif recovery for de novo enzyme design on M-CSA motifs |
| [benchmarks/prediction/validation/](benchmarks/prediction/validation/) | The structure-prediction validation cohort (see `docs/site/validation.md`) |
| [tests/catcif-storage/](tests/catcif-storage/) | Per-chunk catcif storage and how to read it back |

## Layout

```
examples/{example-name}/
    README.md            # what it does and what to change
    pipeline.py          # entry point (uses init_cli())
    files/               # input files (PDBs, JSON configs)
    rundir/              # created at run time, never committed
```

`pipeline.py` should only configure and run blocks. Inputs come from
`FILES_DIR = SCRIPT_DIR / "files"`, settings from `.env.local` (or a supplied, untracked `.env.ipd` at IPD), or the command line.
If a pipeline needs bespoke logic, keep it in a small module beside the script and
promote it to `src/blocks/` once it is general.

## Running

Test run (one design per block, fast inference):

```bash
./examples/{EXAMPLE}/pipeline.py --test
```

Full scale:

```bash
./examples/{EXAMPLE}/pipeline.py
```

Override block parameters:

```bash
./examples/{EXAMPLE}/pipeline.py --rfd3.n_batches=2 --rfd3.batch_size=1 --partition=<your-gpu-partition>
```

## Benchmark scope and validation

The PPI scripts compare inverse-folding backends through refolding and interface
metrics; DISCO measures backbone and ligand pose recovery; AME measures catalytic
motif recovery; the prediction cohort compares predicted structures with PDB
references. These are computational benchmarks, not experimental activity assays.
A script does not constitute benchmark results, and FoundryMPNN equivalence to
the legacy MPNN backend has not been established here. The latter requires access
to its private upstream repository. Reuse identical backbones and scoring settings
for a controlled backend comparison.

These campaigns are not executed in routine CI; no fresh-cluster validation of
every example is claimed. Test prerequisites and a small input on your own system
before increasing scale. See [Core Examples](../docs/site/core-examples.md) for
objectives and validation limits.

Add a new example when it documents a distinct reusable workflow. Include inputs,
prerequisites, a bounded test command, expected outputs, acceptance criteria, and
the scope of completed validation. Examples are useful references for both people
and agents, but are not required for an agent to discover blocks or compose a new
workflow from the skills and block reference.
