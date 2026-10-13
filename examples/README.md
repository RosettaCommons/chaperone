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

Test run (one sample per selected input; models still require GPU inference):

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

## Bounded example smoke runs

`--test` uses defaults shipped with the Python package, so it works without a
`tests/test_overrides.yaml` file. It caps supported generators and sequence models
to one sample per input and predictors to one sample/seed. It disables automatic
archival cleanup by default; explicit CLI settings still override this. It does
not generally shorten targets or limit input-table size. An explicit `--config`
is layered above the smoke defaults, and block flags win over both.

Run from the repository root with the Chaperone environment activated. On an
already allocated GPU node, append `--local` to run all stages in that allocation;
omit it when the engine should submit work to Slurm. Use a distinct `--rundir`
for each run so a smoke test cannot reuse a production output directory.

| Script (prefix with `python examples/`) | Smoke work before scientific filters | Required tools/assets |
|---|---|---|
| `benchmarks/ppi/pipeline_foundry.py --test` | One RFD3 backbone, one FoundryMPNN sequence, one RF3 sample | Foundry container and checkpoints |
| `benchmarks/ppi/pipeline_legacy_mpnn.py --test` | One backbone, one legacy MPNN sequence, one RF3 sample | Foundry and private-access fused_mpnn/modelhub runtime and weights |
| `benchmarks/disco/pipeline.py --test` | One DISCO heme-binder design at the fast preset, one AF3 sample | DISCO weights/container and licensed AF3 weights/container |
| `benchmarks/ame/pipeline.py --test` | One `M0024_1nzy` and one `dera_aldolase` backbone, one sequence and AF3 sample each | Foundry, legacy MPNN, AF3, Dunbrack library |
| `benchmarks/ame/pipeline_full.py --test` | One `M0024_1nzy` backbone, one sequence, one AF3 sample | Same AME dependencies |
| `fgfr2-binder/pipeline.py --test` | One backbone, one sequence, one templated AF3 sample | Foundry, legacy MPNN, AF3; enough GPU memory for the full receptor |
| `tests/catcif-storage/pipeline.py --test --auto_store_format=catcif --auto_store_cleanup` | One backbone, one sequence, one AF3 sample, followed by archive/materialization | Foundry, legacy MPNN, AF3; explicit flags re-enable the storage behavior under test |

For example:

```bash
python examples/benchmarks/ppi/pipeline_foundry.py --test --local --rundir=/absolute/path/to/smoke/ppi-foundry
```

The prediction validation directory contains a cohort CSV, not an executable
example script. Use the [validation guide](../docs/site/validation.md) and select
a few targets explicitly before folding the full cohort. No smoke flag relaxes
scientific filters: a one-sample run may produce zero passing designs, and an early
filter may leave later stages unexercised. Record that separately from a successful
end-to-end model execution.
