# Examples

Reference pipeline scripts that show how blocks fit together. Each example is a
single `pipeline.py` that configures blocks and runs them with `ExecutionEngine`.
They are worked examples, not tests: they need a GPU cluster and are not run in CI.

| Example | Shows |
|---|---|
| `fgfr2-binder/` | Competitive mini-protein binders for FGFR2 (PDB 1DJS), ranked by ipSAE |
| `benchmarks/ppi/` | Binder design for the insulin receptor (4ZXB) with two inverse-folding backends |
| `benchmarks/disco/` | DISCO co-design of a heme B binder, checked by AF3 refolding |
| `benchmarks/ame/` | Active-site motif recovery for de novo enzyme design on M-CSA motifs |
| `benchmarks/prediction/validation/` | The structure-prediction validation cohort (see `docs/site/validation.md`) |
| `tests/catcif-storage/` | Per-chunk catcif storage and how to read it back |

## Layout

```
examples/{example-name}/
    README.md            # what it does and what to change
    pipeline.py          # entry point (uses init_cli())
    files/               # input files (PDBs, JSON configs)
    rundir/              # created at run time, never committed
```

`pipeline.py` should only configure and run blocks. Inputs come from
`FILES_DIR = SCRIPT_DIR / "files"`, settings from `.env.ipd`, or the command line.
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
./examples/{EXAMPLE}/pipeline.py --rfd3.n_batches=50 --rfd3.batch_size=20 --partition=<your-gpu-partition>
```
