# Examples

Reference pipeline scripts that show how blocks fit together. Most examples provide a
`pipeline.py` (the PPI benchmark has two named variants) that configures blocks and runs them with `ExecutionEngine`.
Model inference is not run in routine CI; CPU tests check selected configurations
and helper contracts. Most examples need a GPU, while multi-MPNN can run on CPU.

See [Design task descriptions](TASK_DESCRIPTIONS.md) for narrative summaries of
each design case, including its objective, workflow, and evaluation criteria.

| Example | Shows |
|---|---|
| [nanoparticle-design/](nanoparticle-design/) | C3 antigen-scaffold generation, I3 docking, tied MPNN, and AF3 trimer refolding |
| [peptide-binder/](peptide-binder/) | Two-round design around a fixed peptide sequence with flexible coordinates |
| [benchmarks/multi-mpnn/](benchmarks/multi-mpnn/) | ProteinMPNN, LigandMPNN, and SolubleMPNN on identical backbones |
| [cortisol-binder/](cortisol-binder/) | Cortisol conditioning, aromatic-fixed MPNN, gated AF3 cycling and final filtering |
| [dna-binder/](dna-binder/) | Structure-conditioned DNA binding with public RFD3 |
| [enzyme-glycosidase/](enzyme-glycosidase/) | Apo, covalent-intermediate and ligand-bound enzyme branches |
| [enzyme-protease/](enzyme-protease/) | Substrate-conditioned enzyme design and monomer/complex checks |
| [ribozyme-hammerhead/](ribozyme-hammerhead/) | RNA motif rescaffolding and catalytic geometry |
| [fgfr2-binder/](fgfr2-binder/) | Competitive mini-protein binders for FGFR2 (PDB 1DJS), ranked by ipSAE |
| [benchmarks/ppi/](benchmarks/ppi/) | Binder design for the insulin receptor (4ZXB) with two inverse-folding backends |
| [benchmarks/disco/](benchmarks/disco/) | DISCO co-design of a heme B binder, checked by AF3 refolding |
| [benchmarks/ame/](benchmarks/ame/) | Active-site motif recovery for de novo enzyme design on M-CSA motifs |
| [benchmarks/prediction/validation/](benchmarks/prediction/validation/) | The structure-prediction validation cohort (see `docs/site/validation.md`) |
| [benchmarks/guidance-cycling/](benchmarks/guidance-cycling/) | Measured data: whether another round of guided generation improves an already-good backbone, and which MPNN model to feed it |
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
the `mpnn` block (LigandMPNN) has not been established here. Reuse identical
backbones and scoring settings for a controlled backend comparison.

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
| `nanoparticle-design/pipeline.py --test` | One RFD1 C3 backbone, one I3 pose, one tied sequence, one AF3 trimer sample | `rfdiffusion1`, `rpxdock`, `ligandmpnn`, AF3 and weights |
| `peptide-binder/pipeline.py --test` | One RFD3 backbone, one sequence and AF3 sample per round; round two only after passing | Foundry, `ligandmpnn`, AF3 and weights |
| `benchmarks/multi-mpnn/pipeline.py --test` | One sequence from each of three models; optional `--refold` | `ligandmpnn`; AF3 and weights only for refolding |
| `cortisol-binder/pipeline.py --test` | One backbone, two gated MPNN/AF3 cycles, then final AF3 for survivors | Foundry, `ligandmpnn`, AF3 and HBPLUS |
| `dna-binder/pipeline.py --test` | One DNA-conditioned backbone and one sequence/refold per passing round | Foundry RFD3, `ligandmpnn`, AF3 |
| `enzyme-glycosidase/pipeline.py --test` | One backbone per two motif tasks; one sequence and apo/PTM/ligand refolds for passing rows | Foundry, `ligandmpnn`, AF3 with custom CCD input support |
| `enzyme-protease/pipeline.py --test` | One backbone per two motif tasks; one sequence, refinement and monomer/complex refold | Foundry RFD3/RF3, `ligandmpnn`, AF3 |
| `ribozyme-hammerhead/pipeline.py --test` | One 60-nt RNA and one AF3 refold if its motif filter passes | Foundry RFD3NA checkpoint, AF3 |
| `benchmarks/ppi/pipeline_foundry.py --test` | One RFD3 backbone, one FoundryMPNN sequence, one RF3 sample | Foundry container and checkpoints |
| `benchmarks/ppi/pipeline_mpnn.py --test` | One backbone, one `mpnn` sequence, one RF3 sample | Foundry and the `ligandmpnn` image |
| `benchmarks/disco/pipeline.py --test` | One DISCO heme-binder design at the fast preset, one AF3 sample | DISCO weights/container and licensed AF3 weights/container |
| `benchmarks/ame/pipeline.py --test` | One `M0024_1nzy` and one `dera_aldolase` backbone, one sequence and AF3 sample each | Foundry, `ligandmpnn`, AF3, Dunbrack library |
| `benchmarks/ame/pipeline_full.py --test` | One `M0024_1nzy` backbone, one sequence, one AF3 sample | Same AME dependencies |
| `fgfr2-binder/pipeline.py --test` | One backbone, one sequence, one templated AF3 sample | Foundry, `ligandmpnn`, AF3; enough GPU memory for the full receptor |
| `tests/catcif-storage/pipeline.py --test --auto_store_format=catcif --auto_store_cleanup` | One backbone, one sequence, one AF3 sample, followed by archive/materialization | Foundry, `ligandmpnn`, AF3; explicit flags re-enable the storage behavior under test |

The `mpnn` block runs LigandMPNN from the `ligandmpnn` image
(`chaperone install ligandmpnn`); `pipeline_mpnn.py` uses it, while
`pipeline_foundry.py` uses `foundry_mpnn`. GPU validation of the `mpnn` example and
scientific equivalence between the two remain separate checks.

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

## Harbor benchmark isolation

The Harbor agent workspace snapshot excludes this entire examples directory and
all workflow pages. Only `docs/site/blocks-reference.md` is included from the
site documentation. Public example availability is separate from benchmark
workspace access; agents with network access may still find public recipes.
