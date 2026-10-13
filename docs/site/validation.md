# Running a Validation Dataset

This page shows how to score a structure predictor on the PDB validation cohort.
Every predictor goes through the same three steps. Compare results only with matched targets, sampling counts, chain coverage, and scoring settings. This page describes the benchmark protocol, not a report that every listed predictor has completed it.

```
manifest_val_structures  →  <predictor>  →  score_val_structures
   reference assemblies       predictions        lDDT / PEPPR per target
```

## The cohort

The cohort is `examples/benchmarks/prediction/validation/entry_level_val_df.csv`: 1,728 PDB assemblies in five
buckets.

| Bucket | Targets |
|--------|---------|
| `protein` | 50 |
| `protein-protein` | 325 |
| `protein-nucleic` | 164 |
| `protein-ligand` | 636 |
| `misc` | 553 |

Each row carries `pdb_id`, `assembly_id`, `interfaces_to_score`, `pn_units_to_score`,
`example_id` and `bucket`. The scoring columns travel with each target, so the scorer
needs no other input. Any `.csv` or `.parquet` with these columns works.

## Settings

Set these in `.env.local` for your installation. IPD users can supply an untracked `.env.ipd`; it is optional and is not distributed as an active configuration:

| Setting | Used for |
|---------|----------|
| `PDB_MIRROR_PATH` | Local PDB mirror. Reference assemblies are read from here, never downloaded. |
| `PROTEIN_MSA_DIRS` | On-disk protein MSA stores, searched in order. Needed when `use_local_msas` or `msa_dirs` is set. |
| `SIMPLEFOLD_WEIGHTS_DIR` | SimpleFold checkpoints. |

## Run one predictor

```python
from chaperone import ExecutionEngine, JobArgs

engine = ExecutionEngine(
    rundir="runs/val-af3",
    job_args_default=JobArgs(gpu_class="large", mem_gb=64),
)

h = engine.run(
    "manifest_val_structures",
    parquet_path="examples/benchmarks/prediction/validation/entry_level_val_df.csv",
    bucket="protein",            # or "all", "protein-ligand", ...
)
h = engine.run(
    "af3",
    data=h,
    input_mode="mmcif",          # keeps modifications and covalent bonds
    diffusion_batch_size=5,      # five samples, so best-of-5 is available
    read_most_confident_model=False,   # every sample must reach the scorer
    use_local_msas=True,
    raise_if_missing_msa_for_protein_of_length_n=50,
)
h = engine.run("score_val_structures", data=h, metrics=["rf3_allatom_lddt", "peppr"])

df = h.unpack()
```

`manifest_val_structures` writes one reference CIF per target. `score_val_structures`
scores the last structure in each row (the prediction) against the first (the
reference), in place. It does not add a new step.

To run another predictor, change only the middle step:

| Predictor | Block | Settings used for validation |
|-----------|-------|------------------------------|
| AlphaFold3 | `af3` | as above |
| AlphaFold2 | `colabfold` | `model_type="auto"`, `num_models=5`, `read_most_confident_model=False`, `use_local_msas=True` |
| RoseTTAFold3 | `rf3` | `input_mode="file"`, `diffusion_batch_size=5`, `read_most_confident_model=False`, `msa_dirs=[...]` |
| ESMFold2 | `esmfold2` | `diffusion_batch_size=5`, optionally `use_local_msas=True` for MSA conditioning |
| SimpleFold | `simplefold` | `model="simplefold_360M"`, `diffusion_batch_size=5`, `convert_outputs_to_cif=True`; see below |

Test a few targets first by running with `RuntimeArgs(mode="local")` on a GPU node, then
drop the argument to fan out over SLURM. See the [Syntax Reference](syntax.md).

## Reading the scores

`score_val_structures` computes:

- `rf3_allatom_lddt`: all-atom lDDT over the assembly (`rf3_all_atom_lddt`) and over
  each scored interface and PN unit (`rf3_by_type_lddt`). Each row is one predicted
  sample, so the score is `best_of_1_lddt`. Scored units that could not be resolved are
  listed in `rf3_unscored_pn_units`, and atom matching is reported in
  `rf3_atom_alignment`.
- `peppr`: the PEPPR metric set from `peppr_metrics`.

Predictions and references are matched on `(chain_id, res_id, atom_name)`.

With `diffusion_batch_size=5` every target has five scored rows. Best-of-5 is the
maximum over a target's rows, which you take yourself on the unpacked table.

## Monomer predictors

SimpleFold predicts one protein chain and writes it renumbered, always as chain `A`. The
reference assembly usually has more chains and different numbering, so matching atoms
finds nothing. Turn on remapping:

```python
h = engine.run(
    "score_val_structures",
    data=h,
    metrics=["rf3_allatom_lddt", "peppr"],
    remap_outputs_to_input=True,   # map the prediction onto the scored protein chain
    min_atom_coverage=0.95,        # fail a target if less than this fraction maps
)
```

A monomer score is against one chain, so it is not comparable with the whole-assembly
scores from AF3 or RF3. Compare monomer predictors only with each other.

## Tips

- Use `bucket="protein"` (50 targets) for a first pass. The ligand bucket is the
  largest at 636.
- Keep `read_most_confident_model=False` so the scorer sees all five samples.
- `raise_if_missing_msa_for_protein_of_length_n=50` stops a run that would otherwise
  silently fold a long chain with no MSA.
- A PN unit that cannot be resolved is skipped and listed in `rf3_unscored_pn_units`
  with the reason. The target is still scored on the units that remain.
