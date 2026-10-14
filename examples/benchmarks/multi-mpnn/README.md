# Compare three MPNN model families

Run ProteinMPNN, LigandMPNN, and SolubleMPNN on the **same** bundled
biotin-associated design backbone. Each branch uses the public LigandMPNN
implementation and its corresponding checkpoint. Results are native sequence
rows retaining their backbone context; the branches are combined with the
standard handler API.

```bash
chaperone install ligandmpnn
python examples/benchmarks/multi-mpnn/pipeline.py --test --local --rundir=/absolute/path/mpnn-smoke
```

`--test` requests one sequence per model (three total). The default requests four
per model. MPNN runs on CPU by default. To also compare AF3 refolding, install
`alphafold3`, configure separately obtained `ALPHAFOLD3_WEIGHTS_DIR`, and add
`--refold`. Use `--local` only where the needed hardware is already allocated;
omit it to submit through Slurm. AF3 uses one sample; the illustrative filter is
ranking score ≥0.5 and backbone RMSD ≤3 Å.

Outputs are ordinary engine artifacts plus `results.csv` in the run directory.
All three model branches should retain the same input structure and distinct
model names. Without `--refold`, sequence generation is the endpoint and there
is no quality ranking. With refolding, an empty final table means no candidate
met the filters.

The bundled CIF is a computational design from the original
`baker-laboratory/pipelines` multi-MPNN example, snapshot `95ef0b6`; it is not an
experimental reference. Configuration and native sequence/lineage regression
checks do not establish scientific equivalence between model families or
validate GPU inference. The public port uses current `engine.run` and handler
concatenation APIs in place of the predecessor's unavailable `submit/gather`.
