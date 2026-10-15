# Compare three MPNN model families

Run ProteinMPNN, LigandMPNN and SolubleMPNN on the same **user-supplied**
backbones. The historical generated biotin backbone is excluded pending
redistribution permission. Provide an authorized directory of PDB/CIF files:

```bash
chaperone install ligandmpnn
python examples/benchmarks/multi-mpnn/pipeline.py --input_dir=/path/to/structures --test --local --rundir=/absolute/path/mpnn-smoke
```

`--test` requests one sequence per model per input; the default requests four.
Each designed sequence yields one packed structure. Model-specific filenames,
design IDs and prediction steps keep the branches separate through merge/repack.
Without `--refold`, no design-quality ranking is performed.

For AF3 comparison, install `alphafold3`, configure authorized weights, and add
`--refold`. Each model uses its own predicted structure for ranking score ≥0.5
and backbone RMSD ≤3 Å. These illustrative gates do not establish binding.
Use `--local` within an allocation; omit it for site-configured Slurm execution.

Results and lineage are written to `results.csv` and normal engine artifacts.
Historical H200 checks exercised packed output, fixed motifs, ligand retention
and optional refolding on an input no longer distributed. New inputs require
their own bounded GPU check. ProteinMPNN and SolubleMPNN do not condition sequence
design on ligand atoms; their shared packing stage can still use ligand context.
