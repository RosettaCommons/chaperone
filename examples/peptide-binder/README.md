# Flexible-peptide binder design

Two rounds of RFD3 → MPNN → AF3 selection for the peptide sequence
`TVGSDGPPSFRSLSG`. **The peptide sequence is fixed; its conformation is not.**
The initial CIF is constructed from sequence and has placeholder coordinates.
`select_fixed_atoms=False` allows RFD3 to generate peptide coordinates together
with a 60–140-residue binder. This is not a fixed-structure target example.

## Run

Install the `foundry`, `ligandmpnn`, and `alphafold3` containers and their checkpoints;
AF3 weights require separate access. Configure the model paths following the
installation guide. From the repository root in the Chaperone environment:

```bash
python examples/peptide-binder/pipeline.py --test --rundir=/absolute/path/peptide-smoke
```

Append `--local` on an allocated GPU; otherwise use your Slurm site configuration.
Input CIF/JSON files are generated under the selected run directory. No private
structure or scratch path is needed. The normal default is four RFD3 designs,
two MPNN sequences per design per round, and one AF3 sample. `--test` reduces each
model to one sample. Motif sequence identities remain fixed during MPNN.

## Filters and outputs

Round one requires AF3 ipTM ≥0.7, minimum interface PAE ≤2 Å, and backbone RMSD
≤3 Å. Only passing candidates enter the second MPNN/AF3 round, which requires
ipTM ≥0.85, minimum interface PAE ≤1.1 Å, and backbone RMSD ≤2 Å. RMSD uses the
RFD3 structure as reference. Outputs are normal engine artifacts and
`results.csv`; zero passing rows are a valid scientific outcome and must not be
reported as successful execution of stages skipped after filtering.

This port preserves the predecessor's filters; they have not been calibrated
against experimental peptide binding. Input generation and bounded CLI
configuration are CPU-tested. Full GPU execution and binder quality remain to
be validated. Adapted from `baker-laboratory/pipelines`, snapshot `95ef0b6`, with
corrected description of peptide coordinate conditioning and smaller defaults.
