# Compare three MPNN model families

Run ProteinMPNN, LigandMPNN, and SolubleMPNN on the **same** bundled
biotin-associated design backbone. Each branch uses the public LigandMPNN
implementation and its corresponding checkpoint. Results are side-chain-packed structure
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
All three model branches retain the same input structure and distinct
model names. Each model uses a distinct MPNN filename suffix, which also
keeps design identifiers distinct after merging and unpack/repack. Refolding runs in a separate AF3/RMSD step for each model before
the branches are combined, preventing identical packed basenames from
overwriting another model’s AF3 input. Without `--refold`, sequence design and side-chain packing are the endpoint and there
is no quality ranking. With refolding, an empty final table means no candidate
met the filters.

The bundled CIF is a computational design from the original
`baker-laboratory/pipelines` multi-MPNN example, snapshot `95ef0b6`; it is not an
experimental reference. Configuration and structure/lineage regression
checks complement the bounded GPU execution check below; neither establishes
scientific equivalence between model families. The public port uses current `engine.run` and handler
concatenation APIs in place of the predecessor's unavailable `submit/gather`.

Each designed sequence produces one packed structure (`output_mode="structures"`,
`number_of_packs_per_design=1`); packing does not multiply the sample count.

## GPU checkpoint and packing validation

A bounded CoreWeave H200 diagnostic exercised all **14 public sequence
checkpoints**, explicitly requesting `device="cuda"`:

| Model family | Backbone noise levels |
| --- | --- |
| ProteinMPNN | 0.02, 0.10, 0.20, 0.30 |
| LigandMPNN | 0.05, 0.10, 0.20, 0.30 |
| SolubleMPNN | 0.02, 0.10, 0.20, 0.30 |
| Per-residue membrane MPNN | 0.20 |
| Global-label membrane MPNN | 0.20 |

Each checkpoint generated one sequence and one packed structure from the same
48-residue pocket fragment of the bundled biotin-associated design. The test
used three fixed residues (two from motif metadata and one explicit constraint),
21 ligand heavy atoms, and valid membrane labels for the membrane models.
All 14 cases passed:

- SHA-256 verification of each checkpoint against the block's pinned digest,
  and the sampled FASTA's recorded checkpoint path matched the requested model.
- The packed protein sequence matched its sampled FASTA.
- Fixed-residue identities and observed heavy-atom coordinates were preserved
  exactly after compilation; ligand atom identities and coordinates were also
  unchanged.
- Packed coordinates were finite, protein sidechains were present, and merged
  outputs retained distinct design identifiers after unpacking and repacking.

The public MPNN container performed inference and packing; Chaperone's compile
step restored observed fixed-residue coordinates that the native packer otherwise
reconstructs. This checks routing, motif constraints, ligand retention, and packed
output handling. It does not establish equal design quality across checkpoints,
validate membrane-protein quality, or constitute a fresh container build test.
Preserving ligand atoms in every packed output does not mean every sequence
model conditions on them. LigandMPNN performs ligand-conditioned sequence
design; ProteinMPNN, SolubleMPNN, and the membrane sequence models do not use
ligand atoms as sequence-design context. Their shared side-chain packing stage
can still use ligand context.

## Three-family refolding check

A separate H200 run exercised the complete optional refolding branch on the
full bundled backbone: one packed sequence each from ProteinMPNN, LigandMPNN,
and SolubleMPNN, followed by independent AF3 and RMSD steps. Three additional
fixed motif positions were supplied for this diagnostic. All three packed
outputs preserved those residue identities and coordinates, retained all 21
ligand heavy atoms unchanged, and matched their sampled FASTA sequences.
The merged result retained three distinct design IDs, output paths, and
model-specific prediction/metric histories after unpacking and repacking.

| Sequence model | AF3 ranking score | Backbone RMSD | Illustrative gates |
| --- | --- | --- | --- |
| ProteinMPNN | 0.86 | 0.49 Å | Pass |
| LigandMPNN | 0.86 | 0.53 Å | Pass |
| SolubleMPNN | 0.84 | 0.51 Å | Pass |

Each verdict uses the ranking score and RMSD from that model's own AF3
structure, with the unchanged example thresholds of ≥0.5 and ≤3 Å.
Selection was bypassed during execution to check every branch; all three
subsequently met these illustrative thresholds. One sample per family on one
backbone is insufficient to rank model quality or establish equivalence, and
these gates do not establish biotin binding. This used the public MPNN
container and an existing AF3 image with the public Chaperone runner.
