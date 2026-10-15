# Eval task: de novo major-groove dsDNA binder

REQUIRED_N = {{REQUIRED_N}}
** Submit exactly {{REQUIRED_N}} de novo mini-protein designs that binds to a
specific double-stranded DNA sequence through its major groove.**

The supplied target, `target.pdb`, is a B-form DNA duplex of the
self-complementary Dickerson–Drew dodecamer `CGCGAATTCGCG` (the sequence of
PDB 1BNA) prepared with a structure-prediction model. It is
supplied as two strands: chain **B** is the sense strand
(5'-CGCGAATTCGCG-3', residues 1–12) and chain **C** is the antisense strand
(residues 1–12). Your designed binder must be designated as chain **A**.

The coordinates are predicted rather than taken straight from the PDB entry
for a concrete reason: 1BNA's deposited residue 1 carries no 5'-terminal
phosphate. If you substitute your own duplex,make sure every nucleotide is complete.

Sequence-specific DNA recognition happens in the **major groove**, where each
base presents a distinct hydrogen-bonding edge. `config.json` is the reference generation config and annotates exactly those atoms per position:

| Base | Major-groove donors | Major-groove acceptors |
|------|---------------------|------------------------|
| DA   | N6                  | N7                     |
| DG   | —                   | O6, N7                 |
| DC   | N4                  | —                       |
| DT   | —                   | O4                     |


## Required quality criteria

The scorer supplies canonical duplex chains B and C, predicts each complex
with the submitted binder as chain A, and applies these public physical
constraints:

1. **Major-groove readout** — at least 3 protein↔DNA hydrogen bonds must land
   on major-groove base atoms.
2. **Polar recognition** — at least 8 total protein↔DNA hydrogen bonds and at
   least 10 nucleotide↔amino-acid contacts are required.
3. **Interface size** — contact molecular surface between chain A and chains
   B/C must be `contact_ms ≥ 400`.
4. **Duplex preservation** — nucleotide RMSD after nucleotide-backbone
   alignment must be ≤ 2.0 Å, so the complex cannot manufacture an interface
   by deforming the supplied B-form duplex.
5. **Sterics** — the predicted complex must contain zero backbone clashes.
6. **High interface confidence** — fold quality and the protein-dna interaction are checked by an independent held-out structure-prediction oracle.
    Make sure to create a high confident interaction region to the best of your abilities.

## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence`, containing only uppercase canonical
  amino-acid letters.
- Optional column: `design_id`, a unique non-empty identifier. The
  scorer generates deterministic row IDs when this column is omitted.
- Every pair of designs must have global
  Needleman–Wunsch/BLOSUM62 identity ≤ 70%, using affine gap penalties
  −10/−0.5.
- Do not submit structures, coordinates, paths, the DNA, or any other column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, or a pairwise identity above 70%.
