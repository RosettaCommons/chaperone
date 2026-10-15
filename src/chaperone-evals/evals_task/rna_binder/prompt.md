# Eval task: de novo binder for a structured RNA hairpin

REQUIRED_N = {{REQUIRED_N}}
**TL;DR — submit exactly {{REQUIRED_N}} de novo mini-protein designs that bind the supplied
RNA stem-loop.**

The supplied target, `target.pdb`, is the 20-nucleotide RNA hairpin
`5'-GAGGACCUAGAUCACCCCUC-3'` taken from **PDB 1EC6, source chain B**, and kept
as chain **B**. Your designed binder must be chain **A**.

## Why this target

1EC6 is the 2.4 Å crystal structure of the **Nova-2 KH3 K-homology domain bound
to a 20-mer RNA hairpin**. It was chosen over the other candidate protein–RNA
complexes because:

- **The binding mode is real, not speculative.** The hairpin has a genuine,
  structurally characterized protein partner that reads out the loop.
- **It is a 1:1 complex with a small partner.** Nova-2 KH3 is only 87 residues,
  so a de novo mini-protein is the right size class for this site — unlike
  MS2/PP7 operators, whose coat proteins bind as dimers, or ribosomal-protein
  complexes, whose partners are embedded in a much larger assembly.
- **The RNA is completely ordered.** All 20 nucleotides are present with no
  chain gaps and no missing heavy atoms, so the target geometry and the
  RNA-conformation check are both trustworthy. The obvious alternative, U1A
  bound to U1 snRNA stem-loop II (1URN), has a disordered loop (residue 14
  absent, 13 and 15 partial) and was rejected for that reason.
- **It is compact.** A 20-mer hairpin co-folds far more reliably in modern
  structure-prediction models than a ribozyme or a full snRNA would, which
  keeps the benchmark interpretable.

DSSR annotates the target as a 4-base-pair stem (`G1-C20`, `A2-U19`, `G3-C18`,
`G4-C17`, plus a non-canonical `A5·C16`) closing a 12-nucleotide loop. The
native binding face is that loop, RNA residues `B10`–`B15`, centred on the Nova
UCAC recognition element `U12 C13 A14 C15`. `config.json` supplies that face as
RFD3 hotspots together with base-edge H-bond conditioning.

## Required quality criteria

The scorer supplies canonical RNA chain B, predicts every complex with the
submitted binder as chain A, and applies these public physical constraints:

1. **Base-edge readout** — at least 8 total protein↔RNA hydrogen bonds are
   required, including at least 2 to the minor/sugar edges of RNA bases.
2. **Base stacking** — at least two RNA base-stacking interactions must be
   detected as either a pseudo-pair or pseudo-stack.
3. **Contacts and packing** — at least 12 nucleotide↔amino-acid contacts and
   `contact_ms ≥ 250` are required.
4. **RNA conformation** — backbone RMSD of the base-paired stem residues
   `B1–B5` and `B16–B20`, after fitting that stem, must be ≤ 1.0 Å. The mobile
   loop and whole-hairpin RMSDs are reported but are not gated.
5. **Sterics** — the predicted complex must contain zero backbone clashes.
6. **Held-out confidence** — complex pLDDT must be `≥ 85`, and the held-out
   structure-prediction oracle checks binder↔RNA interaction confidence.

## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence`, containing only uppercase canonical
  amino-acid letters.
- Optional column: `design_id`, a unique non-empty identifier. The
  scorer generates deterministic row IDs when this column is omitted.
- Every pair of designs must have global
  Needleman–Wunsch/BLOSUM62 identity ≤ 70%, using affine gap penalties
  −10/−0.5.
- Do not submit structures, coordinates, paths, the RNA, or any other column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, or a pairwise identity above 70%.
