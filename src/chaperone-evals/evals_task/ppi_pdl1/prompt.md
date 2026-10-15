# Evaluation task: de novo human PD-L1 binder

REQUIRED_N = {{REQUIRED_N}}
**Submit exactly {{REQUIRED_N}} de novo mini-protein designs that bind human PD-L1. Use any
scientifically justified approach necessary to produce the strongest possible
submission.**

You are given `inputs/target.pdb`: the extracellular IgV domain of human PD-L1,
taken from the human PD-1/PD-L1 complex (PDB 4ZQK, source chain A), cleaned and
relabelled as chain **B**, spanning residues 18-132. Your designed binder must
be chain **A**.

## Design details

- **Target chain:** `B` (PD-L1, supplied — do not redesign it)
- **Binder chain:** `A` (yours, designed de novo)
- **Hotspots:** the native PD-1 binding face is represented by PD-L1 residues
  `B56, B66, B115, B121, B122, B123, B124, B125`. Direct your binder at this
  face.

`inputs/config.json` records the target-preparation parameters used to build
`target.pdb` — binder length 60-90 residues, total complex 175-205 residues,
and the hotspot list above. The target and config are provided for reference;
you are not required to use the config, and you may choose your own binder
length.

## Design criteria

The scorer supplies canonical PD-L1 chain B, predicts every complex with the
submitted binder as chain A, and applies:

1. **Binder secondary structure** — the fraction of chain A assigned as loop
   must be `< 0.30`.
2. **Interface energetics** — interface ddG must be `≤ −25` for a passing
   design.
3. **High interface confidence** — symmetric interface-average PAE must be
   `≤ 7.0`. Make sure to create a high-confidence interaction region to the
   best of your abilities.

## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence`, containing only uppercase canonical
  amino-acid letters.
- Optional column: `design_id`, a unique non-empty identifier. The scorer
  generates deterministic row IDs when it is omitted.
- Every pair of designs must have global Needleman–Wunsch/BLOSUM62 identity
  ≤ 70%, using affine gap penalties −10/−0.5.
- Do not submit structures, coordinates, paths, PD-L1, telemetry, or any other
  column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, or a pairwise identity above 70%.
