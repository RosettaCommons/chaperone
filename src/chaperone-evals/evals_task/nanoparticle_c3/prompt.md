# Eval task: de novo C3 cyclic homo-oligomer

REQUIRED_N = {{REQUIRED_N}}
**TL;DR — submit exactly {{REQUIRED_N}} de novo protomer sequences that assemble into a
C3-symmetric homotrimer.**

The design target is not a molecule you are given; it is a **point group**.
Each submitted sequence is one **protomer**, and the assembly is that protomer
in **three identical copies** related by a 120° rotation about a single
three-fold axis. This is the building block of a self-assembling protein
nanoparticle: the trimer is the C3 component that later docks into a larger
icosahedral or octahedral shell, so the trimer itself has to be a real,
confidently predicted, closed ring.

- **Point group:** C3 (cyclic, three protomers, one three-fold axis).
- **Protomer length:** target **60–90 residues** per chain. Shorter than ~60
  and a protomer has too little surface to make three interfaces; longer than
  ~90 and the assembly stops being a compact C3 building block. This range is
  guidance — it is reported by the scorer but is not itself a pass/fail gate.
- **One sequence per design.** You submit a single `sequence`. **The scorer
  replicates it into three identical chains A, B and C** and folds the trimer.
  Do not submit the same sequence three times; do not submit three different
  sequences. A homo-oligomer whose chains differ is a different molecule.

`target.pdb` is a **reference protomer** (chain **A**), included so the
expected size and shape of a submission is inspectable. It is illustrative
only: the scorer never reads it, never aligns to it, and does not require your
designs to resemble it.

`config.json` records the generation configuration the reference pipeline used
(C3, 70 residues per protomer, RFdiffusion v1 symmetric backbones, homo-oligomer
MPNN). You may change any of it.

Use the `chaperone` framework and the design tools in this repository. Note that
`rfd3` has no symmetry mode — `rfd1` (RFdiffusion v1) does, via
`symmetry="c3"`, and its contig string takes the **total** assembly length
(3 × protomer). You may of course reach a C3 trimer another way.

## Required quality criteria

The scorer builds chains A/B/C from your one sequence, predicts the trimer,
and applies these public physical constraints:

1. **C3 rotation geometry** — the rotational-angle deviation from the ideal
   120° relationship must be ≤ 5°.
2. **Equivalent protomer folds** — aligned subunit RMSD must be ≤ 1.0 Å.
   Common-axis agreement is also reported, but has no separate public cutoff.
3. **Interface size** — contact molecular surface at the representative A↔B
   protomer interface must be `contact_ms ≥ 600`.
4. **Protomer structure** — the fraction of chain A assigned as loop must be
   `< 0.35`.
5. **Protomer compactness** — chain A's Cα radius of gyration in the held-out
   prediction must be `≤ 14 Å`. This rejects extended coiled-coil rods that
   satisfy C3 geometry without forming a compact tertiary building block.
6. **Sterics** — the predicted trimer must contain zero backbone clashes.
7. **Interface energetics** (soft) — A↔B interface ddG must be `< −35` for
   the soft-pass result, not the hard physical pass.
8. **High interface confidence** — held-out trimer ipTM must be `≥ 0.85` and
   pLDDT must be `≥ 85`.
    Make sure to create a high confident interaction region to the best of your abilities.

The 60–90-residue protomer range is guidance and is reported, but it is not a
pass/fail gate.

## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence`, containing only uppercase canonical
  amino-acid letters.
- Optional column: `design_id`, a unique non-empty identifier. The
  scorer generates deterministic row IDs when this column is omitted.
- Every pair of designs must have global
  Needleman–Wunsch/BLOSUM62 identity ≤ 70%, using affine gap penalties
  −10/−0.5.
- Do not submit `sequence_B`, `sequence_C`, structures, coordinates, paths, or
  any other column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, a pairwise identity above 70%, or any extra column.
