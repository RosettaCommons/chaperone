# Eval task: de novo zinc metalloprotease

REQUIRED_N = {{REQUIRED_N}}
**TL;DR — submit exactly {{REQUIRED_N}} de novo protease designs that scaffold a HEXXH
zinc-coordination motif against a fibrin Aa substrate peptide.**

You are designing a Zn-dependent protease. The catalytic machinery is a
**HEXXH** motif plus a downstream Tyr/Glu pair, arranged around a single
zinc ion, with the substrate peptide supplied as chain **B**.

## Reference catalytic motif

The supplied `inputs/theozyme.pdb` contains the reference catalytic geometry,
substrate peptide, and zinc site used by the held-out scorer.

Positions are named by their numbering in the reference enzyme. Your design
must place every one of them:

| Reference position | Residue | Role |
|---|---|---|
| `H293` | His | Zn coordination (HEXXH) |
| `E294` | Glu | general base (HEXXH) |
| `H297` | His | Zn coordination (HEXXH) |
| `E316` | Glu | second-shell / substrate positioning |
| `Y377` | Tyr | oxyanion stabilisation |
| `E378` | Glu | third Zn ligand |

## Why this task needs `motif_index_map`

A submission is sequences only — no coordinates. The scorer therefore cannot
tell *which* residues of your design are meant to be catalytic, and cannot
measure motif geometry without that. `motif_index_map` supplies it: for each
reference position above, name the residue in **your** sequence that plays it.

```
motif_index_map = H293:A54;E294:A55;H297:A58;E316:A81;Y377:A122;E378:A123
```

The scorer validates this before spending any compute: every reference
position must be present, each must map to a distinct residue inside your
sequence, and the residue at that index must be the correct amino acid. It then
re-folds your sequence and measures **all-atom motif RMSD** against the
reference geometry — the primary quality signal for this task.


## Quality criteria

The scorer predicts every design with the supplied substrate peptide (chain
**B**) and Zn site (chain **C**), then applies:

1. **Motif fidelity** — all-atom RMSD of the mapped catalytic residues against
   the reference geometry must be ≤ 1.5 Å. This is the public physical gate
   and the headline metric.
2. **High interface confidence** — fold quality and the designed interaction are checked by an independent held-out structure-prediction oracle.
    Make sure to create a high confident interaction region to the best of your abilities.

Design↔substrate contact molecular surface is reported as a diagnostic, but
the current scorer applies no public contact-area, clash, or energetic cutoff.


## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence`, containing only uppercase canonical
  amino-acid letters.
- Required column: `motif_index_map`, in the format described above.
- Optional column: `design_id`, a unique non-empty identifier. The scorer
  generates deterministic row IDs when it is omitted.
- Every pair of designs must have global
  Needleman–Wunsch/BLOSUM62 identity ≤ 70%, using affine gap penalties
  −10/−0.5.
- Do not submit structures, coordinates, paths, the substrate peptide, the
  zinc, or any other column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, a motif map that is incomplete or lands on the wrong residue type, or
a pairwise identity above 70%.
