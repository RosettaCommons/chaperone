# Eval task: antibody CDR design on a supplied framework

REQUIRED_N = {{REQUIRED_N}}
**TL;DR — submit exactly {{REQUIRED_N}} antibody Fv designs that bind hen egg-white
lysozyme. The framework is given and must not change; you design only the six
CDR loops.**

This is a *grafting* task, not a de novo binder task. You are handed a real
antibody framework and an antigen. Everything outside the six declared CDR
spans is supplied, and a submission that changes any framework residue is
rejected as a contract violation — it is a different molecule, not a bad design.

## The system

`framework.pdb` is the IgG1-kappa **D1.3 Fv** (PDB **1VFB**, 1.8 Å), relabelled
to the benchmark chain convention:

| Chain | Molecule |
|---|---|
| `H` | D1.3 VH (116 residues) |
| `L` | D1.3 VL (107 residues) |
| `T` | hen egg-white lysozyme, the antigen (129 residues) |

`target.pdb` is the antigen alone, as chain **T**. The scorer supplies chain T
and co-folds it with every design; do not submit it. The antigen chain is
deliberately `T` rather than `B` so that `H`, `L` and `T` read unambiguously
and no submitted antibody chain can collide with it.

`config.json` carries the same information machine-readably.

## Numbering scheme

**Sequential, 1-based, per chain** — exactly as deposited in 1VFB, which has no
insertion codes and no numbering gaps. Residue *i* of chain H is character *i*
of `sequence_H`. There is no Kabat/Chothia insertion-code arithmetic anywhere
in this benchmark.

## Framework reference sequences (exact)

```
sequence_H (116 aa)
QVQLQESGPGLVAPSQSLSITCTVSGFSLTGYGVNWVRQPPGKGLEWLGMIWGDGNTDYNSALKSRLSISKDNSKSQVFLKMNSLHTDDTARYYCARERDYRLDYWGQGTTLTVSS

sequence_L (107 aa)
DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIK
```

## Designable CDR spans

Spans use the **AbM** CDR definition (the union of Kabat's
sequence-hypervariability and Chothia's structural-loop definitions), mapped
onto the sequential numbering above. Every position listed is designable; every
position not listed is framework and must be reproduced verbatim.

| CDR | Chain | Span (inclusive) | Reference segment | Allowed length |
|---|---|---|---|---|
| CDR-H1 | H | 26–35 | `GFSLTGYGVN` | 10 (fixed) |
| CDR-H2 | H | 50–58 | `MIWGDGNTD` | 9 (fixed) |
| CDR-H3 | H | 98–105 | `ERDYRLDY` | **5–20** |
| CDR-L1 | L | 24–34 | `RASGNIHNYLA` | 11 (fixed) |
| CDR-L2 | L | 50–56 | `YTTTLAD` | 7 (fixed) |
| CDR-L3 | L | 89–97 | `QHFWSTPRT` | 9 (fixed) |

54 designable positions in total. Only **CDR-H3** may change length; the five
canonical loops keep their reference length, because changing a canonical loop
length changes its canonical structural class and would silently invalidate the
supplied framework geometry. CDR-H3 outside 5–20 residues is not a developable
antibody and fails the contract.

Because CDR-H3 may change length, the framework check is a *decomposition*, not
a position-wise comparison: your `sequence_H` must read as
`framework + CDR-H1 + framework + CDR-H2 + framework + CDR-H3 + framework`,
with every framework piece byte-identical to the reference.

## Required quality criteria

The scorer supplies antigen chain `T`, predicts each `H`/`L`/`T` complex, and
applies these public physical constraints:

1. **Framework preservation in 3D** — backbone RMSD over the fixed framework
   positions must be ≤ 1.5 Å after superposition on the reference Fv. A
   sequence-identical framework that refolds elsewhere is not preserved.
2. **Interface packing** — antibody↔antigen contact molecular surface must be
   `contact_ms ≥ 300`.
3. **Sterics** — the predicted complex must contain zero backbone clashes.
4. **Developability** (soft) — the CDRs should contain no `N-X-S/T` sequon
   (`X != P`), the Fv may have at most one inferred unpaired CDR cysteine, the
   combined H/L net charge must lie from −5 to +5, and the longest CDR-H3 run
   made only from `ACFILMVWY` must be ≤ 4 residues. These affect the soft-pass
   result, not the hard physical pass.
5. **High interface confidence**  — fold quality and the antibody↔antigen interaction
   are checked by an independent held-out structure-prediction oracle. Make sure to create a high confident interaction region to the best of your abilities.

## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required columns: `sequence_H` and `sequence_L`, uppercase canonical
  amino-acid letters only.
- Optional column: `design_id`, a unique non-empty identifier. The scorer
  generates deterministic row IDs when it is omitted.
- Every pair of designs must have global Needleman–Wunsch/BLOSUM62 identity
  ≤ 70%, using affine gap penalties −10/−0.5, measured on the concatenation of
  the six submitted CDR segments. The fixed framework is excluded from this
  comparison because it is intentionally identical in every row.
- Do not submit structures, coordinates, paths, the antigen, or any other
  column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate `(sequence_H, sequence_L)` pair, a
non-canonical residue, a mutated framework position, a CDR-H3 outside 5–20
residues, or a pairwise CDR identity above 70%.
