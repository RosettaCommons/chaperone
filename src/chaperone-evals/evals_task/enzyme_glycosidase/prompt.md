# Eval task: de novo retaining glycosidase

REQUIRED_N = {{REQUIRED_N}}
**TL;DR — submit exactly {{REQUIRED_N}} de novo glycosidase designs that scaffold a
six-residue catalytic theozyme, including the Glu/Glu catalytic dyad, around
a 4-methylumbelliferyl-glycoside substrate.**

You are designing a glycoside hydrolase. Unlike a binder task, the object here
is a *catalytic geometry*: two glutamate carboxylates flanking the anomeric
carbon of the substrate, with a hydrogen-bonding shell that positions the
sugar ring.

## Reference catalytic motif

`inputs/theozyme.pdb` contains the reference geometry:

- chain **A** — six catalytic/first-shell side chains (side-chain atoms only;
  the theozyme carries no backbone);
- chain **B** — the substrate, a custom ligand named `L:1` (a
  4-methylumbelliferyl β-glycoside, 22 heavy atoms).

Positions are named by their numbering in the reference enzyme. Your design
must place every one of them:

| Reference position | Residue | Role |
|---|---|---|
| `Y77` | Tyr | sugar-ring stacking / OH positioning |
| `E86` | Glu | **catalytic nucleophile** — 2.5 Å from the anomeric carbon |
| `Y88` | Tyr | second-shell hydroxyl recognition |
| `R122` | Arg | oxyanion / carboxylate pKa tuning |
| `Q136` | Gln | sugar hydroxyl hydrogen bonding |
| `E177` | Glu | **catalytic general acid/base** — 2.8 Å from the glycosidic oxygen |

`E86` and `E177` are the catalytic dyad. In the supplied theozyme their
carboxylate carbons (`CD`) sit **7.39 Å** apart and their closest carboxylate
oxygens **5.87 Å** apart — a *retaining* (double-displacement) arrangement,
not an inverting one. The scorer measures this separation explicitly in every
folded design; a design that places both glutamates but at the wrong
separation is not a glycosidase of this mechanism and does not pass.

`inputs/config.json` is the diffusion generation configuration used by the
reference implementation: it fixes those six side chains, marks the buried and
solvent-exposed faces of the substrate, and requests a 130–220 residue
scaffold.

## Why this task needs `motif_index_map`

A submission is sequences only — no coordinates. The scorer therefore cannot
tell *which* residues of your design are meant to be catalytic, and cannot
measure motif geometry or the dyad separation without that. `motif_index_map`
supplies it: for each reference position above, name the residue in **your**
sequence that plays it.

```
motif_index_map = Y77:A31;E86:A45;Y88:A47;R122:A88;Q136:A104;E177:A150
```

Encode it in CSV as `REF:RESID;REF:RESID;...` with residue IDs of the form
`A<1-based index into sequence>`. The scorer validates this before spending
any compute: every one of the six reference positions must be present, each
must map to a distinct residue inside your sequence, and the residue at that
index must be the correct amino acid (`Y`, `E`, `Y`, `R`, `Q`, `E`
respectively).

## Quality criteria

The scorer predicts each design with the supplied substrate and applies these
public physical constraints:

1. **Motif fidelity** — all-atom RMSD of the six mapped catalytic side chains
   against the reference geometry must be ≤ 1.0 Å.
2. **Catalytic dyad geometry** — the distance between the `E86` and `E177`
   carboxylate carbons must be 5.9–8.9 Å, preserving the retaining mechanism.
3. **Substrate pose** — substrate RMSD against the reference pose after
   superimposing the mapped catalytic residues must be ≤ 2.0 Å.
4. **Pocket burial** — mean relative solvent accessibility of the substrate
   atoms must be ≤ 0.15.
5. **Packing** — design↔substrate contact molecular surface must be
   `contact_ms ≥ 200`.
6. **Polar recognition** (soft) — at least 3 protein↔substrate hydrogen bonds
   are required for the soft-pass result, not the hard physical pass.
7. **High interface confidence** — fold quality and the designed interaction are checked by an independent held-out structure-prediction oracle.
    Make sure to create a high confident interaction region to the best of your abilities.

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
- Do not submit structures, coordinates, paths, the substrate, or any other
  column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, a motif map that is incomplete or lands on the wrong residue type, or
a pairwise identity above 70%.
