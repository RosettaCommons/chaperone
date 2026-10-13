# Eval task: rescaffold a hammerhead ribozyme active site

REQUIRED_N = {{REQUIRED_N}}
**TL;DR — submit exactly {{REQUIRED_N}} de novo RNA sequences that rebuild the
supplied six-nucleotide catalytic core into a single-chain ribozyme fold.**

The supplied motif, `2oeu_motif.pdb`, is the active site of the full-length
*Schistosoma mansoni* hammerhead ribozyme, **PDB 2OEU** (2.0 Å, Mn(II) bound):
six nucleotides drawn from two native strands, plus the catalytic Mn²⁺.

| id in the file | hammerhead name | base | role |
|---|---|---|---|
| `A20` | G8 | G | 2′-OH acts as general acid, protonating the leaving group |
| `A21` | A9 | A | pro-R phosphate oxygen ligates the divalent metal |
| `A22` | G10.1 | G | N7 ligates the same metal |
| `A36` | G12 | G | N1 acts as general base, deprotonating the nucleophile |
| `B6` | C17 | C | its 2′-O is the nucleophile |
| `B7` | C1.1 | C | its O5′ is the leaving group |
| `MN` | — | — | the A9 / G10.1 divalent metal |

## Why this target

The hammerhead is the most thoroughly characterised ribozyme in existence, so
the catalytic assignments are unambiguous and there is a real crystal structure
to score against rather than a model. Two things make it a genuine design
problem rather than a recall problem:

1. **The motif spans two native strands.** G8/A9/G10.1/G12 come from the enzyme
   strand and C17/C1.1 from the substrate. Rebuilding them into **one** chain
   turns a *trans*-acting ribozyme into a *cis*-acting one, so the connectivity
   between the fragments exists in no input — it has to be invented.
2. **Self-consistency is not a valid readout here.** The usual design→refold
   agreement asks whether an oracle agrees with a generator. For a catalytic
   site the only question that matters is whether the *crystal* geometry was
   reproduced, so every held-out metric is scored against 2OEU or against an
   absolute geometric target.

The site is compact — radius of gyration 7.5 Å — which is what makes
scaffolding it tractable.

## The geometry that has to survive

| measurement | native |
|---|---:|
| G12 N1 ⋯ C17 O2′ (general base) | 3.26 Å |
| G8 O2′ ⋯ C1.1 O5′ (general acid) | 3.51 Å |
| C17 O2′ ⋯ scissile P (nucleophilic attack) | 3.30 Å |
| O2′–P–O5′ (in-line attack angle) | 152.0° |
| Mn²⁺ ⋯ A9 OP2 | 2.01 Å |
| Mn²⁺ ⋯ G10.1 N7 | 2.08 Å |

An in-line arrangement of the 2′-O nucleophile, the scissile phosphorus and the
O5′ leaving group is a **necessary condition** for 2′-O transesterification. A
design that fails it cannot cleave no matter how good its RMSD is.

## Required quality criteria

The scorer supplies the Mn²⁺ itself — identical for every design — refolds each
submitted sequence, and applies these public constraints to the refold:

1. **The site is rebuilt.** All-atom RMSD of the six mapped motif residues
   against the 2OEU crystal must be ≤ 2.0 Å. The metal is excluded from this
   measurement.
2. **The attack geometry is in-line.** C17 O2′ ⋯ scissile P ≤ 4.0 Å, and the
   O2′–P–O5′ angle ≥ 140°.
3. **The fold is real.** Complex pLDDT ≥ 70.
4. **The design is novel.** Global Needleman–Wunsch identity of the **scaffold
   region** — every residue outside your motif map — against a held-out set of
   natural ribozyme sequences must be < 0.563. A memorised natural ribozyme
   scores well on every structural metric and is not a design.

Reported but not gated: the general base and general acid distances, both metal
ligation distances (the oracle does not place divalent ions reliably), global
fold similarity to the native, motif-versus-scaffold confidence, the maximum
identity between your own designs, and DSSR-classified base pairing for each of
the six mapped catalytic residues. The base-pairing vector is diagnostic until
it has been calibrated with this DSSR detector on another target.

## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence` — uppercase `ACGU` only, the complete designed
  RNA, 50 to 90 nucleotides.
- Required column: `motif_map` — a JSON object giving, for each of the six
  motif residues, the **0-based index into your sequence** where you placed it:
  `{"A20": 14, "A21": 15, "A22": 16, "A36": 46, "B6": 11, "B7": 12}`.
- Optional column: `design_id`, a unique non-empty identifier. The scorer
  generates deterministic row ids when this column is omitted.
- Do not submit structures, coordinates, paths, or any other column.

Constraints checked before any held-out compute is spent — a violation fails
the submission outright:

- exactly {{REQUIRED_N}} rows, no duplicate sequences;
- every sequence matches `^[ACGU]+$` and is 50–90 nt;
- `motif_map` names exactly the six ids above, with unique in-range indices;
- the base at each mapped index matches the motif's identity
  (`A20:G, A21:A, A22:G, A36:G, B6:C, B7:C`);
- `A20,A21,A22` are contiguous in your sequence, and `B6,B7` are contiguous —
  each is one continuous stretch in the native. The two groups, and `A36`, may
  land anywhere relative to each other;
- every pair of designs is at most 70% identical by global
  Needleman–Wunsch alignment.

The motif map is mandatory and is not recovered by alignment: a design that
mutates or mis-places the motif fails loudly rather than being silently
re-aligned to whatever fits best.
